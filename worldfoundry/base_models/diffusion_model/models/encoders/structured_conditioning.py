"""Structured condition values for native model-internal forwards.

The public diffusion lifecycle uses :class:`Conditioning`.
These helpers cover networks that internally expect a typed
collection of text, media, fps, and mask tensors without
importing an upstream runtime.

:class:`GeneralConditioner` assembles :class:`TextAttr` /
:class:`ReMapkey` embeddings.  :func:`broadcast_condition`
replicates tensors under context parallel.

Not a runner :class:`~...contracts.ConditionEncoder` by
itself — Cosmo / MAGI-style internals only.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from worldfoundry.core.configuration.lazy_config import instantiate
from worldfoundry.core.distributed.model_parallel.context import broadcast
from worldfoundry.core.distributed.runtime.logging import log
from worldfoundry.core.io.assets.easy_io import easy_io
from worldfoundry.core.utils.tensors.batch import batch_mul
from worldfoundry.core.execution.inference_runtime import disabled_train


class DataType(str, Enum):
    """Whether a structured condition batch is image, video, or mixed."""
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BaseCondition(ABC):
    """Typed tensor bundle passed into native model-internal forwards."""
    _is_broadcasted: bool = False

    def to_dict(self, skip_underscore: bool = True) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not (skip_underscore and item.name.startswith("_"))
        }

    @property
    def is_broadcasted(self) -> bool:
        return self._is_broadcasted

    def broadcast(self, process_group: torch.distributed.ProcessGroup | None) -> "BaseCondition":
        return broadcast_condition(self, process_group)


def broadcast_condition(
    condition: BaseCondition,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> BaseCondition:
    """Broadcast a structured condition across context-parallel ranks."""
    if condition.is_broadcasted:
        return condition
    values = condition.to_dict(skip_underscore=False)
    for key, value in values.items():
        if value is not None:
            values[key] = broadcast(value, process_group)
    values["_is_broadcasted"] = True
    return type(condition)(**values)


@dataclass(frozen=True)
class Text2WorldCondition(BaseCondition):
    """Text-to-world structured condition (tokens, masks, fps, type)."""
    crossattn_emb: torch.Tensor | None = None
    data_type: DataType = DataType.VIDEO
    padding_mask: torch.Tensor | None = None
    fps: torch.Tensor | None = None

    def edit_data_type(self, data_type: DataType) -> "Text2WorldCondition":
        values = self.to_dict(skip_underscore=False)
        values["data_type"] = DataType(data_type)
        return type(self)(**values)

    @property
    def is_video(self) -> bool:
        return self.data_type is DataType.VIDEO


class AbstractEmbModel(nn.Module):
    """Base embedding module used by :class:`GeneralConditioner`."""
    def __init__(self) -> None:
        super().__init__()
        self._is_trainable = True
        self._dropout_rate = 0.0
        self._input_key: str | Sequence[str] | None = None
        self._is_return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @is_trainable.setter
    def is_trainable(self, value: bool) -> None:
        self._is_trainable = bool(value)

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value: float) -> None:
        self._dropout_rate = float(value)

    @property
    def input_key(self) -> str | Sequence[str] | None:
        return self._input_key

    @input_key.setter
    def input_key(self, value: str | Sequence[str] | None) -> None:
        self._input_key = value

    @property
    def is_return_dict(self) -> bool:
        return self._is_return_dict

    @is_return_dict.setter
    def is_return_dict(self, value: bool) -> None:
        self._is_return_dict = bool(value)

    def random_dropout_input(
        self,
        value: torch.Tensor,
        dropout_rate: float | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        del key
        rate = self.dropout_rate if dropout_rate is None else float(dropout_rate)
        keep = torch.bernoulli(value.new_full((value.shape[0],), 1.0 - rate))
        return batch_mul(keep, value)

    def summary(self) -> str:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        return (
            f"{type(self).__name__}: input={self.input_key}, "
            f"parameters={parameters:,}, trainable={self.is_trainable}, dropout={self.dropout_rate}"
        )


class TextAttr(AbstractEmbModel):
    """Embed a text-attribute tensor for structured conditioning."""
    def __init__(
        self,
        input_key: Sequence[str],
        dropout_rate: float = 0.0,
        use_empty_string: bool = False,
    ) -> None:
        super().__init__()
        self.input_key = list(input_key)
        self.dropout_rate = float(dropout_rate)
        self.use_empty_string = bool(use_empty_string)
        if self.use_empty_string:
            raise ValueError("use TextAttrEmptyStringDrop when empty-string dropout is required")

    def forward(self, token: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"crossattn_emb": token}

    def random_dropout_input(
        self,
        value: torch.Tensor,
        dropout_rate: float | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return value
        return super().random_dropout_input(value, dropout_rate, key)


class TextAttrEmptyStringDrop(TextAttr):
    """Text attribute embedder that can drop empty strings."""
    def __init__(
        self,
        input_key: Sequence[str],
        dropout_rate: float = 0.0,
        empty_prompt_path: str | None = None,
        credential_path: str | None = None,
    ) -> None:
        super().__init__(input_key, dropout_rate)
        self.empty_prompt_path = empty_prompt_path
        self.credential_path = credential_path
        self._empty_prompt: torch.Tensor | None = None

    def _load_empty_prompt(self) -> torch.Tensor:
        if self._empty_prompt is None:
            if not self.empty_prompt_path:
                raise ValueError("empty_prompt_path is required when text dropout is enabled")
            backend_args = None
            if self.empty_prompt_path.startswith("s3://"):
                backend_args = {"backend": "s3", "s3_credential_path": self.credential_path}
            value = easy_io.load(self.empty_prompt_path, backend_args=backend_args)
            if not isinstance(value, torch.Tensor):
                raise TypeError("empty prompt checkpoint must contain a tensor")
            self._empty_prompt = value
        return self._empty_prompt

    def random_dropout_input(
        self,
        value: torch.Tensor,
        dropout_rate: float | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return value
        rate = self.dropout_rate if dropout_rate is None else float(dropout_rate)
        if rate == 0:
            return value
        empty = self._load_empty_prompt().to(device=value.device, dtype=value.dtype)
        if empty.shape[0] == 1 and value.shape[0] != 1:
            empty = empty.expand(value.shape[0], *empty.shape[1:])
        if empty.shape != value.shape:
            raise ValueError(
                f"empty prompt shape {tuple(empty.shape)} does not match embedding shape {tuple(value.shape)}"
            )
        keep = value.new_empty((value.shape[0],)).bernoulli_(1.0 - rate)
        keep = keep.view(value.shape[0], *([1] * (value.ndim - 1)))
        return keep * value + (1.0 - keep) * empty


class ReMapkey(AbstractEmbModel):
    """Rename / copy a condition tensor under a new key."""
    def __init__(
        self,
        input_key: str,
        output_key: str | None = None,
        dropout_rate: float = 0.0,
        dtype: str | None = None,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key or input_key
        self.dropout_rate = float(dropout_rate)
        self.dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
            "int": torch.int32,
            "long": torch.int64,
        }[dtype]

    def forward(self, value: Any) -> dict[str, Any]:
        if isinstance(value, torch.Tensor) and self.dtype is not None:
            value = value.to(dtype=self.dtype)
        return {self.output_key: value}


class BooleanFlag(AbstractEmbModel):
    """Turn condition dropout into one device-local boolean flag."""

    def __init__(
        self,
        input_key: str,
        output_key: str | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key or input_key
        self.dropout_rate = float(dropout_rate)
        self.flag = torch.ones(1, dtype=torch.bool)

    def random_dropout_input(
        self,
        value: torch.Tensor,
        dropout_rate: float | None = None,
        key: str | None = None,
    ) -> torch.Tensor:
        del key
        rate = self.dropout_rate if dropout_rate is None else float(dropout_rate)
        self.flag = torch.empty(1, device=value.device).bernoulli_(1.0 - rate).bool()
        return value

    def forward(self, *_: Any, **__: Any) -> dict[str, torch.Tensor]:
        return {self.output_key: self.flag}


class GeneralConditioner(nn.Module, ABC):
    """Compose multiple embedding modules into one structured condition."""
    KEY2DIM = {"crossattn_emb": 1}

    def __init__(self, **emb_models: Any) -> None:
        super().__init__()
        self.embedders = nn.ModuleDict()
        for index, (name, config) in enumerate(emb_models.items()):
            embedder = instantiate(config)
            if not isinstance(embedder, AbstractEmbModel):
                raise TypeError(f"condition embedder {name!r} must inherit AbstractEmbModel")
            embedder.is_trainable = bool(getattr(config, "is_trainable", True))
            embedder.dropout_rate = float(getattr(config, "dropout_rate", embedder.dropout_rate))
            if not embedder.is_trainable:
                embedder.train = disabled_train
                embedder.requires_grad_(False).eval()
            log.info("Initialized condition embedder #{}-{}: {}", index, name, embedder.summary())
            self.embedders[name] = embedder

    @abstractmethod
    def forward(
        self,
        batch: Mapping[str, Any],
        override_dropout_rate: Mapping[str, float] | None = None,
    ) -> BaseCondition:
        raise NotImplementedError

    def _forward(
        self,
        batch: Mapping[str, Any],
        override_dropout_rate: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        overrides = dict(override_dropout_rate or {})
        unknown = sorted(set(overrides) - set(self.embedders))
        if unknown:
            raise KeyError(f"unknown condition embedders: {unknown}")
        output: defaultdict[str, list[Any]] = defaultdict(list)
        for name, embedder in self.embedders.items():
            context = nullcontext if embedder.is_trainable else torch.no_grad
            with context():
                keys = embedder.input_key
                if isinstance(keys, str):
                    value = embedder.random_dropout_input(batch[keys], overrides.get(name))
                    embedded = embedder(value)
                elif isinstance(keys, Sequence):
                    values = [
                        embedder.random_dropout_input(batch.get(key), overrides.get(name), key)
                        for key in keys
                    ]
                    embedded = embedder(*values)
                else:
                    raise TypeError(f"condition embedder {name!r} has invalid input_key {keys!r}")
            for key, value in embedded.items():
                output[key].append(value)
        return {
            key: values[0] if len(values) == 1 else torch.cat(values, dim=self.KEY2DIM.get(key, -1))
            for key, values in output.items()
        }

    def get_condition_uncondition(self, data_batch: Mapping[str, Any]) -> tuple[BaseCondition, BaseCondition]:
        positive = {name: 0.0 for name in self.embedders}
        negative = {
            name: 1.0 if embedder.dropout_rate > 1e-4 else 0.0
            for name, embedder in self.embedders.items()
        }
        return self(data_batch, positive), self(data_batch, negative)

    def get_condition_with_negative_prompt(
        self,
        data_batch: Mapping[str, Any],
    ) -> tuple[BaseCondition, BaseCondition]:
        positive = {name: 0.0 for name in self.embedders}
        negative = {
            name: 0.0
            if isinstance(embedder, TextAttr)
            else (1.0 if embedder.dropout_rate > 1e-4 else 0.0)
            for name, embedder in self.embedders.items()
        }
        negative_batch = copy.copy(dict(data_batch))
        if isinstance(negative_batch.get("neg_t5_text_embeddings"), torch.Tensor):
            negative_batch["t5_text_embeddings"] = negative_batch["neg_t5_text_embeddings"]
        return self(data_batch, positive), self(negative_batch, negative)


__all__ = [
    "AbstractEmbModel",
    "BaseCondition",
    "BooleanFlag",
    "DataType",
    "GeneralConditioner",
    "ReMapkey",
    "Text2WorldCondition",
    "TextAttr",
    "TextAttrEmptyStringDrop",
    "broadcast_condition",
]
