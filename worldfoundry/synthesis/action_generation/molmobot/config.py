from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import asdict
from enum import Enum
from typing import Any, TypeVar, cast

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

from .exceptions import OLMoConfigurationError
from .io import PathOrStr, read_file

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


@dataclasses.dataclass
class BaseConfig:
    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """
        Update the legacy config settings whose schemas have undergone backwards-incompatible changes.
        """
        return config

    @classmethod
    def new(cls: type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(e)  # noqa: B904 - Keeps the implicit exception context.

    @classmethod
    def load(
        cls: type[C],
        path: PathOrStr,
        overrides: list[str] | None = None,
        key: str | None = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        schema = om.structured(cls)
        try:
            raw = om.create(read_file(path))

            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(e)  # noqa: B904 - Keeps the implicit exception context.

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Iterable[str] | None = None) -> dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def __setattr__(self, key, value):
        # Make it impossible to add new attributes on accident, such as by mistyping a field name
        if any(key == f.name for f in dataclasses.fields(self)):
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"Cannot add new attribute '{key}'")


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{self!s}'"


class DType(StrEnum):
    """
    An enumeration of supported PyTorch data types.
    """

    float32 = "float32"
    bfloat16 = "bfloat16"
    float16 = "float16"

    @classmethod
    def from_pt(cls, dtype: torch.dtype) -> DType:
        if dtype == torch.float32:
            return DType.float32
        elif dtype == torch.bfloat16:
            return DType.bfloat16
        elif dtype == torch.float16:
            return DType.float16
        else:
            raise NotImplementedError(dtype)

    def as_pt(self) -> torch.dtype:
        return getattr(torch, self)


class TransformerDataParallelWrappingStrategy(StrEnum):
    """
    An enumeration of the different wrapping strategy for the data parallel implementations.
    """

    full = "full"
    """
    Wrap each block and the LM head (only applies to FSDP).
    """

    blocks = "blocks"
    """
    Like full but the LM head is not wrapped separately (only applies to FSDP).
    """

    fine_grained = "fine_grained"
    """
    Wrap certain modules within each block in addition to wrapping each block (only applies to FSDP).
    """
