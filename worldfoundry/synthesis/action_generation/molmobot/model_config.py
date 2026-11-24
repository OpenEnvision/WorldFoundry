import dataclasses
import typing
from collections.abc import Callable
from typing import cast

import numpy as np
from omegaconf import OmegaConf as om

from .config import BaseConfig
from .io import PathOrStr, read_file
from .modeling.model import ModelBase


def get_model_types() -> dict[str, type["BaseModelConfig"]]:
    """Return the action-policy model types supported by this inference runtime."""
    # Imported lazily to avoid the BaseModelConfig/MolmoBotConfig cycle.
    from .modeling.molmobot import MolmoBotConfig

    return {
        MolmoBotConfig._model_name: MolmoBotConfig,
        "molmoact": MolmoBotConfig,
    }


@dataclasses.dataclass
class BaseModelConfig(BaseConfig):
    """Base class for Model configs"""

    _model_name: typing.ClassVar[str]
    """
    Unique name of the model
    """

    model_name: str = dataclasses.field(init=False)
    """
    Unique name to used to identify the subclass when loading configs, should mirror the
    class variable `_model_name`. We duplicate it as a field so OmegaConf saves it
    """

    def __post_init__(self):
        self.model_name = self._model_name

    def build_preprocessor(
        self,
        for_inference,
        is_training=False,
        max_text_len: int | None = None,
        max_seq_len: int | None = None,
    ) -> Callable[[dict, np.random.RandomState], dict]:
        """
        Build a preprocessor that processes individual examples

        :param for_inference: If the examples will be used for inference
        :param is_training: Retained for checkpoint-config API compatibility; inference passes False.
        """
        raise NotImplementedError()

    def build_collator(self, shapes_to_pad_to, pad_mode: str, include_metadata=True) -> Callable[[list[dict]], dict]:
        """
        Build a collator to build batches from preprocessed examples,

        :param shapes_to_pad_to:
        :param pad_mode: How to truncate/pad data
        :param include_metadata: If batch should include non-tensor metadata field
                                 (usually only used or evaluation)
        """
        raise NotImplementedError()

    def build_model(self, device=None) -> ModelBase:
        """
        Build a model that takes batches from the collator as input
        """
        raise NotImplementedError()

    @classmethod
    def get_default_model_name(cls):
        return "molmobot"

    @classmethod
    def load(
        cls,
        path: PathOrStr,
        overrides: list[str] | None = None,
        key: str | None = None,
        validate_paths: bool = True,
        override_model_name: str | None = None,
    ) -> "BaseModelConfig":
        """Load only the model section needed by the in-tree inference runtime."""
        del validate_paths
        raw = om.create(read_file(path))
        if key is not None:
            raw = raw[key]

        # Manually resolve the correct subclass using `model_name`
        model_name = override_model_name if override_model_name else raw.get("model_name", cls.get_default_model_name())
        model_types = get_model_types()
        if model_name not in model_types:
            raise ValueError(f"Unknown model type {model_name}")
        model_cls = model_types[model_name]
        schema = om.structured(model_cls)
        raw = model_cls.update_legacy_settings(raw)
        conf = om.merge(schema, raw)
        if overrides:
            conf.merge_with_dotlist(overrides)
        return cast(model_cls, om.to_object(conf))
