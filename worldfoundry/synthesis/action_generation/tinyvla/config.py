"""Local Transformers configuration classes for TinyVLA checkpoints."""

from __future__ import annotations

import os

from transformers import GPTNeoXConfig, PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlavaPythiaVisionConfig(PretrainedConfig):
    model_type = "llava_pythia_clip_vision_model"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        projection_dim: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 32,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        mm_vision_select_feature: str = "patch",
        mm_vision_select_layer: int = -2,
        vision_model_name_or_path: str = "clip",
        concat: str = "None",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.vision_model_name_or_path = vision_model_name_or_path
        self.concat = concat

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> PretrainedConfig:
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["vision_config"]["vision_tower"]
        return cls.from_dict(config_dict, **kwargs)


class ProjectorConfig(PretrainedConfig):
    model_type = "llava_pythia_projector"

    def __init__(
        self,
        mm_projector_type: str = "linear",
        mm_hidden_size: int = 768,
        hidden_size: int = 2560,
        **kwargs,
    ) -> None:
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> PretrainedConfig:
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["vision_config"]["mm_projector"]
        return cls.from_dict(config_dict, **kwargs)


DEFAULT_VISUAL_CONFIG = {
    "vision_tower": LlavaPythiaVisionConfig().to_dict(),
    "mm_projector": ProjectorConfig().to_dict(),
}


class LlavaPythiaConfig(GPTNeoXConfig):
    """GPT-NeoX configuration extended with vision and action-head fields."""

    model_type = "llava_pythia"

    def __init__(self, vision_config=None, **kwargs) -> None:
        self.vision_config = DEFAULT_VISUAL_CONFIG if vision_config is None else vision_config
        self.concat = kwargs.pop("concat", "None")
        super().__init__(**kwargs)


__all__ = ["LlavaPythiaConfig", "LlavaPythiaVisionConfig", "ProjectorConfig"]
