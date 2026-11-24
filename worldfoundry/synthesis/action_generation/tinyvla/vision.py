"""Vision towers and projector used by TinyVLA checkpoints."""

from __future__ import annotations

import re

from torch import nn
from transformers import CLIPPreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers.models.siglip import SiglipPreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

from .config import LlavaPythiaVisionConfig


class _VisionTowerMixin:
    remove_class_token = False

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def feature_select(self, output):
        features = output.hidden_states[self.config.mm_vision_select_layer]
        if self.config.mm_vision_select_feature == "patch":
            return features[:, 1:] if self.remove_class_token else features
        if self.config.mm_vision_select_feature == "cls_patch":
            return features
        raise ValueError(f"unexpected vision feature selection: {self.config.mm_vision_select_feature}")

    def forward(self, images):
        if isinstance(images, list):
            result = []
            for image in images:
                output = self.vision_model(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                result.append(self.feature_select(output).to(image.dtype))
            return result
        output = self.vision_model(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        return self.feature_select(output).to(images.dtype)

    @property
    def dtype(self):
        return next(self.vision_model.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_model.parameters()).device


class CLIPVisionTower(_VisionTowerMixin, CLIPPreTrainedModel):
    config_class = LlavaPythiaVisionConfig
    remove_class_token = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        self.post_init()


class SiglipVisionTower(_VisionTowerMixin, SiglipPreTrainedModel):
    config_class = LlavaPythiaVisionConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vision_model = SiglipVisionTransformer(config)
        self.post_init()


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        del args, kwargs
        return x


class SimpleResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        return x + self.proj(self.pre_norm(x))


def build_vision_projector(config):
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if match:
        depth = int(match.group(1))
        modules: list[nn.Module] = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.extend((nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)))
        return nn.Sequential(*modules)
    if projector_type == "identity":
        return IdentityMap()
    raise ValueError(f"unknown vision projector type: {projector_type}")


__all__ = ["CLIPVisionTower", "SiglipVisionTower", "build_vision_projector"]
