import torch
import torch.nn as nn

from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import backbones
from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.attention import Attention
from worldfoundry.core.nn.blocks.layers import Mlp


def prepare_dinov2(model: nn.Module) -> None:
    """Keep released DINO-WM encoder pickles on DINOv2's eager attention contract."""
    for module in model.modules():
        if isinstance(module, Attention):
            module.fused_attn = False
        # Released DINOv2 pickles store one dropout; the canonical MLP names both calls.
        if isinstance(module, Mlp) and not hasattr(module, "drop1"):
            module.drop1 = module.drop
            module.drop2 = module.drop


class DinoV2Encoder(nn.Module):
    def __init__(self, name: str, feature_key: str) -> None:
        super().__init__()
        self.name = name
        self.base_model = getattr(backbones, name)()
        prepare_dinov2(self.base_model)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb
