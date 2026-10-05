"""Native wrapper around the canonical Wan CLIP encoder (Fun-Control).

:class:`CLIPModel` here delegates to :mod:`~..clip` so
Fun-Control / Video-X loaders see their expected type name.

Visual token contract is unchanged.  VAE counterpart:
:mod:`~...autoencoders.wan.variants.video_x_fun`.

Compatibility wrapper only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    clip_xlm_roberta_vit_h_14,
)
from worldfoundry.core.model_loading import load_model
from worldfoundry.core.nn import ModuleDeviceDtypeMixin


class CLIPModel(ModuleDeviceDtypeMixin, nn.Module):
    """Fun-Control name for the canonical Wan CLIP encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
        )

    def forward(self, videos: list[torch.Tensor]) -> torch.Tensor:
        size = (self.model.image_size, self.model.image_size)
        frames = torch.cat(
            [
                F.interpolate(
                    video.transpose(0, 1),
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for video in videos
            ]
        )
        frames = self.transforms.transforms[-1](frames.mul_(0.5).add_(0.5))
        # Accelerate's sequential CPU offload places module parameters on the
        # meta device between calls. Its root pre-forward hook has already
        # moved input tensors to the execution device, while ``self.device``
        # can still resolve to a meta parameter owned by a hooked child.
        model_device = self.device
        if model_device.type == "meta":
            model_device = frames.device
        return self.model.visual(
            frames.to(device=model_device, dtype=self.dtype),
            use_31_block=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        transformer_additional_kwargs: dict | None = None,
    ) -> "CLIPModel":
        del transformer_additional_kwargs
        return load_model(
            cls,
            pretrained_model_path,
            torch_dtype=torch.float32,
            device="cpu",
            state_dict_converter=lambda state: {
                (key if key.startswith("model.") else f"model.{key}"): value
                for key, value in state.items()
            },
        )


__all__ = ["CLIPModel"]
