"""T2V-Turbo :class:`~...contracts.ConditionEncoder` over LVDM OpenCLIP.

Loads ``cond_stage_model.*`` from a VideoCrafter checkpoint
and encodes prompts to ``context``.  Also exports
:func:`guidance_embedding` for LCM guidance-scale sinusoidal
features used by the turbo denoiser.

:class:`T2VTurboConditioner` is the runner-facing adapter.
OpenCLIP graph internals live in :mod:`.lvdm.condition`.

This is the LVDM / VideoCrafter family, not Wan UMT5.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from ...components import ComponentBuildContext
from ...contracts import Conditioning, DiffusionRequest
from ...loaders import ModuleLoadSpec, NativeModuleLoader


def _conditioner_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    prefix = "cond_stage_model."
    converted = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not converted:
        raise KeyError("VideoCrafter checkpoint contains no cond_stage_model parameters")
    return converted


def guidance_embedding(
    values: torch.Tensor,
    *,
    embedding_dim: int = 256,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode LCM guidance values using the VDM sinusoidal convention."""

    if values.ndim != 1:
        raise ValueError("guidance values must be a one-dimensional tensor")
    scaled = values.to(dtype=dtype) * 1000.0
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        torch.arange(half_dim, device=values.device, dtype=dtype)
        * (-math.log(10000.0) / (half_dim - 1))
    )
    angles = scaled[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding_dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class T2VTurboConditioner:
    """Encode prompts to LVDM ``context`` tokens for T2V-Turbo."""
    def __init__(self, text_encoder: Any) -> None:
        self.text_encoder = text_encoder

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        if request.negative_prompt is not None:
            raise ValueError("T2V-Turbo uses distilled guidance and does not accept negative_prompt")
        if bool(request.inputs.get("use_motion_cond", False)):
            raise ValueError("the VC2 T2V-Turbo checkpoint does not expose motion conditioning")
        text = self.text_encoder(list(request.prompts)).to(device=device, dtype=dtype)
        values = torch.full(
            (request.batch_size,),
            float(request.sampling.guidance_scale),
            device=device,
            dtype=torch.float32,
        )
        return Conditioning(
            positive={
                "context": text,
                "timestep_cond": guidance_embedding(values, dtype=dtype),
                "fps": int(request.inputs.get("fps", 16)),
            }
        )


def build_t2v_turbo_conditioner(context: ComponentBuildContext) -> T2VTurboConditioner:
    """Load the VideoCrafter OpenCLIP tower as a T2V-Turbo conditioner."""
    from .lvdm.condition import FrozenOpenCLIPEmbedder

    encoder = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=FrozenOpenCLIPEmbedder,
            config={
                "freeze": True,
                "layer": "penultimate",
                "device": str(context.policy.device),
            },
            state_dict_converter=_conditioner_state_dict,
        ),
        context.require_checkpoint("base"),
        context.policy,
    )
    if not isinstance(encoder, FrozenOpenCLIPEmbedder):
        raise TypeError(f"expected FrozenOpenCLIPEmbedder, got {type(encoder).__name__}")
    return T2VTurboConditioner(encoder)


__all__ = [
    "T2VTurboConditioner",
    "build_t2v_turbo_conditioner",
    "guidance_embedding",
]
