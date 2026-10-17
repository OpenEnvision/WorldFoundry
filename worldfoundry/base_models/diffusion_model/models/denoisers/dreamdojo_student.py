# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""DreamDojo distilled student checkpoint loading and TrigFlow preconditioning.

Adapted from NVIDIA/DreamDojo at 02f119b759d5c7f84a399fdeea3c6e82e7ed6cff,
interactive/models/action_video2world_self_forcing.py and modules/denoiser_scaling.py.
"""

from __future__ import annotations

from typing import cast

import torch

from worldfoundry.core.attention.cache.kvcache import CompactingKVCache

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.cosmos2p5.action_causal import CausalActionCosmosTransformer
from .cosmos2p5 import COSMOS25_2B_CONFIG
from .dreamdojo import convert_dreamdojo_state_dict


def trigflow_scaling(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Released rectified-flow sCM coefficients for sigma_data=1, in input dtype."""
    angle = time.double()
    sine, cosine = angle.sin(), angle.cos()
    inverse = (cosine + sine).reciprocal()
    skip, out, scale, noise_time = (
        value.to(time.dtype) for value in (inverse, -sine * inverse, inverse, sine * inverse)
    )
    return skip, out, scale, noise_time


class DreamDojoStudentDenoiser:
    """Predict clean latents, not teacher velocity, with request-local KV state."""

    def __init__(self, model: CausalActionCosmosTransformer) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        context, mask, caches = (values.get(key) for key in ("context", "condition_mask", "kv_cache"))
        if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(caches, dict):
            raise TypeError("DreamDojo student requires text, a condition mask, and request-owned KV caches")
        dtype = next(self.model.parameters()).dtype
        latent = model_input.latents
        time = model_input.timestep.to(device=latent.device, dtype=dtype).reshape(latent.shape[0], 1, 1, 1, 1)
        mask = mask.to(latent)
        time = torch.atan(torch.full_like(time, 0.0001)) * mask.mean((1, 3, 4), keepdim=True) + time * (
            1 - mask.mean((1, 3, 4), keepdim=True)
        )
        skip, out, scale, noise_time = trigflow_scaling(time)
        action = values.get("action")
        prediction = (
            self.model.forward_frame(
                (latent * scale).to(dtype),
                noise_time.reshape(latent.shape[0], 1),
                context.to(device=latent.device, dtype=dtype),
                action=action.to(device=latent.device, dtype=dtype) if isinstance(action, torch.Tensor) else None,
                frame_index=int(cast(int, values["frame_index"])),
                kv_cache=cast(dict[int, CompactingKVCache], caches),
                store_kv=bool(values.get("store_kv", False)),
                condition_mask=mask,
                padding_mask=cast(torch.Tensor | None, values.get("padding_mask")),
            )
            .float()
            .to(latent.dtype)
        )
        return DenoiserOutput(sample=skip * latent + out * prediction)


def build_dreamdojo_student_denoiser(context: ComponentBuildContext) -> DreamDojoStudentDenoiser:
    """Load explicit distilled EMA weights without a teacher fallback."""
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=CausalActionCosmosTransformer,
            config={**COSMOS25_2B_CONFIG, "rope_scale": (1.0, 3.0, 3.0), "action_dim": 384},
            state_dict_converter=convert_dreamdojo_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, CausalActionCosmosTransformer):
        raise TypeError("DreamDojo student loader must build a CausalActionCosmosTransformer")
    return DreamDojoStudentDenoiser(model)
