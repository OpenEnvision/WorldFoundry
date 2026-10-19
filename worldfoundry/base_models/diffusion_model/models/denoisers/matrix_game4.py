# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Released Matrix-Game 4 checkpoint and low-poly conditioning adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from worldfoundry.core.attention.cache.causal_cache import (
    CausalVideoCacheGeometry,
    allocate_causal_video_cache,
)

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ...schedulers.wan import InferenceFlowMatchScheduler
from ..networks.wan.variants.matrix_game4 import MatrixGame4Model

MATRIX_GAME4_CONFIG = {
    "patch_size": (1, 2, 2),
    "text_len": 512,
    "in_dim": 148,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "local_attn_size": 21,
    "sink_size": 1,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}


def convert_matrix_game4_state_dict(state: Mapping[str, object]) -> Mapping[str, object]:
    """Remove the published wrapper prefix and the unused base-model ref_conv."""
    return {
        name.removeprefix("model."): value
        for name, value in state.items()
        if not name.removeprefix("model.").startswith("ref_conv.")
    }


class MatrixGame4Denoiser:
    """Adapt BCTHW latents to control concatenation and distilled x0 prediction."""

    block_size = 1
    variant = "causal-control"

    def __init__(self, model: MatrixGame4Model) -> None:
        self.model = model
        self.flow_schedule = InferenceFlowMatchScheduler(
            num_inference_steps=1000, shift=5.0, sigma_min=0.0, extra_one_step=True
        )

    def create_kv_cache(
        self,
        *,
        batch_size: int,
        n_views: int,
        latent_frames_per_view: int,
        frame_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        if batch_size != 1 or n_views != 1:
            raise ValueError("Matrix-Game 4 supports one video per request")
        if latent_frames_per_view > self.model.max_rope_frames:
            raise ValueError("Matrix-Game 4 video exceeds the 8192-latent positional budget")
        cache = cast(
            dict[str, Any],
            allocate_causal_video_cache(
                CausalVideoCacheGeometry(
                    batch_size=batch_size,
                    total_frames=latent_frames_per_view,
                    frame_tokens=frame_sequence_length,
                    frames_per_block=1,
                    num_layers=len(self.model.blocks),
                    num_heads=self.model.num_heads,
                    head_dim=self.model.dim // self.model.num_heads,
                    local_attention_frames=self.model.local_attn_size,
                    sink_frames=1,
                ),
                device=device,
                dtype=dtype,
            ),
        )
        for layer in cache["kv_cache"]:
            layer["k_sink_raw"] = torch.zeros_like(layer["k"][:, :frame_sequence_length])
        return cache

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = cast(Mapping[str, Any], model_input.conditioning)
        noisy = model_input.latents
        control = values.get("control_latents")
        commit = model_input.branch.endswith("cache-commit")
        if commit:
            condition = torch.zeros_like(noisy)
        else:
            if not isinstance(control, Tensor) or control.shape != noisy.shape:
                raise ValueError("Matrix-Game 4 requires one aligned 48-channel control latent")
            condition = control.to(noisy)
        unused_channels = self.model.in_dim - 2 * noisy.shape[1]
        packed = torch.cat(
            (noisy, condition, noisy.new_zeros(noisy.shape[0], unused_channels, *noisy.shape[2:])),
            dim=1,
        )
        cache = values["kv_cache"]
        timestep = model_input.timestep.float().reshape(1, 1)
        flow = self.model(
            packed,
            t=timestep,
            context=values["context"],
            seq_len=noisy.shape[-2] * noisy.shape[-1] // 4,
            kv_cache=cache["kv_cache"],
            crossattn_cache=cache["crossattn_cache"],
            current_start=int(values["current_start"]),
        )
        # Clean cache commits use terminal sigma zero; sampled predictions
        # use the published CPU FP32 grid, promoted before x0 conversion.
        clean = noisy.clone() if commit else self.flow_schedule.flow_to_x0(flow, noisy, timestep.flatten())
        return DenoiserOutput(clean)


def build_matrix_game4_denoiser(context: ComponentBuildContext) -> MatrixGame4Denoiser:
    """Load the causal Wan graph through the canonical module loader."""
    overrides = cast(Mapping[str, Any], context.component_options.get("model_config", {}))
    config = {**MATRIX_GAME4_CONFIG, **overrides}
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=MatrixGame4Model,
            config=config,
            state_dict_converter=convert_matrix_game4_state_dict,
            layer_container="blocks",
        ),
        context.require_checkpoint(),
        context.policy,
    )
    return MatrixGame4Denoiser(cast(MatrixGame4Model, model))
