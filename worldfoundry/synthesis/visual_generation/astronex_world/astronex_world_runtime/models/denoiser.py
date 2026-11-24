# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, models/wan_wrapper.py.
"""Build the Astronex denoiser from the released weights and run it on ``[B, F, C, H, W]`` latents."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.astronex.causal import (
    AstronexCausalWanModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.astronex.model import AstronexWanModel

# Fields of the released ``transformer/config.json`` that define the Wan2.2 backbone.
BACKBONE_FIELDS = (
    "patch_size",
    "text_len",
    "in_dim",
    "dim",
    "ffn_dim",
    "freq_dim",
    "text_dim",
    "out_dim",
    "num_heads",
    "num_layers",
    "qk_norm",
    "cross_attn_norm",
    "eps",
)


class AstronexDenoiser(nn.Module):
    """Flow predictor; ``self.model`` holds the network under the checkpoint's ``model.`` prefix."""

    def __init__(
        self,
        *,
        model_root: str,
        model_name: str = "transformer",
        is_causal: bool,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_camera: bool = False,
        use_action: bool = False,
        action_dim: int = 64,
        num_embodiments: int = 32,
        prope_chunk_anchor: bool = False,
        prope_num_layers: int | None = None,
    ):
        super().__init__()
        config = json.loads((Path(model_root) / model_name / "config.json").read_text())
        kwargs = {name: config[name] for name in BACKBONE_FIELDS}
        kwargs.update(
            use_camera=use_camera,
            prope_num_layers=prope_num_layers,
            use_action=use_action,
            action_dim=action_dim,
            num_embodiments=num_embodiments,
        )
        if is_causal:
            self.model = AstronexCausalWanModel(
                **kwargs, local_attn_size=local_attn_size, sink_size=sink_size, prope_chunk_anchor=prope_chunk_anchor
            )
        else:
            self.model = AstronexWanModel(**kwargs)
        self.model.eval()
        self.is_causal = is_causal

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int = 0,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        prope_kv_cache: list[dict] | None = None,
        prope_window_viewmats: torch.Tensor | None = None,
        prope_window_Ks: torch.Tensor | None = None,
        cache_keep_frames: list[int] | None = None,
    ) -> torch.Tensor:
        """Return the flow prediction for ``[B, F, C, H, W]`` latents at per-frame ``timestep`` ``[B, F]``."""
        latents = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        prompt_embeds = conditional_dict["prompt_embeds"]
        if not self.is_causal:
            # TI2V conditions the reference frame at t = 0; otherwise one timestep serves the clip.
            per_frame = timestep.dim() == 2 and timestep.shape[1] > 1 and (timestep[:, 1:] != timestep[:, :1]).any()
            _, frames, _, height, width = noisy_image_or_video.shape
            flow = self.model(
                latents,
                t=timestep if per_frame else timestep[:, 0],
                context=prompt_embeds,
                seq_len=frames * (height // 2) * (width // 2),
                viewmats=viewmats,
                Ks=Ks,
            )
            return flow.permute(0, 2, 1, 3, 4)

        # A full-length action stream is sliced to this chunk off ``current_start``.
        actions = conditional_dict.get("actions")
        if actions is not None and actions.shape[1] != noisy_image_or_video.shape[1]:
            patch = self.model.patch_size
            frame_tokens = (noisy_image_or_video.shape[-2] // patch[1]) * (noisy_image_or_video.shape[-1] // patch[2])
            start_frame = current_start // frame_tokens
            actions = actions[:, start_frame : start_frame + noisy_image_or_video.shape[1]]
        flow = self.model(
            latents,
            t=timestep,
            context=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            viewmats=viewmats,
            Ks=Ks,
            prope_kv_cache=prope_kv_cache,
            prope_window_viewmats=prope_window_viewmats,
            prope_window_Ks=prope_window_Ks,
            cache_keep_frames=cache_keep_frames,
            actions=actions,
            embodiment_id=conditional_dict.get("embodiment_id"),
        )
        return flow.permute(0, 2, 1, 3, 4)
