# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025-2026 The MiniMax authors and The HuggingFace Team.
# Checkpoint mapping: huggingface/diffusers v0.40.0, convert_minimax_h3_to_diffusers.py.
"""Load HyperFlow's released H3 codecs into the canonical video/audio VAEs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..networks.minimax_h3.variants.hyperflow.conditioning import (
    HyperFlowQKVProjection,
    HyperFlowSwiGLUProjection,
)
from .minimax_h3_loading import _shard_paths, remap_weight_norm_keys

if TYPE_CHECKING:
    from torch import nn

    from .minimax_h3_audio import MiniMaxH3AudioVAE
    from .minimax_h3_video import MiniMaxH3VideoVAE


def prepare_hyperflow_video_vae(model: nn.Module) -> None:
    """Retain released unfused projection keys without copying the video VAE graph."""
    # The released checkpoint omits the unused all-zero masked-training token.
    model.decoder._non_persistent_buffers_set.add("mask_token")
    for block in model.decoder.transformer_blocks:
        block.attn.to_qkv = HyperFlowQKVProjection(block.attn.to_qkv, head_dim=block.attn.dim_head)
        block.ff.w1 = HyperFlowSwiGLUProjection(block.ff.w1)


def hyperflow_video_vae_key(key: str) -> str:
    """Map the released codec key bijectively, preserving each tensor's row order."""
    if key.startswith("encoder.down_blocks."):
        level, rest = key.removeprefix("encoder.down_blocks.").split(".", 1)
        rest = rest.replace("resnets.", "block.", 1).replace("conv_shortcut.", "nin_shortcut.", 1)
        rest = rest.replace("downsamplers.0.", "downsample.", 1)
        key = f"encoder.down.{level}.{rest}"
    for source, target in (
        ("decoder.proj_in.", "decoder.x_embedder."),
        (".attn.to_q.", ".attn.to_qkv.q."),
        (".attn.to_k.", ".attn.to_qkv.k."),
        (".attn.to_v.", ".attn.to_qkv.v."),
        (".attn.to_out.0.", ".attn.to_out."),
        (".ff.net.0.proj.", ".ff.w1.proj."),
        (".ff.net.2.", ".ff.w2."),
    ):
        key = key.replace(source, target)
    return key


def load_hyperflow_video_vae(component_dir: str | Path) -> MiniMaxH3VideoVAE:
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from .minimax_h3_video import MiniMaxH3VideoVAE
    from .minimax_h3_video.config import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig

    directory = Path(component_dir)
    config = json.loads((directory / "config.json").read_text())
    arch = MiniMaxH3VideoVAEArchConfig(latents_mean=config["latents_mean"], latents_std=config["latents_std"])
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3VideoVAE(MiniMaxH3VideoVAEConfig(arch_config=arch))
        prepare_hyperflow_video_vae(model)
    state = {}
    for shard in _shard_paths(directory):
        state.update({hyperflow_video_vae_key(key): value for key, value in load_file(str(shard)).items()})
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)


def load_hyperflow_audio_vae(component_dir: str | Path) -> MiniMaxH3AudioVAE:
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    from .minimax_h3_audio import MiniMaxH3AudioVAE
    from .minimax_h3_audio.config import MiniMaxH3AudioVAEArchConfig, MiniMaxH3AudioVAEConfig

    directory = Path(component_dir)
    config = json.loads((directory / "config.json").read_text())
    arch = MiniMaxH3AudioVAEArchConfig(latents_mean=config["latents_mean"], latents_std=config["latents_std"])
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=arch))
    state = {}
    for shard in _shard_paths(directory):
        state.update(remap_weight_norm_keys(load_file(str(shard))))
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)
