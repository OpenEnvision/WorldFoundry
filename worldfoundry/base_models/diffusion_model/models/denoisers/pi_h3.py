# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hansheng Chen.
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3:
#   lakonlab/models/architectures/diffusers/minimax_h3.py
#   lakonlab/pipelines/utils.py
"""Strict loader for ``multimodalart/MiniMax-H3-Pruned`` and ``Lakonik/pi-H3`` adapters."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..networks.minimax_h3.config import MiniMaxH3DiTArchConfig
from ..networks.minimax_h3.loading import _shard_paths
from ..networks.minimax_h3.variants._lora import LORA_SUFFIXES
from ..networks.minimax_h3.variants.pi_h3.model import (
    PI_H3_DEFAULT_DWD_HEAD_CONFIG,
    PiH3DiTModel,
)

_CONFIG_RENAME: dict[str, str] = {
    "num_refiner_layers": "token_refiner_num_layers",
    "ffn_dim": "ffn_hidden_size",
    "in_channels": "latents_dim",
    "audio_in_channels": "audio_latents_dim",
    "freq_dim": "timestep_input_dim",
    "time_embed_hidden_dim": "time_embed_hidden_size",
    "rope_freq_dim": "rope_inv_freq_len",
}

_ARCH_FIELDS = frozenset(
    {
        "num_layers",
        "token_refiner_num_layers",
        "hidden_size",
        "num_attention_heads",
        "attention_head_dim",
        "ffn_hidden_size",
        "latents_dim",
        "audio_latents_dim",
        "patch_size",
        "text_dim",
        "timestep_input_dim",
        "time_embed_hidden_size",
        "time_embed_dim",
        "rope_inv_freq_len",
        "norm_eps",
        "qk_norm_eps",
        "final_norm_eps",
    }
)


def pi_h3_arch_config(config: Mapping[str, Any]) -> MiniMaxH3DiTArchConfig:
    """Translate Diffusers/LakonLab H3 config fields into :class:`MiniMaxH3DiTArchConfig`."""

    options: dict[str, Any] = {}
    for name, value in config.items():
        if name.startswith("_"):
            continue
        mapped = _CONFIG_RENAME.get(name, name)
        if mapped in _ARCH_FIELDS:
            options[mapped] = value
    hidden = int(options.get("hidden_size", 5376))
    options["adaln_out_features"] = 18 * hidden
    options["final_adaln_out_features"] = 2 * hidden
    return MiniMaxH3DiTArchConfig(**options)


def _strip_pi_h3_prefix(key: str) -> str:
    for prefix in ("base_model.model.", "transformer.", "module."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def pi_h3_foundation_key(key: str) -> str:
    """Map one ``MiniMax-H3-Pruned`` / ``Lakonik/pi-H3`` Diffusers tensor key to :class:`PiH3DiTModel`."""

    key = _strip_pi_h3_prefix(key)
    if key.endswith(".folded_bias"):
        key = key[: -len("folded_bias")] + "linear.bias"
    if key.startswith(("dwd_norm_out.", "dwd_pixel_refiner.", "time_embedder.")) or key in (
        "adaln_basis",
        "adaln_mean",
    ):
        return key
    for source, target in (
        ("audio_proj_in.", "audio_patch_proj."),
        ("proj_in.", "video_patch_proj."),
        ("audio_proj_out.", "final_layer.audio_out."),
        ("proj_out.", "final_layer.video_out."),
        ("context_embedder.", "condition_proj."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("norm_out.norm.", "final_layer.norm."),
    ):
        if key.startswith(source):
            key = target + key[len(source) :]
            break
    for source, target in (
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
        (".attn.to_q.", ".attn.qkv_proj.q."),
        (".attn.to_k.", ".attn.qkv_proj.k."),
        (".attn.to_v.", ".attn.qkv_proj.v."),
        (".attn.to_out.0.", ".attn.out_proj."),
        (".attn.norm_q.", ".attn.q_norm."),
        (".attn.norm_k.", ".attn.k_norm."),
        (".ff.net.0.proj.", ".mlp.fc1.proj."),
        (".ff.net.2.", ".mlp.fc2."),
    ):
        key = key.replace(source, target)
    return key


def split_pi_h3_adapter_state_dict(
    raw_state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Split a pi-H3 adapter state dict into canonical non-LoRA overrides and grouped LoRA A/B pairs."""

    overrides: dict[str, torch.Tensor] = {}
    lora_groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for raw_key, tensor in raw_state_dict.items():
        stripped = _strip_pi_h3_prefix(raw_key)
        matched_lora = False
        for suffix, side in LORA_SUFFIXES:
            if stripped.endswith(suffix):
                module_diffusers = stripped[: -len(suffix)]
                module_canonical = pi_h3_foundation_key(module_diffusers + ".weight").removesuffix(".weight")
                if side in lora_groups[module_canonical]:
                    raise ValueError(f"Duplicate pi-H3 LoRA {side} tensor for {module_canonical}.")
                lora_groups[module_canonical][side] = tensor
                matched_lora = True
                break
        if matched_lora:
            continue
        if "lora" in stripped:
            raise ValueError(f"Unsupported pi-H3 LoRA tensor key: {raw_key}")
        canonical_key = pi_h3_foundation_key(stripped)
        if canonical_key in overrides:
            raise ValueError(f"Duplicate pi-H3 adapter override tensor: {canonical_key}")
        overrides[canonical_key] = tensor

    for module, pair in lora_groups.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete pi-H3 LoRA pair for {module}: {sorted(pair)}")
    return overrides, dict(lora_groups)


def _read_safetensors_source(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    if path.is_file():
        return dict(load_file(str(path), device="cpu"))
    if not path.is_dir():
        raise FileNotFoundError(f"pi-H3 checkpoint source not found: {path}")
    state: dict[str, torch.Tensor] = {}
    for shard in _shard_paths(path):
        for key, value in load_file(str(shard), device="cpu").items():
            if key in state:
                raise ValueError(f"Duplicate tensor key across shards in {path}: {key}")
            state[key] = value
    return state


def _infer_dwd_head_config(overrides: Mapping[str, torch.Tensor]) -> dict[str, Any] | None:
    has_dwd = any(k.startswith(("dwd_norm_out.", "dwd_pixel_refiner.")) for k in overrides)
    if not has_dwd:
        return None
    block_indices = {
        int(k.split(".")[2]) for k in overrides if k.startswith("dwd_pixel_refiner.blocks.") and len(k.split(".")) > 2
    }
    config = dict(PI_H3_DEFAULT_DWD_HEAD_CONFIG)
    refiner = dict(config["pixel_refiner"])
    if block_indices:
        refiner["num_layers"] = max(block_indices) + 1
    config["pixel_refiner"] = refiner
    return config


def load_pi_h3_transformer(
    base_transformer_dir: str | Path,
    adapter_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lora_scale: float = 1.0,
) -> PiH3DiTModel:
    """Strictly load ``MiniMax-H3-Pruned`` and fuse a ``Lakonik/pi-H3`` adapter.

    Args:
        base_transformer_dir: Directory containing ``MiniMax-H3-Pruned`` transformer
            ``config.json`` and safetensor shards.
        adapter_dir: Directory (or ``.safetensors`` file) of a ``Lakonik/pi-H3``
            adapter (``pi_h3_{2,4}step[_dwd]_adapter``).
        device: Target device for the loaded transformer.
        dtype: Target floating-point dtype for weights (``rope.inv_freq`` stays fp32).
        lora_scale: Multiplier applied when fusing LoRA pairs (upstream uses
            ``lora_alpha == r``, so ``1.0``).

    Returns:
        Evaluated, frozen :class:`PiH3DiTModel` with zero missing or unexpected keys.
    """

    base_path = Path(base_transformer_dir).expanduser().resolve()
    adapter_path = Path(adapter_dir).expanduser().resolve()
    base_config_path = base_path / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"MiniMax-H3-Pruned config.json not found in {base_path}")
    base_config = json.loads(base_config_path.read_text())

    adapter_config_path = (adapter_path.parent if adapter_path.is_file() else adapter_path) / "config.json"
    adapter_config = json.loads(adapter_config_path.read_text()) if adapter_config_path.is_file() else {}
    merged_config = {**base_config, **adapter_config}

    raw_base = _read_safetensors_source(base_path)
    state: dict[str, torch.Tensor] = {}
    for raw_key, tensor in raw_base.items():
        mapped = pi_h3_foundation_key(raw_key)
        if mapped in state:
            raise ValueError(f"Duplicate MiniMax-H3-Pruned mapped tensor: {mapped}")
        state[mapped] = tensor

    raw_adapter = _read_safetensors_source(adapter_path)
    overrides, lora_groups = split_pi_h3_adapter_state_dict(raw_adapter)
    state.update(overrides)

    for module_name, pair in lora_groups.items():
        weight_key = f"{module_name}.weight"
        if weight_key not in state:
            raise ValueError(f"pi-H3 LoRA target {weight_key} is missing from base/override weights.")
        lora_a, lora_b = pair["A"], pair["B"]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(f"pi-H3 LoRA tensors must be 2-D matrices for {module_name}.")
        rank_a, in_features = int(lora_a.shape[0]), int(lora_a.shape[1])
        out_features, rank_b = int(lora_b.shape[0]), int(lora_b.shape[1])
        if rank_a <= 0 or rank_a != rank_b:
            raise ValueError(
                f"pi-H3 LoRA rank mismatch for {module_name}: A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}."
            )
        base_weight = state[weight_key]
        if tuple(base_weight.shape) != (out_features, in_features):
            raise ValueError(
                f"pi-H3 LoRA shape mismatch for {module_name}: base={tuple(base_weight.shape)} "
                f"vs LoRA ({out_features}, {in_features})."
            )
        # Note: block.mlp.fc1 is HyperFlowSwiGLUProjection storing Diffusers [up|gate] order directly,
        # so ff.net.0.proj LoRA fuses without row swapping.
        delta = (lora_b.float() @ lora_a.float()) * float(lora_scale)
        state[weight_key] = (base_weight.float() + delta).to(dtype=dtype)

    dwd_head_config = merged_config.get("dwd_head_config")
    if dwd_head_config is None:
        dwd_head_config = _infer_dwd_head_config(overrides)

    adaln_rank = int(merged_config.get("adaln_rank") or 8)
    time_table_size = int(merged_config.get("time_table_size") or 1025)
    rope_theta = float(merged_config.get("rope_theta", 10000.0))
    arch = pi_h3_arch_config(merged_config)

    for key, tensor in list(state.items()):
        if tensor.is_floating_point() and tensor.dtype != dtype:
            state[key] = tensor.to(dtype=dtype)

    with torch.device("meta"):
        model = PiH3DiTModel(
            arch,
            adaln_rank=adaln_rank,
            time_table_size=time_table_size,
            rope_theta=rope_theta,
            dwd_head_config=dwd_head_config,
            dtype=dtype,
        )
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device)
    model.post_load_weights()
    return model.eval().requires_grad_(False)


__all__ = [
    "load_pi_h3_transformer",
    "pi_h3_arch_config",
    "pi_h3_foundation_key",
    "split_pi_h3_adapter_state_dict",
]
