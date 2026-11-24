# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from fuyi25/H3-Tiny at a3f17f0ef85e9af5d07b2007c4b2dbd6d035e69c (Apache-2.0).
"""Bijective Diffusers-to-WorldFoundry key mapping and strict loader for H3-Tiny."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..networks.minimax_h3.config import MiniMaxH3DiTArchConfig
from ..networks.minimax_h3.variants.h3_tiny.model import H3_TINY_ENDPOINT_MODES, H3TinyDiTModel

_TOP_LEVEL_DIFFUSERS_TO_WF: tuple[tuple[str, str], ...] = (
    ("r_time_embedder.linear_1.", "time_embedder.r_time_embedder.proj_in."),
    ("r_time_embedder.linear_2.", "time_embedder.r_time_embedder.proj_out."),
    ("time_embedder.linear_1.", "time_embedder.proj_in."),
    ("time_embedder.linear_2.", "time_embedder.proj_out."),
    ("audio_proj_in.", "audio_patch_proj."),
    ("proj_in.", "video_patch_proj."),
    ("audio_proj_out.", "final_layer.audio_out."),
    ("proj_out.", "final_layer.video_out."),
    ("context_embedder.", "condition_proj."),
    ("norm_out.linear.", "final_layer.adaln_proj.linear."),
    ("norm_out.norm.", "final_layer.norm."),
    ("token_refiner.final_norm.", "token_refiner.final_norm."),
)

_TOP_LEVEL_WF_TO_DIFFUSERS: tuple[tuple[str, str], ...] = tuple(
    (target, source) for source, target in _TOP_LEVEL_DIFFUSERS_TO_WF
)

_BLOCK_DIFFUSERS_TO_WF: dict[str, str] = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight",
    "attn.to_q.weight": "attn.qkv_proj.q.weight",
    "attn.to_k.weight": "attn.qkv_proj.k.weight",
    "attn.to_v.weight": "attn.qkv_proj.v.weight",
    "attn.to_out.0.weight": "attn.out_proj.weight",
    "ff.net.0.proj.weight": "mlp.fc1.proj.weight",
    "ff.net.2.weight": "mlp.fc2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
    "adaln_proj.linear_1.weight": "adaln_proj.linear_1.weight",
    "adaln_proj.linear_1.bias": "adaln_proj.linear_1.bias",
    "adaln_proj.linear_2.weight": "adaln_proj.linear_2.weight",
    "adaln_proj.linear_2.bias": "adaln_proj.linear_2.bias",
}

_BLOCK_WF_TO_DIFFUSERS: dict[str, str] = {target: source for source, target in _BLOCK_DIFFUSERS_TO_WF.items()}

_DIFFUSERS_BLOCK_RE = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")
_WF_BLOCK_RE = re.compile(r"^(blocks|token_refiner\.blocks)\.(\d+)\.(.+)$")

_ARCH_FIELD_MAP: dict[str, str] = {
    "num_attention_heads": "num_attention_heads",
    "attention_head_dim": "attention_head_dim",
    "hidden_size": "hidden_size",
    "num_layers": "num_layers",
    "num_refiner_layers": "token_refiner_num_layers",
    "ffn_dim": "ffn_hidden_size",
    "in_channels": "latents_dim",
    "audio_in_channels": "audio_latents_dim",
    "patch_size": "patch_size",
    "text_dim": "text_dim",
    "freq_dim": "timestep_input_dim",
    "time_embed_hidden_dim": "time_embed_hidden_size",
    "time_embed_dim": "time_embed_dim",
    "rope_freq_dim": "rope_inv_freq_len",
    "norm_eps": "norm_eps",
    "qk_norm_eps": "qk_norm_eps",
    "final_norm_eps": "final_norm_eps",
}


def h3_tiny_diffusers_to_wf_key(key: str) -> str:
    """Map one Diffusers H3-Tiny tensor key onto its exact WorldFoundry key."""
    match = _DIFFUSERS_BLOCK_RE.match(key)
    if match is not None:
        group, index, rest = match.groups()
        mapped_rest = _BLOCK_DIFFUSERS_TO_WF.get(rest)
        if mapped_rest is None:
            raise ValueError(f"Unsupported H3-Tiny Diffusers block tensor key: {key!r}")
        if group == "token_refiner.refiner_blocks" and rest.startswith("adaln_proj."):
            raise ValueError(f"Token refiner blocks do not carry AdaLN weights: {key!r}")
        dst_group = "blocks" if group == "transformer_blocks" else "token_refiner.blocks"
        return f"{dst_group}.{index}.{mapped_rest}"

    for source, target in _TOP_LEVEL_DIFFUSERS_TO_WF:
        if key.startswith(source):
            suffix = key[len(source) :]
            if suffix in ("weight", "bias") and not (
                source in ("norm_out.norm.", "token_refiner.final_norm.") and suffix != "weight"
            ):
                return target + suffix
            raise ValueError(f"Unsupported H3-Tiny Diffusers parameter suffix for key: {key!r}")

    raise ValueError(f"Unsupported H3-Tiny Diffusers tensor key: {key!r}")


def h3_tiny_wf_to_diffusers_key(key: str) -> str:
    """Inverse of :func:`h3_tiny_diffusers_to_wf_key`."""
    match = _WF_BLOCK_RE.match(key)
    if match is not None:
        group, index, rest = match.groups()
        mapped_rest = _BLOCK_WF_TO_DIFFUSERS.get(rest)
        if mapped_rest is None:
            raise ValueError(f"Unsupported H3-Tiny WorldFoundry block tensor key: {key!r}")
        if group == "token_refiner.blocks" and rest.startswith("adaln_proj."):
            raise ValueError(f"Token refiner blocks do not carry AdaLN weights: {key!r}")
        dst_group = "transformer_blocks" if group == "blocks" else "token_refiner.refiner_blocks"
        return f"{dst_group}.{index}.{mapped_rest}"

    for source, target in _TOP_LEVEL_WF_TO_DIFFUSERS:
        if key.startswith(source):
            suffix = key[len(source) :]
            if suffix in ("weight", "bias") and not (
                source in ("final_layer.norm.", "token_refiner.final_norm.") and suffix != "weight"
            ):
                return target + suffix
            raise ValueError(f"Unsupported H3-Tiny WorldFoundry parameter suffix for key: {key!r}")

    raise ValueError(f"Unsupported H3-Tiny WorldFoundry tensor key: {key!r}")


def h3_tiny_arch_config(config: Mapping[str, Any]) -> tuple[MiniMaxH3DiTArchConfig, dict[str, Any]]:
    """Translate Diffusers H3-Tiny ``config.json`` / ``metadata.json`` into canonical arch + variant kwargs."""
    refiner_heads = config.get("refiner_num_attention_heads")
    if refiner_heads not in (None, config.get("num_attention_heads")):
        raise ValueError(f"refiner_num_attention_heads {refiner_heads} != num_attention_heads is not supported")

    options: dict[str, Any] = {}
    for src_name, dst_name in _ARCH_FIELD_MAP.items():
        if src_name in config and config[src_name] is not None:
            options[dst_name] = config[src_name]
    if "patch_size" in options:
        options["patch_size"] = tuple(int(v) for v in options["patch_size"])
    hidden = int(options.get("hidden_size", 5376))
    options["adaln_out_features"] = 18 * hidden
    options["final_adaln_out_features"] = 2 * hidden
    arch = MiniMaxH3DiTArchConfig(**options)

    variant_kwargs: dict[str, Any] = {
        "refiner_ffn_hidden_size": (None if config.get("refiner_ffn_dim") is None else int(config["refiner_ffn_dim"])),
        "adaln_rank": None if config.get("adaln_rank") is None else int(config["adaln_rank"]),
        "rope_theta": float(config.get("rope_theta", 10000.0)),
    }
    return arch, variant_kwargs


def _safetensors_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = path / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            return [path / filename for filename in sorted(set(weight_map.values()))]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found in {path}")
    return shards


def _resolve_checkpoint_layout(weights_path: str | Path) -> tuple[dict[str, Any], list[Path]]:
    """Resolve either a ``metadata.json`` + payload export or a Diffusers checkpoint dir."""
    weights = Path(weights_path).expanduser()
    metadata_file = weights / "metadata.json"
    if metadata_file.is_file():
        meta = json.loads(metadata_file.read_text())
        if "model_config" not in meta or "payload_file" not in meta:
            raise ValueError(f"Invalid H3-Tiny metadata.json in {weights}: requires model_config and payload_file")
        payload = weights / meta["payload_file"]
        if not payload.is_file():
            raise FileNotFoundError(f"H3-Tiny payload safetensors not found: {payload}")
        return dict(meta["model_config"]), [payload]

    root = next((weights / sub for sub in ("transformer", "transformer_ref") if (weights / sub).is_dir()), weights)
    config_file = root / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"H3-Tiny checkpoint config.json not found in {root}")
    raw_config = json.loads(config_file.read_text())
    return dict(raw_config), _safetensors_files(root)


def read_h3_tiny_config(weights_path: str | Path) -> dict[str, Any]:
    """Read the raw H3-Tiny model config from ``metadata.json`` or ``config.json``."""
    raw_config, _ = _resolve_checkpoint_layout(weights_path)
    return raw_config


def _read_header_keys(files: list[Path]) -> set[str]:
    keys: set[str] = set()
    for file in files:
        with open(file, "rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
        keys.update(k for k in header if k != "__metadata__")
    return keys


def detect_h3_tiny_endpoint_mode(weights_path: str | Path) -> str:
    """Return ``"meanflow"`` if the checkpoint stores ``r_time_embedder.*`` tensors, else ``"none"``."""
    _, files = _resolve_checkpoint_layout(weights_path)
    keys = _read_header_keys(files)
    return "meanflow" if any(k.startswith("r_time_embedder.") for k in keys) else "none"


def load_h3_tiny_transformer(
    weights_path: str | Path,
    *,
    endpoint_mode: str | None = None,
    device: str | torch.device = "cpu",
) -> H3TinyDiTModel:
    """Strictly load an H3-Tiny transformer checkpoint preserving exact stored dtypes.

    Supports both sharded/single-file Diffusers directories (``config.json`` +
    safetensors/index, directly or under ``transformer`` / ``transformer_ref``) and
    student ``metadata.json`` + payload checkpoints. Auto-detects ``endpoint_mode``
    from ``r_time_embedder.*`` header presence when ``endpoint_mode`` is ``None``.
    """
    from safetensors import safe_open

    raw_config, files = _resolve_checkpoint_layout(weights_path)
    if endpoint_mode is None:
        header_keys = _read_header_keys(files)
        resolved_mode = "meanflow" if any(k.startswith("r_time_embedder.") for k in header_keys) else "none"
    else:
        if endpoint_mode not in H3_TINY_ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {H3_TINY_ENDPOINT_MODES}, got {endpoint_mode!r}")
        resolved_mode = endpoint_mode

    arch, variant_kwargs = h3_tiny_arch_config(raw_config)
    with torch.device("meta"):
        model = H3TinyDiTModel(
            arch,
            endpoint_mode=resolved_mode,
            **variant_kwargs,
        )

    mapped_state: dict[str, torch.Tensor] = {}
    target_device = str(torch.device(device))
    for shard in files:
        with safe_open(str(shard), framework="pt", device=target_device) as handle:
            for src_key in list(handle.keys()):
                dst_key = h3_tiny_diffusers_to_wf_key(src_key)
                if dst_key in mapped_state:
                    raise ValueError(f"Duplicate H3-Tiny tensor after mapping: {dst_key!r} (from {src_key!r})")
                mapped_state[dst_key] = handle.get_tensor(src_key)

    expected = dict(model.state_dict())
    missing = sorted(set(expected) - set(mapped_state))
    unexpected = sorted(set(mapped_state) - set(expected))
    shape_mismatches = [
        (key, tuple(mapped_state[key].shape), tuple(expected[key].shape))
        for key in sorted(set(mapped_state) & set(expected))
        if tuple(mapped_state[key].shape) != tuple(expected[key].shape)
    ]
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"H3-Tiny weights {weights_path} do not match model: "
            f"missing={missing[:8]} ({len(missing)}), "
            f"unexpected={unexpected[:8]} ({len(unexpected)}), "
            f"shapes={shape_mismatches[:5]}"
        )

    model.load_state_dict(mapped_state, strict=True, assign=True)
    model.rebuild_rope(device=device)
    leftover_meta = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor is not None and tensor.is_meta
    ]
    if leftover_meta:
        raise RuntimeError(f"Meta tensors left after H3-Tiny load: {leftover_meta}")
    model.post_load_weights()
    return model.eval().requires_grad_(False)


__all__ = [
    "detect_h3_tiny_endpoint_mode",
    "h3_tiny_arch_config",
    "h3_tiny_diffusers_to_wf_key",
    "h3_tiny_wf_to_diffusers_key",
    "load_h3_tiny_transformer",
    "read_h3_tiny_config",
]
