"""Checkpoint loading helpers for the MiniMax H3 VAEs.

The published MiniMax H3 checkpoint stores weight-normalized convolutions with
the legacy ``weight_g`` / ``weight_v`` names, while modern PyTorch
``torch.nn.utils.parametrizations.weight_norm`` registers the same tensors as
``parametrizations.weight.original0`` / ``original1``. The two describe
identical math (magnitude ``g`` + direction ``v``); only the key names differ.

These helpers remap the checkpoint keys and load one or more safetensors shards
into a ported VAE module. Verified against the real ``MiniMaxAI/MiniMax-H3``
audio VAE (1087 keys, 0 missing / 0 unexpected after remap).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def remap_weight_norm_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename legacy ``weight_g``/``weight_v`` to the parametrization form."""

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            out[key[: -len(".weight_g")] + ".parametrizations.weight.original0"] = value
        elif key.endswith(".weight_v"):
            out[key[: -len(".weight_v")] + ".parametrizations.weight.original1"] = value
        else:
            out[key] = value
    return out


def load_safetensors_shards(paths: Iterable[str | Path]) -> dict[str, torch.Tensor]:
    """Load and merge one or more safetensors files into a single state dict."""

    from safetensors.torch import load_file

    merged: dict[str, torch.Tensor] = {}
    for path in paths:
        merged.update(load_file(str(path)))
    return merged


def _shard_paths(component_dir: Path) -> list[Path]:
    """Resolve the safetensors shard(s) in a diffusers-style component dir."""

    index = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        return [component_dir / name for name in names]
    shards = sorted(component_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found in {component_dir}")
    return shards


def load_minimax_h3_vae_weights(
    model: torch.nn.Module,
    component_dir: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a MiniMax H3 VAE checkpoint directory into ``model``.

    Handles single- or multi-shard safetensors plus the ``weight_g``/``weight_v``
    → parametrization remap. Returns ``(missing_keys, unexpected_keys)``.
    """

    component_dir = Path(component_dir)
    raw = load_safetensors_shards(_shard_paths(component_dir))
    converted = remap_weight_norm_keys(raw)
    result = model.load_state_dict(converted, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


__all__ = [
    "load_minimax_h3_vae_weights",
    "load_safetensors_shards",
    "remap_weight_norm_keys",
]
