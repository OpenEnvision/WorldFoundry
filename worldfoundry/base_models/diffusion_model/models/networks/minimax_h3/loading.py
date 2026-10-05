"""Checkpoint loading for the MiniMax H3 DiT.

The published ``MiniMaxAI/MiniMax-H3`` transformer stores the fused
``blocks.N.attn.qkv_proj.weight`` in **grouped per-head** ``[q, k, v]`` order
(one head's q, k, v contiguous, repeated per head). This port — like SGLang —
stores the fused projection as ``[q_all, k_all, v_all]``, so the qkv rows must
be reordered at load time. Every other key matches the module names directly
(verified: 535/535 keys, 0 missing / 0 unexpected).

The checkpoint also carries the mixed fp32/bf16 split (patch projections, time
embedder, output heads, ``rope.inv_freq`` in fp32; blocks in bf16), which is
loaded verbatim and re-checked by ``MiniMaxH3DiTModel.post_load_weights``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import MiniMaxH3DiTArchConfig
from .dit import reorder_grouped_qkv_to_qkv

_QKV_SUFFIX = ".attn.qkv_proj.weight"


def _shard_paths(component_dir: Path) -> list[Path]:
    """Resolve safetensor shard paths from an index JSON or a directory glob."""

    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = component_dir / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            return [component_dir / name for name in sorted(set(weight_map.values()))]
    shards = sorted(component_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found in {component_dir}")
    return shards


def convert_minimax_h3_dit_state_dict(
    state_dict: dict[str, torch.Tensor],
    arch: MiniMaxH3DiTArchConfig,
) -> dict[str, torch.Tensor]:
    """Reorder grouped per-head qkv rows to ``[q_all, k_all, v_all]``."""

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(_QKV_SUFFIX):
            out[key] = reorder_grouped_qkv_to_qkv(
                value,
                num_query_groups=arch.num_attention_heads,
                heads_per_group=1,
                head_dim=arch.attention_head_dim,
            )
        else:
            out[key] = value
    return out


def load_minimax_h3_dit_weights(
    model: torch.nn.Module,
    component_dir: str | Path,
    arch: MiniMaxH3DiTArchConfig | None = None,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a MiniMax H3 transformer checkpoint directory into ``model``.

    Applies the grouped-qkv reorder, loads (multi-)shard safetensors preserving
    the on-disk dtypes, and runs ``post_load_weights`` to assert the fp32/bf16
    split. Returns ``(missing_keys, unexpected_keys)``.
    """

    from safetensors.torch import load_file

    component_dir = Path(component_dir)
    arch = arch if arch is not None else getattr(model, "arch", MiniMaxH3DiTArchConfig())
    raw: dict[str, torch.Tensor] = {}
    for shard in _shard_paths(component_dir):
        raw.update(load_file(str(shard)))
    converted = convert_minimax_h3_dit_state_dict(raw, arch)
    result = model.load_state_dict(converted, strict=strict, assign=True)
    if hasattr(model, "post_load_weights"):
        model.post_load_weights()
    return list(result.missing_keys), list(result.unexpected_keys)


__all__ = [
    "convert_minimax_h3_dit_state_dict",
    "load_minimax_h3_dit_weights",
]
