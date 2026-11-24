# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Danze Chen. Adapted from Danzer1xxxxChan/H3-World, revision
# f0c7be2acbbde8256b2473f6e7b088f58272acae (Apache-2.0).
"""H3-World LoRA adapter loading and directed per-latent attention routing.

H3-World trains a rank-32 LoRA on MiniMax-H3's attention ``qkv_proj`` and
``out_proj`` projections (both ``blocks`` and ``token_refiner.blocks``) against
DiffSynth's grouped per-head QKV layout, paired with a directed attention mask
that binds each per-latent action sentence to its corresponding video latent.

This module adapts both contracts onto WorldFoundry's canonical
:class:`MiniMaxH3DiTModel` without duplicating the transformer:

* ``load_h3_world_adapter`` reorders ``qkv_proj`` LoRA outputs from grouped
  ``[num_heads, 3, head_dim]`` to WorldFoundry's ``[q_all, k_all, v_all]`` via
  :func:`reorder_grouped_qkv_to_qkv` and merges the deltas in-place.
* ``install_h3_world_directed_attention`` attaches an instance-local
  ``attention_fn`` to the DiT blocks (leaving ``token_refiner`` untouched,
  whose isolation uses segmented ``refiner_cu_seqlens``). On CUDA it uses
  compiled ``flex_attention`` over persistent per-segment index buffers; on CPU
  or fallback paths it applies the exact same directed mask via SDPA.
"""

from __future__ import annotations

import itertools
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MiniMaxH3DiTArchConfig
from ..dit import _varlen_attention, reorder_grouped_qkv_to_qkv
from ._lora import read_lora_groups, validate_lora_pair, validate_lora_scale

_H3_WORLD_TARGET_SUFFIXES = (".attn.qkv_proj", ".attn.out_proj")
_H3_WORLD_STATE_ATTR = "_h3_world_attention_state"

# Persistent per-(device, start, length) buffers for compiled flex_attention.
# Reusing both the tensor storage and the mask_mod closure avoids Dynamo
# recompiles and prevents stale mask_mod captures across requests.
_MASK_BUFFERS: dict[tuple[str, int, int], dict[str, Any]] = {}
_COMPILED_FLEX_ATTENTION: Any = None
_FLEX_ATTENTION_UNAVAILABLE = False


@dataclass(frozen=True)
class H3WorldActionMaskSpec:
    """Per-branch directed attention specification for H3-World."""

    action_text_rows: torch.Tensor
    video_start: int
    frame_rows: int


class H3WorldAttentionState:
    """Shared request-scoped state consumed by DiT blocks during a forward pass."""

    def __init__(self, *, use_flex: bool = True) -> None:
        self.use_flex = use_flex
        self.spec: H3WorldActionMaskSpec | None = None
        self._cached_key: tuple[str, tuple[int, ...], bool] | None = None
        self._cached_masks: dict[int, Any] | None = None

    def clear(self) -> None:
        self.spec = None
        self._cached_key = None
        self._cached_masks = None

    def set_spec(self, spec: H3WorldActionMaskSpec | None) -> None:
        self.spec = spec
        self._cached_key = None
        self._cached_masks = None


def _get_compiled_flex_attention() -> Any | None:
    """Return a lazily compiled ``flex_attention`` callable on CUDA, or None."""

    global _COMPILED_FLEX_ATTENTION, _FLEX_ATTENTION_UNAVAILABLE
    if _FLEX_ATTENTION_UNAVAILABLE:
        return None
    if _COMPILED_FLEX_ATTENTION is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention as flex_attention_func

            # Avoid mode="max-autotune-no-cudagraphs": on SM90/H200 the
            # BLOCK_M=BLOCK_N=128 candidate exceeds hardware shared memory and
            # can poison the CUDA context during autotune benchmarking.
            _COMPILED_FLEX_ATTENTION = torch.compile(
                flex_attention_func,
                dynamic=False,
                fullgraph=True,
                backend="inductor",
            )
        except Exception:
            _FLEX_ATTENTION_UNAVAILABLE = True
            return None
    return _COMPILED_FLEX_ATTENTION


def _mask_buffers(device: torch.device, start: int, length: int) -> dict[str, Any]:
    key = (str(device), int(start), int(length))
    buf = _MASK_BUFFERS.get(key)
    if buf is None:
        ann = torch.full((length,), -1, dtype=torch.int32, device=device)
        frm = torch.full((length,), -1, dtype=torch.int32, device=device)

        def mask_mod(b: Any, h: Any, q: Any, kv: Any, ann: torch.Tensor = ann, frm: torch.Tensor = frm) -> Any:
            del b, h
            aq, akv, fq, fkv = ann[q], ann[kv], frm[q], frm[kv]
            same_ann = (aq >= 0) & (akv >= 0) & (aq == akv)
            v_reads_own = (fq >= 0) & (akv >= 0) & (fq == akv)
            leak_out = (akv >= 0) & ~same_ann & ~v_reads_own
            leak_in = (aq >= 0) & (fkv >= 0) & (aq != fkv)
            return ~(leak_out | leak_in)

        buf = {"ann": ann, "frm": frm, "mask_mod": mask_mod}
        _MASK_BUFFERS[key] = buf
    return buf


def _populate_segment_indices(
    *,
    ann: torch.Tensor,
    frm: torch.Tensor,
    spec: H3WorldActionMaskSpec,
    start: int,
    length: int,
) -> tuple[bool, bool]:
    """Fill segment-local annotation/frame ownership vectors in-place."""

    ann.fill_(-1)
    frm.fill_(-1)

    action_rows = spec.action_text_rows.detach().cpu().tolist()
    latent_t = len(action_rows)
    any_ann = False
    for k, (row_lo, row_hi) in enumerate(action_rows):
        lo = max(int(row_lo) - start, 0)
        hi = min(int(row_hi) - start, length)
        if hi > lo:
            ann[lo:hi] = k
            any_ann = True

    vlo = int(spec.video_start) - start
    vhi = vlo + latent_t * int(spec.frame_rows)
    any_frm = bool(vhi > 0 and vlo < length)
    if any_frm:
        lo, hi = max(vlo, 0), min(vhi, length)
        idx = torch.arange(lo, hi, device=frm.device, dtype=torch.int32)
        frm[lo:hi] = (idx + (start - int(spec.video_start))) // int(spec.frame_rows)
    return any_ann, any_frm


def build_h3_world_sdpa_mask(
    spec: H3WorldActionMaskSpec,
    *,
    start: int,
    length: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the exact boolean directed-attention mask ``[1, 1, L, L]`` for one segment.

    Returns ``None`` when the segment contains no active annotation-video bindings.
    """

    ann = torch.empty((length,), dtype=torch.int32, device=device)
    frm = torch.empty((length,), dtype=torch.int32, device=device)
    any_ann, any_frm = _populate_segment_indices(ann=ann, frm=frm, spec=spec, start=start, length=length)
    if not (any_ann and any_frm):
        return None
    aq, akv = ann[:, None], ann[None, :]
    fq, fkv = frm[:, None], frm[None, :]
    same_ann = (aq >= 0) & (akv >= 0) & (aq == akv)
    v_reads_own = (fq >= 0) & (akv >= 0) & (fq == akv)
    leak_out = (akv >= 0) & ~same_ann & ~v_reads_own
    leak_in = (aq >= 0) & (fkv >= 0) & (aq != fkv)
    allowed = ~(leak_out | leak_in)
    return allowed.unsqueeze(0).unsqueeze(0)


def _build_segment_masks(
    spec: H3WorldActionMaskSpec,
    *,
    bounds: tuple[int, ...],
    device: torch.device,
    use_flex: bool,
) -> dict[int, Any]:
    masks: dict[int, Any] = {}
    create_block_mask = None
    if use_flex:
        from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask

        create_block_mask = _create_block_mask

    for start, stop in itertools.pairwise(bounds):
        length = stop - start
        if length <= 0:
            continue
        if use_flex and create_block_mask is not None:
            buf = _mask_buffers(device, start, length)
            any_ann, any_frm = _populate_segment_indices(
                ann=buf["ann"], frm=buf["frm"], spec=spec, start=start, length=length
            )
            if not (any_ann and any_frm):
                continue
            masks[start] = create_block_mask(buf["mask_mod"], None, None, length, length, device=device)
        else:
            sdpa_mask = build_h3_world_sdpa_mask(spec, start=start, length=length, device=device)
            if sdpa_mask is not None:
                masks[start] = sdpa_mask
    return masks


class H3WorldDirectedAttention:
    """Instance-local MiniMax-H3 block attention applying H3-World's directed mask."""

    def __init__(self, state: H3WorldAttentionState) -> None:
        self.state = state

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        spec = self.state.spec
        if spec is None:
            return _varlen_attention(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=softmax_scale)

        bounds = tuple(int(value) for value in cu_seqlens.tolist())
        flex_fn = None
        can_flex = (
            self.state.use_flex
            and q.is_cuda
            and not torch.compiler.is_compiling()
            and int(q.shape[-1]) >= 16
            and (int(q.shape[-1]) & (int(q.shape[-1]) - 1)) == 0
        )
        if can_flex:
            flex_fn = _get_compiled_flex_attention()
        use_flex = flex_fn is not None

        cache_key = (str(q.device), bounds, use_flex)
        if self.state._cached_key != cache_key or self.state._cached_masks is None:
            self.state._cached_masks = _build_segment_masks(spec, bounds=bounds, device=q.device, use_flex=use_flex)
            self.state._cached_key = cache_key
        segment_masks = self.state._cached_masks

        out = torch.empty_like(q)
        for start, stop in itertools.pairwise(bounds):
            if stop <= start:
                continue
            q_seg = q[start:stop].transpose(0, 1).unsqueeze(0)
            k_seg = k[start:stop].transpose(0, 1).unsqueeze(0)
            v_seg = v[start:stop].transpose(0, 1).unsqueeze(0)
            mask = segment_masks.get(start)
            if mask is None:
                attended = F.scaled_dot_product_attention(q_seg, k_seg, v_seg, is_causal=False, scale=softmax_scale)
            elif use_flex and flex_fn is not None:
                attended = flex_fn(q_seg, k_seg, v_seg, block_mask=mask, scale=softmax_scale)
            else:
                attended = F.scaled_dot_product_attention(
                    q_seg, k_seg, v_seg, attn_mask=mask, is_causal=False, scale=softmax_scale
                )
            out[start:stop] = attended.squeeze(0).transpose(0, 1)
        return out


def install_h3_world_directed_attention(model: nn.Module, *, use_flex: bool = True) -> H3WorldAttentionState:
    """Attach H3-World directed-mask attention to all DiT blocks of ``model``.

    ``token_refiner.blocks`` intentionally keep standard varlen attention so
    each action sentence is isolated via segmented ``refiner_cu_seqlens``.
    """

    existing = getattr(model, _H3_WORLD_STATE_ATTR, None)
    if isinstance(existing, H3WorldAttentionState):
        existing.use_flex = use_flex
        return existing
    state = H3WorldAttentionState(use_flex=use_flex)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("H3-World directed attention requires a MiniMaxH3DiTModel with .blocks")
    attention = H3WorldDirectedAttention(state)
    for block in blocks:
        block.attn.attention_fn = attention
    setattr(model, _H3_WORLD_STATE_ATTR, state)
    return state


@contextmanager
def h3_world_directed_attention_context(
    model: nn.Module,
    spec: H3WorldActionMaskSpec | None,
) -> Generator[None, None, None]:
    """Bind ``spec`` to ``model`` for the enclosed forward pass(es) and clear on exit."""

    state = getattr(model, _H3_WORLD_STATE_ATTR, None)
    if spec is None:
        yield
        return
    if not isinstance(state, H3WorldAttentionState):
        state = install_h3_world_directed_attention(model)
    previous = state.spec
    state.set_spec(spec)
    try:
        yield
    finally:
        state.set_spec(previous)


def load_h3_world_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    rank: int = 32,
    alpha: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Merge a released H3-World LoRA checkpoint into a canonical MiniMax-H3 DiT.

    H3-World trains rank-32 LoRA (``lora_alpha = rank``, effective scale 1.0)
    on ``attn.qkv_proj`` and ``attn.out_proj`` in DiffSynth's grouped per-head
    QKV layout. ``qkv_proj`` adapter deltas are reordered to WorldFoundry's
    ``[q_all, k_all, v_all]`` layout via :func:`reorder_grouped_qkv_to_qkv`.
    """

    adapter_path = Path(path).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(f"H3-World LoRA checkpoint not found: {adapter_path}")
    rank, alpha = validate_lora_scale(rank=rank, alpha=alpha, label="H3-World")
    scale = float(alpha) / float(rank)

    arch = getattr(model, "arch", None)
    if not isinstance(arch, MiniMaxH3DiTArchConfig):
        arch = MiniMaxH3DiTArchConfig()

    groups = read_lora_groups(adapter_path, label="H3-World")
    if not groups:
        raise ValueError(f"H3-World LoRA checkpoint contains no LoRA tensors: {adapter_path}")

    merged: list[str] = []
    for target, pair in groups.items():
        if not target.endswith(_H3_WORLD_TARGET_SUFFIXES):
            if strict:
                raise ValueError(f"Unsupported H3-World LoRA target: {target}")
            continue
        matrix_a, matrix_b = validate_lora_pair(target, pair, rank=rank, label="H3-World")
        module = model.get_submodule(target)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"H3-World LoRA target is not linear: {target}")

        b_float = matrix_b.float()
        if target.endswith(".attn.qkv_proj"):
            b_float = reorder_grouped_qkv_to_qkv(
                b_float,
                num_query_groups=arch.num_attention_heads,
                heads_per_group=1,
                head_dim=arch.attention_head_dim,
            )
        delta = (b_float @ matrix_a.float()) * scale
        expected = tuple(module.weight.shape)
        if tuple(delta.shape) != expected:
            raise ValueError(f"H3-World LoRA delta shape {tuple(delta.shape)} != {expected} for {target}")
        with torch.no_grad():
            module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        merged.append(target)

    if not merged:
        raise ValueError(f"H3-World LoRA checkpoint contains no supported targets: {adapter_path}")
    return {"path": str(adapter_path), "rank": rank, "alpha": alpha, "merged_targets": tuple(merged)}


__all__ = [
    "H3WorldActionMaskSpec",
    "H3WorldAttentionState",
    "H3WorldDirectedAttention",
    "build_h3_world_sdpa_mask",
    "h3_world_directed_attention_context",
    "install_h3_world_directed_attention",
    "load_h3_world_adapter",
]
