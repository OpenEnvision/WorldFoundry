"""Static cross-attention K/V cache for diffusion denoise loops.

Cross-attention keys/values are a pure function of the conditioning context
(text / image embeddings), which does not change across the denoise steps of a
single request. Recomputing ``k(context)`` / ``v(context)`` every step and every
block is therefore redundant. This module provides a drop-in cross-attention
*processor* for Wan-style ``CrossAttention`` that memoizes the projected K/V per
attention module and reuses them while the conditioning is unchanged.

Correctness (mirrors ``inference_operator_optimization_plan.md`` §7.3):

- The cache key uses tensor identity plus an explicit ``version`` counter, and
  each entry strongly retains its source context. This prevents allocator
  address reuse across classifier-free-guidance branches without introducing
  GPU-to-CPU synchronizations in the denoise hot path.
- Each attention module holds *multiple* cached entries (keyed by content
  fingerprint), so the positive and negative CFG contexts are cached side by
  side rather than evicting each other every step.
- Only the context-derived tensors (K/V and optional image K/V) are cached; the
  query path (which depends on the per-step latent ``x``) always recomputes.
- Bump ``version`` or call :func:`reset_static_cross_kv` between requests, on a
  LoRA/weight change, or whenever the conditioning is replaced.

This is opt-in: install with :func:`install_static_cross_kv_cache`; the model's
default numerical result is unchanged (the same projections, just reused).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _context_signature(context: torch.Tensor, version: int) -> tuple[Any, ...]:
    return (
        version,
        id(context),
        tuple(context.shape),
        str(context.dtype),
        str(context.device),
    )


class StaticCrossKVProcessor:
    """Caching wrapper around a Wan cross-attention processor.

    Delegates to ``inner`` for the query/attention math but memoizes the
    context-derived key/value projections keyed by conditioning identity. A
    shared ``version`` (via the owning cache) invalidates all entries at once.
    """

    def __init__(self, inner: Any, cache: "_StaticCrossKVCache") -> None:
        self._inner = inner
        self._cache = cache

    def __call__(self, attention: nn.Module, x: torch.Tensor, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Unknown processor kwargs may change how context is interpreted.
        if kwargs:
            self._cache.record_bypass(kwargs)
            return self._inner(attention, x, context, **kwargs)

        entry = self._cache.get(attention, context)
        if entry is None:
            from ..models.networks.wan.model import project_wan_cross_kv

            if getattr(attention, "has_image_input", False):
                image_context = context[:, :257]
                text_context = context[:, 257:]
                image_key, image_value = project_wan_cross_kv(attention, image_context, image=True)
            else:
                text_context = context
            key, value = project_wan_cross_kv(attention, text_context)
            tensors = (key, value)
            if getattr(attention, "has_image_input", False):
                tensors += (image_key, image_value)
            self._cache.put(attention, context, *tensors)
        else:
            key, value, *image_entry = entry
        query = attention.norm_q(attention.q(x))
        output = attention.attn(query, key, value)
        if getattr(attention, "has_image_input", False):
            if entry is None:
                cached_image_key, cached_image_value = image_key, image_value
            else:
                cached_image_key, cached_image_value = image_entry
            output = output + attention.attn(query, cached_image_key, cached_image_value)
        return attention.o(output)


class _StaticCrossKVCache:
    """Per-installation store of cached cross-attention K/V, with versioning."""

    # Cap entries per module so an adversarial stream of distinct contexts can't
    # grow the cache without bound. CFG needs 2 (positive + negative); a little
    # headroom covers multi-conditioning variants.
    _MAX_ENTRIES_PER_MODULE = 4

    def __init__(self) -> None:
        self.version = 0
        # id(attention) -> {signature: (key, value)}, so the positive and
        # negative CFG contexts are cached side by side instead of colliding.
        self._entries: dict[
            int,
            dict[
                tuple[Any, ...],
                tuple[torch.Tensor, tuple[torch.Tensor, ...]],
            ],
        ] = {}
        # Projecting raw UMT5/CLIP context to DiT width is also invariant across
        # denoise steps. Keep the source tensors strongly referenced so an
        # allocator cannot recycle their identities between CFG branches.
        self._condition_entries: dict[
            tuple[int, int, int | None],
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
        ] = {}
        self.hits = 0
        self.misses = 0
        self.condition_hits = 0
        self.condition_misses = 0
        self.bypasses = 0
        self.bypass_kwargs: set[str] = set()
        self.lifetime_hits = 0
        self.lifetime_misses = 0
        self.lifetime_condition_hits = 0
        self.lifetime_condition_misses = 0
        self.lifetime_bypasses = 0
        self.wrapped_blocks = 0

    def reset_runtime_window(self) -> None:
        """Start an empty request-scoped telemetry window."""

        self.hits = 0
        self.misses = 0
        self.condition_hits = 0
        self.condition_misses = 0
        self.bypasses = 0
        self.bypass_kwargs.clear()

    def record_bypass(self, kwargs: dict[str, Any]) -> None:
        """Record a call delegated to the original dense processor."""

        self.bypasses += 1
        self.lifetime_bypasses += 1
        self.bypass_kwargs.update(str(name) for name in kwargs)

    def _condition_key(
        self,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
    ) -> tuple[int, int, int | None]:
        return (
            self.version,
            id(context),
            None if clip_feature is None else id(clip_feature),
        )

    def get_condition_context(
        self,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return a projected condition only for the same live source tensors."""

        entry = self._condition_entries.get(self._condition_key(context, clip_feature))
        if entry is not None and entry[0] is context and entry[1] is clip_feature:
            self.condition_hits += 1
            self.lifetime_condition_hits += 1
            return entry[2]
        self.condition_misses += 1
        self.lifetime_condition_misses += 1
        return None

    def put_condition_context(
        self,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        projected: torch.Tensor,
    ) -> None:
        """Cache one raw-to-projected condition mapping for later denoise steps."""

        key = self._condition_key(context, clip_feature)
        if key not in self._condition_entries and len(self._condition_entries) >= self._MAX_ENTRIES_PER_MODULE:
            self._condition_entries.pop(next(iter(self._condition_entries)))
        self._condition_entries[key] = (context, clip_feature, projected)

    def get(self, attention: nn.Module, context: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        signature = _context_signature(context, self.version)
        cached = self._entries.get(id(attention))
        if cached is not None and signature in cached:
            source, projected = cached[signature]
            if source is context:
                self.hits += 1
                self.lifetime_hits += 1
                return projected
        self.misses += 1
        self.lifetime_misses += 1
        return None

    def put(
        self,
        attention: nn.Module,
        context: torch.Tensor,
        *projected: torch.Tensor,
    ) -> None:
        signature = _context_signature(context, self.version)
        entries = self._entries.setdefault(id(attention), {})
        if signature not in entries and len(entries) >= self._MAX_ENTRIES_PER_MODULE:
            entries.pop(next(iter(entries)))
        entries[signature] = (context, projected)

    def invalidate(self) -> None:
        self.version += 1
        self._entries.clear()
        self._condition_entries.clear()
        self.reset_runtime_window()

    def report(self) -> dict[str, object]:
        # entries counts total cached (K,V) pairs across all modules — with the
        # multi-entry cache a module can hold both CFG branches, so this is the
        # sum of per-module entry counts, not the number of cached modules.
        entries = sum(len(per_module) for per_module in self._entries.values())
        if self.hits > 0 and self.bypasses > 0:
            effective = "partial-kv-reuse"
        elif self.hits > 0:
            effective = "kv-reuse"
        elif self.bypasses > 0:
            effective = "dense-bypass"
        elif self.misses > 0:
            effective = "dense-no-hit"
        else:
            effective = "installed (runtime-pending)"
        return {
            "version": self.version,
            "wrapped_blocks": self.wrapped_blocks,
            "effective": effective,
            "hits": self.hits,
            "misses": self.misses,
            "entries": entries,
            "condition_hits": self.condition_hits,
            "condition_misses": self.condition_misses,
            "condition_entries": len(self._condition_entries),
            "bypasses": self.bypasses,
            "bypass_kwargs": sorted(self.bypass_kwargs),
            "lifetime_hits": self.lifetime_hits,
            "lifetime_misses": self.lifetime_misses,
            "lifetime_condition_hits": self.lifetime_condition_hits,
            "lifetime_condition_misses": self.lifetime_condition_misses,
            "lifetime_bypasses": self.lifetime_bypasses,
        }


def install_static_cross_kv_cache(model: nn.Module) -> _StaticCrossKVCache:
    """Wrap every eligible cross-attention processor with a caching processor.

    Returns the shared cache handle; call ``.invalidate()`` between requests or
    when conditioning changes. Idempotent: re-installing reuses the wrapper.
    """

    cache = _StaticCrossKVCache()
    wrapped = 0
    for module in model.modules():
        if (
            type(module).__name__ == "CrossAttention"
            and hasattr(module, "get_processor")
            and hasattr(module, "set_processor")
        ):
            inner = module.get_processor()
            if isinstance(inner, StaticCrossKVProcessor):
                inner = inner._inner
            module.set_processor(StaticCrossKVProcessor(inner, cache))
            wrapped += 1
    cache.wrapped_blocks = wrapped
    model._worldfoundry_static_cross_kv = cache
    return cache


def reset_static_cross_kv(cache: _StaticCrossKVCache) -> None:
    """Invalidate all cached cross-attention K/V (new request / new prompt)."""

    cache.invalidate()


__all__ = [
    "StaticCrossKVProcessor",
    "install_static_cross_kv_cache",
    "reset_static_cross_kv",
]
