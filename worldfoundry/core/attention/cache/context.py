"""Request-local attention storage for replayed clean temporal context."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from .kvcache import CompactingKVCache


@dataclass(frozen=True)
class ContextAttentionCache:
    """Share committed K/V between calls without storing state on a model.

    Write views append clean blocks; read views expose history plus the current
    noisy block. Rebuilding this container rebuilds position-dependent keys.
    """

    stores: dict[object, CompactingKVCache] = field(default_factory=dict)
    key_masks: dict[object, Tensor] = field(default_factory=dict)
    write: bool = False

    def writing(self, enabled: bool = True) -> ContextAttentionCache:
        return replace(self, write=enabled)

    def combine(
        self, key: object, keys: Tensor, values: Tensor, key_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, int]:
        store = self.stores.get(key)
        history_tokens = 0 if store is None else store.length
        current_keys, current_values, current_mask = keys, values, key_mask
        if history_tokens:
            history_keys, history_values = store.cached()
            keys = torch.cat((history_keys, keys), dim=1)
            values = torch.cat((history_values, values), dim=1)
            if key_mask is not None:
                key_mask = torch.cat((self.key_masks[key], key_mask), dim=1)
        if self.write:
            if store is None:
                store = CompactingKVCache(frame_tokens=1)
                self.stores[key] = store
            store.append(current_keys.detach(), current_values.detach(), frame_count=current_keys.shape[1])
            if current_mask is not None:
                self.key_masks[key] = key_mask.detach()
        return keys, values, key_mask, history_tokens


def prepend_history_mask(mask: Tensor | None, history_tokens: int) -> Tensor | None:
    """Extend an additive attention mask with unrestricted history keys."""
    if mask is None or not history_tokens:
        return mask
    return torch.cat((mask.new_zeros((*mask.shape[:-1], history_tokens)), mask), dim=-1)
