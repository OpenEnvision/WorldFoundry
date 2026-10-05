from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.blockmask import BlockPattern
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.rope import RopeCache


class KVCacheMode(str, Enum):
    DISABLED = "disabled"  # normal forward
    APPEND = "append"  # append current K/V into cache
    READONLY = "readonly"  # do NOT modify cache; attend to (cache + current K/V)


class KVCache:
    """
    Stores K/V in the SAME layout as seen by local_attn.
    - CP enabled: post-A2A layout: [B, S_global, H_local, D]
    - CP disabled:            layout: [B, S,       H_total, D]
    Stored tensors are detached from the computation graph.
    Keys may be pre-RoPE or post-RoPE depending on ``RopeCache.cached_k_rotated``.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.k = None
        self.v = None
        self.cur = 0
        self._chunk_lens: List[int] = []
        self._cum_ends: List[int] = []

    def reset(self, free_buffers: bool = True):
        if free_buffers:
            self.k = self.v = None
        self.cur = 0
        self._chunk_lens = []
        self._cum_ends = []

    def ensure(self, B: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
        if self.k is None:
            self.k = torch.empty((B, self.max_len, H, D), device=device, dtype=dtype)
            self.v = torch.empty((B, self.max_len, H, D), device=device, dtype=dtype)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[int, int]:
        B, s, H, D = k.shape
        if self.cur + s > self.max_len:
            raise RuntimeError(f"KV cache overflow: cur={self.cur}, append={s}, max_len={self.max_len}")
        start = self.cur
        end = start + s
        self.ensure(B, H, D, k.dtype, k.device)
        self.k[:, start:end].copy_(k.detach())
        self.v[:, start:end].copy_(v.detach())
        self.cur = end
        self._chunk_lens.append(s)
        self._cum_ends.append(self.cur)
        return start, end

    def get(self, block_range: Optional[int] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns a prefix of cached K/V.

        block_range semantics:
          - None: return the full cached prefix [0, cur)
          - 0:    return no cached prefix (None, None)
          - n>0:  return cached prefix up to the end of chunk n (exclusive by chunk count)
        """
        if self.cur == 0:
            return None, None
        if self.k is None:
            return None, None
        if block_range is None:
            end = self.cur
        else:
            if block_range < 0:
                raise ValueError(f"block_range must be >= 0, got {block_range}")
            if block_range == 0:
                return None, None
            if block_range > len(self._cum_ends):
                raise IndexError(f"block_range={block_range} exceeds cached chunks={len(self._cum_ends)}")
            end = self._cum_ends[block_range - 1]
        return self.k[:, :end], self.v[:, :end]

    def compact_(self, keep_indices: Optional[torch.Tensor]):
        """Keep only the tokens at ``keep_indices``.

        Args:
            keep_indices: 1D ``[budget]`` for shared selection across all batch
                elements, or 2D ``[B, budget]`` for per-sample selection.
        """
        if self.k is None or self.cur == 0 or keep_indices is None:
            return
        keep_indices = keep_indices.to(device=self.k.device, dtype=torch.long)

        if keep_indices.ndim == 1:
            if keep_indices.numel() == 0:
                self.cur = 0
                self._chunk_lens = []
                self._cum_ends = []
                return
            if torch.any(keep_indices < 0) or torch.any(keep_indices >= self.cur):
                raise IndexError(f"KV compact indices out of bounds for cur={self.cur}")
            k_new = self.k.index_select(dim=1, index=keep_indices)
            v_new = self.v.index_select(dim=1, index=keep_indices)
            new_len = keep_indices.numel()
        elif keep_indices.ndim == 2:
            B, budget = keep_indices.shape
            if budget == 0:
                self.cur = 0
                self._chunk_lens = []
                self._cum_ends = []
                return
            if torch.any(keep_indices < 0) or torch.any(keep_indices >= self.cur):
                raise IndexError(f"KV compact indices out of bounds for cur={self.cur}")
            H, D = self.k.shape[2], self.k.shape[3]
            idx = keep_indices.unsqueeze(-1).unsqueeze(-1).expand(B, budget, H, D)
            k_new = torch.gather(self.k[:, : self.cur], dim=1, index=idx)
            v_new = torch.gather(self.v[:, : self.cur], dim=1, index=idx)
            new_len = budget
        else:
            raise ValueError(f"keep_indices must be 1D or 2D, got {keep_indices.ndim}D")

        self.k[:, :new_len].copy_(k_new)
        self.v[:, :new_len].copy_(v_new)
        self.cur = new_len
        self._chunk_lens = [new_len]
        self._cum_ends = [new_len]

    def get_prefix_end(self, block_range: Optional[int] = None) -> int:
        """End position of the committed prefix up to ``block_range`` chunks."""
        if block_range is None:
            return self.cur
        if block_range <= 0:
            return 0
        if block_range > len(self._cum_ends):
            raise IndexError(f"block_range={block_range} exceeds cached chunks={len(self._cum_ends)}")
        return self._cum_ends[block_range - 1]

    def write_transient(self, k: torch.Tensor, v: torch.Tensor, at: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write K/V at position ``at`` WITHOUT updating metadata.

        Returns slices ``[0, at + s)`` covering the committed prefix plus the
        just-written current block.  Safe to call repeatedly at the same ``at``
        across denoising steps (overwrites previous transient data).
        """
        B, s, H, D = k.shape
        self.ensure(B, H, D, k.dtype, k.device)
        end = at + s
        self.k[:, at:end].copy_(k.detach())
        self.v[:, at:end].copy_(v.detach())
        return self.k[:, :end], self.v[:, :end]

    @property
    def current_len(self) -> int:
        return self.cur

    @property
    def committed_blocks(self) -> int:
        """Number of blocks the caller has appended.

        Identical to the retained chunk count here. A cache that evicts must
        override this so the caller's global block cursor stays comparable even
        after chunks are dropped.
        """
        return len(self._cum_ends)


@dataclass
class AttnContext:
    """Per-layer attention context carrying KV cache, RoPE, and observer."""

    mode: KVCacheMode = KVCacheMode.DISABLED
    kv_cache: Optional[KVCache] = None
    block_range: Optional[int] = 0
    layer_idx: int = 0
    q_block_idx: int = 0
    attn_observer: Optional[object] = None
    rope: Optional[RopeCache] = None
    fast_infer: bool = False


@dataclass
class CausalInferenceState:
    """Per-forward-pass inference state for causal video generation.

    **Caller-set fields**:

    - ``block_cursor``: index of the block being generated.
    - ``block_range``: how much of the KV cache to read.
      ``None`` = full prefix, int ``n`` = first *n* chunks.
      Defaults to ``block_cursor`` when omitted.
    - ``cached_t_indices``, ``current_t_indices``: explicit per-token temporal
      indices for RoPE.  When set, ``_compute_rope`` uses pre-RoPE caching
      (``cached_k_rotated=False``).  When ``None``, sequential positions are
      used with post-RoPE caching.

    **Model-populated fields** (set by ``_compute_rope`` before the block loop):

    - ``rope``: shared RoPE state for all layers.
    """

    mode: KVCacheMode = KVCacheMode.DISABLED
    kv_caches: Optional[List[KVCache]] = None
    pattern: Optional[BlockPattern] = None
    block_cursor: int = 0
    block_range: Optional[int] = -1
    cached_t_indices: Optional[torch.Tensor] = None
    current_t_indices: Optional[torch.Tensor] = None
    attn_observer: Optional[object] = None
    rope: Optional[RopeCache] = None
    fast_infer: bool = False

    def __post_init__(self):
        if self.block_range == -1:
            self.block_range = self.block_cursor

    @property
    def t_offset(self) -> int:
        if self.pattern is None:
            return 0
        return self.pattern.blocks_to_frames(self.block_cursor)

    def attn_ctx(self, layer_idx: int) -> Optional[AttnContext]:
        if self.mode == KVCacheMode.DISABLED or self.kv_caches is None:
            return None
        return AttnContext(
            mode=self.mode,
            kv_cache=self.kv_caches[layer_idx],
            block_range=self.block_range,
            layer_idx=layer_idx,
            q_block_idx=self.block_cursor,
            attn_observer=self.attn_observer,
            rope=self.rope,
            fast_infer=self.fast_infer,
        )
