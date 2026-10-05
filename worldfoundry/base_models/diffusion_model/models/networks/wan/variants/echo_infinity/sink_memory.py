"""Echo-Infinity sink-token KV memory for causal Wan.

Keeps a fixed sink window of Q/K/V per block.  On update, sink hidden
states are re-encoded against evicted KV so the compact bank stays
fresh.  Causal / memory hook only.
"""

import torch

from .attention import attention
from .causal import causal_rope_apply


class SinkMemory:
    """Fixed-size sink-token KV bank used by Echo-Infinity causal rollouts.

    Each transformer block stores Q/K/V for ``sink_size`` frames.  After a
    chunk evicts older KV, :meth:`update` re-encodes the sink hidden
    states against the evicted context so the compact bank stays current.
    """

    def __init__(self, num_blocks, num_heads, head_dim, tokens_per_frame, sink_size, hidden_dim):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tokens_per_frame = tokens_per_frame
        self.sink_size = sink_size
        self.M = sink_size * tokens_per_frame
        self.hidden_dim = hidden_dim
        self.memory_q = [None] * num_blocks
        self.memory_k = [None] * num_blocks
        self.memory_v = [None] * num_blocks
        self.sink_hidden = None
        self.initialized = False
        self.has_history = False
        self._update_count = 0
        self._pending_sink_captures = {}
        self._pending_sink_hidden = None

    def reset(self):
        """Clear per-block QKV, sink hidden states, and pending captures."""
        for i in range(self.num_blocks):
            self.memory_q[i] = None
            self.memory_k[i] = None
            self.memory_v[i] = None
        self.sink_hidden = None
        self.initialized = False
        self.has_history = False
        self._update_count = 0
        self._pending_sink_captures = {}
        self._pending_sink_hidden = None

    def initialize_block(self, block_idx, q_roped, k_pre_rope, v):
        """Seed one block's sink QKV from the first chunk."""
        self.memory_q[block_idx] = q_roped.clone().detach()
        self.memory_k[block_idx] = k_pre_rope.clone().detach()
        self.memory_v[block_idx] = v.clone().detach()
        if block_idx == self.num_blocks - 1:
            self.initialized = True

    def initialize_hidden(self, hidden):
        """Store the sink-window hidden states used by :meth:`update`."""
        self.sink_hidden = hidden.clone().detach()

    def get_kv(self, block_idx):
        """Return ``(K, V)`` for ``block_idx`` once history exists."""
        if not self.has_history:
            return None
        return (self.memory_k[block_idx], self.memory_v[block_idx])

    @torch.no_grad()
    def update(self, blocks, evicted_kv_all, e0, context, context_lens, freqs, grid_sizes, crossattn_cache_template=None, evicted_k_is_pre_rope=False):
        """Re-run the sink tokens through each block against evicted KV."""
        if not self.initialized or self.sink_hidden is None:
            return
        x_mem = self.sink_hidden
        B = x_mem.shape[0]
        Q_frames = self.sink_size
        tpf = self.tokens_per_frame
        n, d = (self.num_heads, self.head_dim)
        mem_grid = grid_sizes.clone()
        mem_grid[:, 0] = Q_frames
        for i in range(self.num_blocks):
            block = blocks[i]
            evicted_k_i, evicted_v_i = evicted_kv_all[i]
            e_block = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
            h = block.norm1(x_mem)
            h = (h.unflatten(1, (Q_frames, tpf)) * (1 + e_block[1]) + e_block[0]).flatten(1, 2)
            mem_k_roped = causal_rope_apply(self.memory_k[i], mem_grid, freqs, start_frame=0)
            if evicted_k_is_pre_rope and evicted_k_i is not None and (evicted_k_i.shape[1] > 0):
                ev_frames = evicted_k_i.shape[1] // self.tokens_per_frame
                if ev_frames > 0:
                    ev_grid = grid_sizes.clone()
                    ev_grid[:, 0] = ev_frames
                    evicted_k_i = causal_rope_apply(evicted_k_i, ev_grid, freqs, start_frame=self.sink_size)
            ctx_k = torch.cat([mem_k_roped, evicted_k_i], dim=1)
            ctx_v = torch.cat([self.memory_v[i], evicted_v_i], dim=1)
            attn_out = attention(self.memory_q[i], ctx_k, ctx_v)
            y = block.self_attn.o(attn_out.flatten(2))
            x_mem = x_mem + (y.unflatten(1, (Q_frames, tpf)) * e_block[2]).flatten(1, 2)
            x_mem = x_mem + block.cross_attn(block.norm3(x_mem), context, context_lens)
            y_ffn = block.ffn((block.norm2(x_mem).unflatten(1, (Q_frames, tpf)) * (1 + e_block[4]) + e_block[3]).flatten(1, 2))
            x_mem = x_mem + (y_ffn.unflatten(1, (Q_frames, tpf)) * e_block[5]).flatten(1, 2)
            if i < self.num_blocks - 1:
                next_block = blocks[i + 1]
                next_e = (next_block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
                h_next = next_block.norm1(x_mem)
                h_next_mod = (h_next.unflatten(1, (Q_frames, tpf)) * (1 + next_e[1]) + next_e[0]).flatten(1, 2)
                self.memory_k[i + 1] = next_block.self_attn.norm_k(next_block.self_attn.k(h_next_mod)).view(B, self.M, n, d)
                self.memory_v[i + 1] = next_block.self_attn.v(h_next_mod).view(B, self.M, n, d)
        self.has_history = True
        self._update_count += 1
        if self._update_count <= 100 or self._update_count % 10 == 0:
            sample_layers = [0, self.num_blocks // 2, self.num_blocks - 1]
            norms_k = [f'L{i}={self.memory_k[i].norm().item():.1f}' for i in sample_layers]
            norms_v = [f'L{i}={self.memory_v[i].norm().item():.1f}' for i in sample_layers]
