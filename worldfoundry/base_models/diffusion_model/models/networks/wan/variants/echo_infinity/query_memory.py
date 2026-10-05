"""Echo-Infinity learned query-memory encoder.

A small stack of cross-attention layers compresses cached K/V into
learned query tokens (optionally multi-group) that later chunks attend
to.  Memory adapter only.
"""

import torch
import torch.nn as nn

from .attention import attention
from .model import WanRMSNorm


class MemoryCrossAttentionLayer(nn.Module):
    """One query-memory encoder layer: attend queries to cached K/V, then FFN."""

    def __init__(self, dim, num_heads, ffn_dim, qk_norm=True, eps=1e-06):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm1 = WanRMSNorm(dim, eps=eps)
        self.norm2 = WanRMSNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))

    def forward(self, query_state, cached_k, cached_v):
        """Update learned queries by cross-attending to a cached memory bank."""
        b = query_state.shape[0]
        m = query_state.shape[1]
        n, d = (self.num_heads, self.head_dim)
        residual = query_state
        h = self.norm1(query_state)
        q = self.norm_q(self.q(h)).view(b, m, n, d)
        out = attention(q, cached_k, cached_v)
        out = out.flatten(2).type_as(h)
        out = self.o(out)
        query_state = residual + out
        residual = query_state
        query_state = residual + self.ffn(self.norm2(query_state))
        return query_state

class QueryMemoryEncoder(nn.Module):
    """Compress cached K/V into learned query tokens for later Infinity chunks."""

    def __init__(self, config):
        super().__init__()
        Q_frames = getattr(config, 'Q_frames', 3)
        tokens_per_frame = getattr(config, 'tokens_per_frame', 1560)
        n_encoder_layers = getattr(config, 'n_encoder_layers', 2)
        hidden_dim = getattr(config, 'hidden_dim', 1536)
        num_heads = getattr(config, 'num_heads', 12)
        head_dim = getattr(config, 'head_dim', 128)
        ffn_dim = hidden_dim * 4
        gate_init_bias = getattr(config, 'gate_init_bias', 2.0)
        qk_norm = getattr(config, 'qk_norm', True)
        eps = 1e-06
        self.M = Q_frames * tokens_per_frame
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.Q_frames = Q_frames
        self.tokens_per_frame = tokens_per_frame
        self.use_batch_update = getattr(config, 'use_batch_update', False)
        self.batch_update_interval = getattr(config, 'batch_update_interval', 1)
        self.use_sink_anchor = getattr(config, 'use_sink_anchor', False)
        self.use_vib = getattr(config, 'use_vib', False)
        self.bptt_clips = getattr(config, 'bptt_clips', 1)
        self.encoder_lr_multiplier = getattr(config, 'encoder_lr_multiplier', 5.0)
        self.normalize_memory_k = getattr(config, 'normalize_memory_k', False)
        self.use_residual_update = getattr(config, 'use_residual_update', False)
        self.use_post_norm = getattr(config, 'use_post_norm', False)
        self.memory_recache = getattr(config, 'memory_recache', False)
        self.num_query_groups = getattr(config, 'num_query_groups', 1)
        initializer_range = getattr(config, 'initializer_range', 0.014)
        self.layers = nn.ModuleList([MemoryCrossAttentionLayer(hidden_dim, num_heads, ffn_dim, qk_norm, eps) for _ in range(n_encoder_layers)])
        if self.use_post_norm:
            self.post_norm = WanRMSNorm(hidden_dim, eps=eps)
        if self.num_query_groups > 1:
            self.query_inits = nn.ParameterList([nn.Parameter(torch.randn(1, self.M, hidden_dim) * initializer_range) for _ in range(self.num_query_groups)])
            self.to_k_groups = nn.ModuleList([nn.Linear(hidden_dim, num_heads * head_dim) for _ in range(self.num_query_groups)])
            self.to_v_groups = nn.ModuleList([nn.Linear(hidden_dim, num_heads * head_dim) for _ in range(self.num_query_groups)])
            if self.normalize_memory_k:
                self.norm_k_out_groups = nn.ModuleList([WanRMSNorm(num_heads * head_dim, eps=eps) for _ in range(self.num_query_groups)])
            self.connector_projs = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(approximate='tanh'), nn.Linear(hidden_dim, hidden_dim), WanRMSNorm(hidden_dim, eps=eps)) for _ in range(self.num_query_groups)])
            self.gate_linears = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(self.num_query_groups)])
            for gl in self.gate_linears:
                with torch.no_grad():
                    gl.bias.fill_(gate_init_bias)
        else:
            self.query_init = nn.Parameter(torch.randn(1, self.M, hidden_dim) * initializer_range)
            self.connector_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(approximate='tanh'), nn.Linear(hidden_dim, hidden_dim), WanRMSNorm(hidden_dim, eps=eps))
            self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            with torch.no_grad():
                self.gate_linear.bias.fill_(gate_init_bias)
            self.to_k = nn.Linear(hidden_dim, num_heads * head_dim)
            self.to_v = nn.Linear(hidden_dim, num_heads * head_dim)
            if self.normalize_memory_k:
                self.norm_k_out = WanRMSNorm(num_heads * head_dim, eps=eps)
        if self.use_vib:
            self.mu_proj = nn.Linear(hidden_dim, hidden_dim)
            self.logvar_proj = nn.Linear(hidden_dim, hidden_dim)
        self._evicted_k_buffer = []
        self._evicted_v_buffer = []
        self.query_state = None
        self.has_history = False
        self._update_count = 0
        total_params = sum((p.numel() for p in self.parameters()))

    def reset(self, batch_size=1, device=None, dtype=None):

        def _expand_init(init):
            if init.dim() != 3 or init.shape[1] != self.M:
                raise RuntimeError(f'query_init has unexpected shape {list(init.shape)} (expected [1, {self.M}, {self.hidden_dim}]). This usually means the encoder was placed inside FSDP, which flattens parameters. The encoder must remain outside FSDP (attached via object.__setattr__).')
            state = init.expand(batch_size, -1, -1).clone()
            if device is not None:
                state = state.to(device=device)
            if dtype is not None:
                state = state.to(dtype=dtype)
            return state
        if self.num_query_groups > 1:
            self.query_states = [_expand_init(init) for init in self.query_inits]
            self.query_state = self.query_states[0]
        else:
            self.query_state = _expand_init(self.query_init)
        self.has_history = False
        self._update_count = 0
        self._evicted_k_buffer = []
        self._evicted_v_buffer = []
        shape = list(self.query_states[0].shape) if self.num_query_groups > 1 else list(self.query_state.shape)

    def update(self, evicted_k, evicted_v, sink_k=None, sink_v=None):
        if self.query_state is None:
            self.reset(batch_size=evicted_k.shape[0], device=evicted_k.device, dtype=evicted_k.dtype if evicted_k.dtype.is_floating_point else torch.bfloat16)
        query_state = self.query_state
        if self.use_batch_update:
            self._evicted_k_buffer.append(evicted_k)
            self._evicted_v_buffer.append(evicted_v)
            if len(self._evicted_k_buffer) < self.batch_update_interval:
                return 0.0
            evicted_k = torch.cat(self._evicted_k_buffer, dim=1)
            evicted_v = torch.cat(self._evicted_v_buffer, dim=1)
            self._evicted_k_buffer = []
            self._evicted_v_buffer = []
        if self.use_sink_anchor and sink_k is not None:
            ctx_k = torch.cat([sink_k, evicted_k], dim=1)
            ctx_v = torch.cat([sink_v, evicted_v], dim=1)
        else:
            ctx_k = evicted_k
            ctx_v = evicted_v

        def _update_single_group(qs, connector_proj, gate_linear):
            state = qs
            for layer in self.layers:
                state = layer(state, ctx_k, ctx_v)
            if self.use_residual_update:
                new_state = state
                gate_mean = None
            else:
                projected = connector_proj(state)
                gate = torch.sigmoid(gate_linear(torch.cat([qs, projected], dim=-1)))
                new_state = gate * qs + (1 - gate) * projected
                gate_mean = gate.mean().item()
            if self.use_post_norm:
                new_state = self.post_norm(new_state)
            return (new_state, gate_mean)
        kl_loss = 0.0
        if self.num_query_groups > 1:
            gate_means = []
            for g in range(self.num_query_groups):
                new_state, gm = _update_single_group(self.query_states[g], self.connector_projs[g], self.gate_linears[g])
                if self.use_vib:
                    new_state = self.mu_proj(new_state)
                self.query_states[g] = new_state
                gate_means.append(gm)
            self.query_state = self.query_states[0]
            gate_str = ', '.join((f"g{i}={('res' if gm is None else f'{gm:.3f}')}" for i, gm in enumerate(gate_means)))
            norm_str = ', '.join((f'g{i}={self.query_states[i].norm().item():.1f}' for i in range(self.num_query_groups)))
        else:
            new_state, gate_mean = _update_single_group(query_state, self.connector_proj, self.gate_linear)
            if self.use_vib:
                new_state = self.mu_proj(new_state)
            self.query_state = new_state
            gate_str = f"{('residual_update' if gate_mean is None else f'gate_mean={gate_mean:.3f}')}"
            norm_str = f'{new_state.norm().item():.1f}'
        self.has_history = True
        self._update_count += 1
        return kl_loss

    def get_kv(self, group_index=None):
        if not self.has_history:
            return None
        if self.num_query_groups > 1 and group_index is not None:
            state = self.query_states[group_index]
            k = self.to_k_groups[group_index](state)
            if self.normalize_memory_k:
                k = self.norm_k_out_groups[group_index](k)
            k = k.view(state.shape[0], self.M, self.num_heads, self.head_dim)
            v = self.to_v_groups[group_index](state).view(state.shape[0], self.M, self.num_heads, self.head_dim)
            return (k, v)
        else:
            state = self.query_state
            k = self.to_k(state)
            if self.normalize_memory_k:
                k = self.norm_k_out(k)
            k = k.view(state.shape[0], self.M, self.num_heads, self.head_dim)
            v = self.to_v(state).view(state.shape[0], self.M, self.num_heads, self.head_dim)
            return (k, v)

    def detach_state(self):
        if self.num_query_groups > 1:
            self.query_states = [s.detach() for s in self.query_states]
            self.query_state = self.query_states[0]
        elif self.query_state is not None:
            self.query_state = self.query_state.detach()
