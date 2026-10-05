"""Echo-Infinity causal Wan 2.x with chunk KV cache.

Causal self-attention plus ``causal_rope_apply`` for streaming chunks.
Sink / query memory banks are in sibling modules.

Adds causal attention.  No VACE or linear attn.
"""
from .attention import attention
from .model import WanRMSNorm, rope_apply, WanLayerNorm, WAN_CROSSATTENTION_CLASSES, rope_params, MLPProj, sinusoidal_embedding_1d
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
import os
import torch.nn as nn
import torch
import math
import torch.distributed as dist
flex_attention = torch.compile(flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs')

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Causal rope apply.

    Args:
        x: The x.
        grid_sizes: The grid sizes.
        freqs: The freqs.
        start_frame: The start frame.
    """
    n, c = (x.size(2), x.size(3) // 2)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        if f == 0:
            output.append(x[i])
            continue
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1), freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)

class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with optional KV cache and frame-offset RoPE."""

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-06):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            local_attn_size: The local attn size.
            sink_size: The sink size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, '__iter__'):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.dr_rope = False
        self.tri_rope_cont = False
        self.tri_rope_pmax = 21
        self.relative_rope = False
        self.relative_rope_pmax = 21
        self.num_frame_per_block_attr = 3
        self._layer_id = -1

    @staticmethod
    def _compute_relative_positions(current_start_frame, B, R, N_Q, N_S, pmax, num_frame_per_block):
        """Helper function to compute relative positions.

        Args:
            current_start_frame: The current start frame.
            B: The b.
            R: The r.
            N_Q: The n q.
            N_S: The n s.
            pmax: The pmax.
            num_frame_per_block: The num frame per block.
        """
        is_bulk_forward = B > num_frame_per_block
        q_last_pos = min(current_start_frame + B - 1, pmax - 1)
        q_start_pos = q_last_pos - B + 1
        local_end_pos = q_last_pos
        local_start_pos = local_end_pos - R + 1 if R > 0 else q_last_pos
        if is_bulk_forward or N_Q == 0:
            use_memory = False
            mem_start_pos = -1
            mem_end_pos = -1
        else:
            use_memory = True
            mem_end_pos = local_start_pos - 1
            mem_start_pos = mem_end_pos - N_Q + 1
        return dict(is_bulk_forward=is_bulk_forward, use_memory=use_memory, q_start=q_start_pos, q_last=q_last_pos, local_start=local_start_pos, local_end=local_end_pos, mem_start=mem_start_pos, mem_end=mem_end_pos, sink_start=0)

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, cache_start=None, sink_recache_after_switch=False, memory_kv=None, capture_sink_qkv=False):
        """Forward.

        Args:
            x: The x.
            seq_lens: The seq lens.
            grid_sizes: The grid sizes.
            freqs: The freqs.
            block_mask: The block mask.
            kv_cache: The kv cache.
            current_start: The current start.
            cache_start: The cache start.
            sink_recache_after_switch: The sink recache after switch.
            memory_kv: The memory kv.
            capture_sink_qkv: The capture sink qkv.
        """
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return (q, k, v)
        q, k, v = qkv_fn(x)
        _sink_qkv_captured = None
        if capture_sink_qkv and kv_cache is not None:
            sink_tokens = self.sink_size * math.prod(grid_sizes[0][1:]).item()
            if sink_tokens > 0 and s >= sink_tokens:
                _sink_qkv_captured = (q[:, :sink_tokens].clone(), k[:, :sink_tokens].clone(), v[:, :sink_tokens].clone())
        if kv_cache is None:
            is_tf = s == seq_lens[0].item() * 2
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat([roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)], dim=1)
                padded_roped_key = torch.cat([roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)], dim=1)
                padded_v = torch.cat([v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)
                x = flex_attention(query=padded_roped_query.transpose(2, 1), key=padded_roped_key.transpose(2, 1), value=padded_v.transpose(2, 1), block_mask=block_mask)[:, :, :-padded_length].transpose(2, 1)
            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat([roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)], dim=1)
                padded_roped_key = torch.cat([roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)], dim=1)
                padded_v = torch.cat([v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)
                x = flex_attention(query=padded_roped_query.transpose(2, 1), key=padded_roped_key.transpose(2, 1), value=padded_v.transpose(2, 1), block_mask=block_mask)[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            _pre_rope_cache = self.dr_rope or self.tri_rope_cont or self.relative_rope
            if not _pre_rope_cache:
                roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                current_end = current_start + roped_query.shape[1]
            else:
                current_end = current_start + q.shape[1]
            k_for_cache = k if _pre_rope_cache else roped_key
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache['k'].shape[1]
            num_new_tokens = q.shape[1]
            cache_update_info = None
            is_recompute = current_end <= kv_cache['global_end_index'].item() and current_start > 0
            if self.local_attn_size != -1 and current_end > kv_cache['global_end_index'].item() and (num_new_tokens + kv_cache['local_end_index'].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache['local_end_index'].item() - kv_cache_size
                num_rolled_tokens = kv_cache['local_end_index'].item() - num_evicted_tokens - sink_tokens
                local_end_index = kv_cache['local_end_index'].item() + current_end - kv_cache['global_end_index'].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                temp_k = kv_cache['k'].clone()
                temp_v = kv_cache['v'].clone()
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = k_for_cache[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
                cache_update_info = {'action': 'roll_and_insert', 'sink_tokens': sink_tokens, 'num_rolled_tokens': num_rolled_tokens, 'num_evicted_tokens': num_evicted_tokens, 'local_start_index': local_start_index, 'local_end_index': local_end_index, 'write_start_index': write_start_index, 'write_end_index': local_end_index, 'new_k': k_for_cache[:, roped_offset:roped_offset + write_len], 'new_v': v[:, roped_offset:roped_offset + write_len], 'current_end': current_end, 'is_recompute': is_recompute}
            else:
                local_end_index = kv_cache['local_end_index'].item() + current_end - kv_cache['global_end_index'].item()
                local_start_index = local_end_index - num_new_tokens
                temp_k = kv_cache['k'].clone()
                temp_v = kv_cache['v'].clone()
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = k_for_cache[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
                cache_update_info = {'action': 'direct_insert', 'local_start_index': local_start_index, 'local_end_index': local_end_index, 'write_start_index': write_start_index, 'write_end_index': local_end_index, 'new_k': k_for_cache[:, roped_offset:roped_offset + write_len], 'new_v': v[:, roped_offset:roped_offset + write_len], 'current_end': current_end, 'is_recompute': is_recompute}
            if sink_tokens > 0:
                local_budget = self.max_attention_size - sink_tokens
                k_sink_stored = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
                num_new_frames_tri = num_new_tokens // frame_seqlen if frame_seqlen > 0 else 0
                current_end_frame_tri = current_start_frame + num_new_frames_tri
                if self.tri_rope_cont:
                    N_S = self.sink_size
                    N_L_max = (self.max_attention_size - N_S * frame_seqlen) // frame_seqlen
                    N_Q_active = memory_kv[0].shape[1] // frame_seqlen if memory_kv is not None else 0
                    delta_cum = max(0, current_start_frame - (N_S + N_L_max))
                    delta_target = self.tri_rope_pmax - (N_S + N_Q_active + N_L_max)
                    delta_eff = max(0, min(delta_cum, delta_target))
                    N_S_cache_tri = min(current_end_frame_tri, N_S)
                    tri_sink_start = delta_eff
                    tri_mem_start = delta_eff + N_S_cache_tri
                    tri_local_start = delta_eff + N_S_cache_tri + N_Q_active
                    CausalWanSelfAttention._delta_sum = getattr(CausalWanSelfAttention, '_delta_sum', 0) + int(delta_eff)
                    CausalWanSelfAttention._delta_count = getattr(CausalWanSelfAttention, '_delta_count', 0) + 1
                    if delta_target > 0:
                        CausalWanSelfAttention._delta_at_cap = getattr(CausalWanSelfAttention, '_delta_at_cap', 0) + int(delta_eff == delta_target)
                    else:
                        CausalWanSelfAttention._delta_at_cap = getattr(CausalWanSelfAttention, '_delta_at_cap', 0)
                    if delta_eff > 0 and (not getattr(CausalWanSelfAttention, '_tri_delta_logged', False)):
                        CausalWanSelfAttention._tri_delta_logged = True
                    if delta_target > 0 and delta_eff == delta_target and (not getattr(CausalWanSelfAttention, '_tri_cap_logged', False)):
                        CausalWanSelfAttention._tri_cap_logged = True
                else:
                    delta_eff = 0
                    N_S_cache_tri = self.sink_size
                    N_Q_active = 0
                    tri_sink_start = 0
                    tri_mem_start = 0
                    tri_local_start = 0
                if self.relative_rope:
                    N_Q_rr = memory_kv[0].shape[1] // frame_seqlen if memory_kv is not None else 0
                    N_S_cache_rr = min(current_end_frame_tri, self.sink_size)
                if self.tri_rope_cont:
                    sink_grid = grid_sizes.clone()
                    sink_grid[:, 0] = N_S_cache_tri
                    k_sink = causal_rope_apply(k_sink_stored, sink_grid, freqs, start_frame=tri_sink_start).type_as(v)
                elif self.dr_rope:
                    sink_grid = grid_sizes.clone()
                    sink_grid[:, 0] = self.sink_size
                    k_sink = causal_rope_apply(k_sink_stored, sink_grid, freqs, start_frame=0).type_as(v)
                elif self.relative_rope:
                    sink_grid = grid_sizes.clone()
                    sink_grid[:, 0] = N_S_cache_rr
                    k_sink = causal_rope_apply(k_sink_stored, sink_grid, freqs, start_frame=0).type_as(v)
                else:
                    k_sink = k_sink_stored
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local_stored = temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]
                    R_active = max(0, k_local_stored.shape[1] // frame_seqlen) if frame_seqlen > 0 else 0
                    num_new_frames = num_new_tokens // frame_seqlen if frame_seqlen > 0 else 0
                    if self.tri_rope_cont:
                        if R_active > 0:
                            local_grid = grid_sizes.clone()
                            local_grid[:, 0] = R_active
                            k_local = causal_rope_apply(k_local_stored, local_grid, freqs, start_frame=tri_local_start).type_as(v)
                        else:
                            k_local = k_local_stored
                        tri_q_start = tri_local_start + R_active - num_new_frames
                        q_slot_start = local_start_index // frame_seqlen
                        q_grid = grid_sizes.clone()
                        q_grid[:, 0] = num_new_frames
                        roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=tri_q_start).type_as(v)
                    elif self.dr_rope:
                        local_start_rope = local_start_for_window // frame_seqlen
                        q_start_rope = local_start_index // frame_seqlen
                        if R_active > 0:
                            local_grid = grid_sizes.clone()
                            local_grid[:, 0] = R_active
                            k_local = causal_rope_apply(k_local_stored, local_grid, freqs, start_frame=local_start_rope).type_as(v)
                        else:
                            k_local = k_local_stored
                        q_grid = grid_sizes.clone()
                        q_grid[:, 0] = num_new_frames
                        roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=q_start_rope).type_as(v)
                    elif self.relative_rope:
                        rr_pos = CausalWanSelfAttention._compute_relative_positions(current_start_frame=current_start_frame, B=num_new_frames, R=R_active, N_Q=N_Q_rr, N_S=self.sink_size, pmax=self.relative_rope_pmax, num_frame_per_block=self.num_frame_per_block_attr)
                        _diag_rr = f"[diag-rr] layer={self._layer_id} cur_start_f={current_start_frame} B={num_new_frames} R={R_active} N_Q={N_Q_rr} N_S={self.sink_size} pmax={self.relative_rope_pmax} B_f={self.num_frame_per_block_attr} is_bulk={rr_pos['is_bulk_forward']} use_mem={rr_pos['use_memory']} sink_s={rr_pos['sink_start']} mem_s={rr_pos['mem_start']} mem_e={rr_pos['mem_end']} local_s={rr_pos['local_start']} local_e={rr_pos['local_end']} q_s={rr_pos['q_start']} q_l={rr_pos['q_last']}"
                        assert rr_pos['q_last'] <= self.relative_rope_pmax - 1, f'RelRope overflow: q_last > pmax-1 | {_diag_rr}'
                        assert rr_pos['q_start'] >= 0, f'RelRope underflow: q_start < 0 | {_diag_rr}'
                        if rr_pos['use_memory']:
                            assert rr_pos['mem_start'] >= self.sink_size, f'RelRope memory overlaps sink: mem_start < sink_size | {_diag_rr}'
                            assert rr_pos['mem_end'] + 1 == rr_pos['local_start'], f'RelRope memory-local not contiguous | {_diag_rr}'
                        if R_active > 0 and (not rr_pos['is_bulk_forward']):
                            assert rr_pos['q_last'] == rr_pos['local_end'], f'RelRope Q-local tail mismatch: q_last != local_end | {_diag_rr}'
                        if R_active > 0:
                            local_grid = grid_sizes.clone()
                            local_grid[:, 0] = R_active
                            k_local = causal_rope_apply(k_local_stored, local_grid, freqs, start_frame=rr_pos['local_start']).type_as(v)
                        else:
                            k_local = k_local_stored
                        q_grid = grid_sizes.clone()
                        q_grid[:, 0] = num_new_frames
                        roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=rr_pos['q_start']).type_as(v)
                        CausalWanSelfAttention._rr_q_last_sum = getattr(CausalWanSelfAttention, '_rr_q_last_sum', 0) + int(rr_pos['q_last'])
                        CausalWanSelfAttention._rr_total_count = getattr(CausalWanSelfAttention, '_rr_total_count', 0) + 1
                        if rr_pos['is_bulk_forward']:
                            CausalWanSelfAttention._rr_bulk_count = getattr(CausalWanSelfAttention, '_rr_bulk_count', 0) + 1
                        if rr_pos['q_last'] == self.relative_rope_pmax - 1 and (not rr_pos['is_bulk_forward']):
                            CausalWanSelfAttention._rr_long_count = getattr(CausalWanSelfAttention, '_rr_long_count', 0) + 1
                        if not rr_pos['is_bulk_forward'] and rr_pos['q_last'] == self.relative_rope_pmax - 1 and (not getattr(CausalWanSelfAttention, '_rr_long_logged', False)):
                            mem_str = f"mem[{rr_pos['mem_start']}..{rr_pos['mem_end']}]" if rr_pos['use_memory'] else 'mem=(inactive)'
                            CausalWanSelfAttention._rr_long_logged = True
                        if rr_pos['is_bulk_forward'] and (not getattr(CausalWanSelfAttention, '_rr_bulk_logged', False)):
                            CausalWanSelfAttention._rr_bulk_logged = True
                    else:
                        k_local = k_local_stored
                    _rr_skip_memory = self.relative_rope and rr_pos['is_bulk_forward']
                    if memory_kv is not None and (not _rr_skip_memory):
                        mem_k, mem_v = memory_kv
                        Q_frames = mem_k.shape[1] // frame_seqlen if frame_seqlen > 0 else 3
                        if self.tri_rope_cont:
                            mem_start = tri_mem_start
                        elif self.dr_rope:
                            mem_start = 0
                        elif self.relative_rope:
                            mem_start = rr_pos['mem_start']
                        else:
                            W_frames = k_local.shape[1] // frame_seqlen if frame_seqlen > 0 else 0
                            oldest_recent_frame = current_start_frame + num_new_frames - W_frames
                            mem_start = max(0, oldest_recent_frame - Q_frames)
                        mem_grid = grid_sizes.clone()
                        mem_grid[:, 0] = Q_frames
                        mem_k_roped = causal_rope_apply(mem_k, mem_grid, freqs, start_frame=mem_start)
                        k_cat = torch.cat([k_sink, mem_k_roped, k_local], dim=1)
                        v_cat = torch.cat([v_sink, mem_v, v_local], dim=1)
                        _prof_n_mem = mem_k_roped.shape[1]
                        if self.tri_rope_cont:
                            _diag = f'[diag] layer={self._layer_id} cur_start_f={current_start_frame} cur_end_f={current_end_frame_tri} num_new_f={num_new_frames} num_new_tok={num_new_tokens} δ_cum={delta_cum} δ_tgt={delta_target} δ_eff={delta_eff} N_S={N_S} N_L_max={N_L_max} N_Q_act={N_Q_active} Q_frames(local)={Q_frames} N_S_cache={N_S_cache_tri} sink_s={tri_sink_start} mem_s={tri_mem_start} loc_s={tri_local_start} q_s={tri_q_start} q_slot_s={q_slot_start} R_active={R_active} local_end_idx={local_end_index} local_start_idx={local_start_index} loc_start_for_win={local_start_for_window} mem_k_shape1={memory_kv[0].shape[1]} self.local_attn={self.local_attn_size} max_attn={self.max_attention_size}'
                            assert tri_sink_start + N_S_cache_tri == tri_mem_start, f'TriRope-13 sink-mem gap: {tri_sink_start + N_S_cache_tri} != {tri_mem_start} | {_diag}'
                            assert tri_mem_start + Q_frames == tri_local_start, f'TriRope-13 mem-local gap: {tri_mem_start + Q_frames} != {tri_local_start} | {_diag}'
                            assert tri_q_start + num_new_frames <= self.tri_rope_pmax, f'TriRope-13 overflow: q_end={tri_q_start + num_new_frames} > pmax={self.tri_rope_pmax} | {_diag}'
                            assert tri_sink_start >= 0, f'TriRope-13 underflow: sink_start={tri_sink_start} < 0 | {_diag}'
                    else:
                        k_cat = torch.cat([k_sink, k_local], dim=1)
                        v_cat = torch.cat([v_sink, v_local], dim=1)
                        _prof_n_mem = 0
                        if self.tri_rope_cont:
                            _diag_nm = f'[diag-no-mem] layer={self._layer_id} cur_start_f={current_start_frame} cur_end_f={current_end_frame_tri} num_new_f={num_new_frames} num_new_tok={num_new_tokens} δ_cum={delta_cum} δ_tgt={delta_target} δ_eff={delta_eff} N_S={N_S} N_L_max={N_L_max} N_Q_act={N_Q_active} N_S_cache={N_S_cache_tri} sink_s={tri_sink_start} mem_s={tri_mem_start} loc_s={tri_local_start} q_s={tri_q_start} q_slot_s={q_slot_start} R_active={R_active} local_end_idx={local_end_index} local_start_idx={local_start_index} loc_start_for_win={local_start_for_window} self.local_attn={self.local_attn_size} max_attn={self.max_attention_size}'
                            expected_local_start = tri_sink_start + N_S_cache_tri
                            assert expected_local_start == tri_local_start, f'TriRope-13 sink-local gap (no mem): {expected_local_start} != {tri_local_start} | {_diag_nm}'
                            assert tri_q_start + num_new_frames <= self.tri_rope_pmax, f'TriRope-13 overflow (no mem): q_end={tri_q_start + num_new_frames} > pmax={self.tri_rope_pmax} | {_diag_nm}'
                else:
                    num_new_frames = num_new_tokens // frame_seqlen if frame_seqlen > 0 else 0
                    if self.tri_rope_cont:
                        q_slot_start = local_start_index // frame_seqlen
                        tri_q_start = tri_sink_start + N_S_cache_tri - num_new_frames
                        q_grid = grid_sizes.clone()
                        q_grid[:, 0] = num_new_frames
                        roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=tri_q_start).type_as(v)
                    elif self.dr_rope:
                        q_start_rope = local_start_index // frame_seqlen
                        q_grid = grid_sizes.clone()
                        q_grid[:, 0] = num_new_frames
                        roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=q_start_rope).type_as(v)
                    elif self.relative_rope:
                        rr_pos = CausalWanSelfAttention._compute_relative_positions(current_start_frame=current_start_frame, B=num_new_frames, R=0, N_Q=N_Q_rr, N_S=self.sink_size, pmax=self.relative_rope_pmax, num_frame_per_block=self.num_frame_per_block_attr)
                        q_grid = grid_sizes.clone()
                        q_grid[:, 0] = num_new_frames
                        roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=rr_pos['q_start']).type_as(v)
                    k_cat = k_sink
                    v_cat = v_sink
                    _prof_n_mem = 0
                if getattr(self, '_profile_attn', False):
                    with torch.no_grad():
                        _ns = sink_tokens
                        _nm = _prof_n_mem
                        _nr = k_cat.shape[1] - _ns - _nm
                        _ql = roped_query.shape[1]
                        _si = torch.linspace(0, _ql - 1, min(64, _ql), dtype=torch.long, device=roped_query.device)
                        _qs = roped_query[:, _si]
                        _sc = torch.einsum('bqhd,bkhd->bhqk', _qs.float(), k_cat.float()) * self.head_dim ** (-0.5)
                        _w = torch.softmax(_sc, dim=-1)
                        _wh_s = _w[:, :, :, :_ns].sum(-1).mean(dim=(0, 2))
                        _wh_m = _w[:, :, :, _ns:_ns + _nm].sum(-1).mean(dim=(0, 2)) if _nm > 0 else torch.zeros(self.num_heads, device=_w.device)
                        _wh_r = _w[:, :, :, _ns + _nm:].sum(-1).mean(dim=(0, 2))
                        if not hasattr(self, '_attn_profile_log'):
                            self._attn_profile_log = []
                        self._attn_profile_log.append({'sink': round(_wh_s.mean().item(), 5), 'memory': round(_wh_m.mean().item(), 5), 'recent': round(_wh_r.mean().item(), 5), 'per_head_memory': [round(x, 5) for x in _wh_m.tolist()], 'n_sink': _ns, 'n_mem': _nm, 'n_recent': _nr})
                x = attention(roped_query, k_cat, v_cat)
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                temp_k_window = temp_k[:, window_start:local_end_index]
                R_active = max(0, temp_k_window.shape[1] // frame_seqlen) if frame_seqlen > 0 else 0
                num_new_frames = num_new_tokens // frame_seqlen if frame_seqlen > 0 else 0
                if self.tri_rope_cont:
                    N_L_max_nosink = self.max_attention_size // frame_seqlen if frame_seqlen > 0 else 0
                    delta_cum_ns = max(0, current_start_frame - N_L_max_nosink)
                    delta_target_ns = max(0, self.tri_rope_pmax - N_L_max_nosink)
                    delta_eff_ns = max(0, min(delta_cum_ns, delta_target_ns))
                    local_start_rope = delta_eff_ns
                    q_start_rope = delta_eff_ns + max(0, R_active - num_new_frames)
                    if R_active > 0:
                        local_grid = grid_sizes.clone()
                        local_grid[:, 0] = R_active
                        k_win = causal_rope_apply(temp_k_window, local_grid, freqs, start_frame=local_start_rope).type_as(v)
                    else:
                        k_win = temp_k_window
                    q_grid = grid_sizes.clone()
                    q_grid[:, 0] = num_new_frames
                    roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=q_start_rope).type_as(v)
                elif self.dr_rope:
                    local_start_rope = window_start // frame_seqlen
                    q_start_rope = local_start_index // frame_seqlen
                    if R_active > 0:
                        local_grid = grid_sizes.clone()
                        local_grid[:, 0] = R_active
                        k_win = causal_rope_apply(temp_k_window, local_grid, freqs, start_frame=local_start_rope).type_as(v)
                    else:
                        k_win = temp_k_window
                    q_grid = grid_sizes.clone()
                    q_grid[:, 0] = num_new_frames
                    roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=q_start_rope).type_as(v)
                elif self.relative_rope:
                    rr_pos = CausalWanSelfAttention._compute_relative_positions(current_start_frame=current_start_frame, B=num_new_frames, R=R_active, N_Q=0, N_S=0, pmax=self.relative_rope_pmax, num_frame_per_block=self.num_frame_per_block_attr)
                    if R_active > 0:
                        local_grid = grid_sizes.clone()
                        local_grid[:, 0] = R_active
                        k_win = causal_rope_apply(temp_k_window, local_grid, freqs, start_frame=rr_pos['local_start']).type_as(v)
                    else:
                        k_win = temp_k_window
                    q_grid = grid_sizes.clone()
                    q_grid[:, 0] = num_new_frames
                    roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=rr_pos['q_start']).type_as(v)
                else:
                    k_win = temp_k_window
                x = attention(roped_query, k_win, temp_v[:, window_start:local_end_index])
        x = x.flatten(2)
        x = self.o(x)
        if kv_cache is not None:
            cache_info = (current_end, local_end_index, cache_update_info)
            if _sink_qkv_captured is not None:
                return (x, cache_info, _sink_qkv_captured)
            return (x, cache_info)
        else:
            return x

class CausalWanAttentionBlock(nn.Module):
    """Causal Wan block: cached self-attn, cross-attn, then gated FFN."""

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-06):
        """Init.

        Args:
            cross_attn_type: The cross attn type.
            dim: The dim.
            ffn_dim: The ffn dim.
            num_heads: The num heads.
            local_attn_size: The local attn size.
            sink_size: The sink size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, block_mask, kv_cache=None, crossattn_cache=None, current_start=0, cache_start=None, sink_recache_after_switch=False, memory_kv=None, capture_sink_qkv=False):
        """Forward.

        Args:
            x: The x.
            e: The e.
            seq_lens: The seq lens.
            grid_sizes: The grid sizes.
            freqs: The freqs.
            context: The context.
            context_lens: The context lens.
            block_mask: The block mask.
            kv_cache: The kv cache.
            crossattn_cache: The crossattn cache.
            current_start: The current start.
            cache_start: The cache start.
            sink_recache_after_switch: The sink recache after switch.
            memory_kv: The memory kv.
            capture_sink_qkv: The capture sink qkv.
        """
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        self_attn_result = self.self_attn((self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2), seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, memory_kv=memory_kv, capture_sink_qkv=capture_sink_qkv)
        sink_qkv_data = None
        if kv_cache is not None:
            if isinstance(self_attn_result, tuple) and len(self_attn_result) == 3:
                y, cache_update_info, sink_qkv_data = self_attn_result
            else:
                y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            """Cross attn ffn.

            Args:
                x: The x.
                context: The context.
                context_lens: The context lens.
                e: The e.
                crossattn_cache: The crossattn cache.
            """
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn((self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x
        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        if cache_update_info is not None:
            if sink_qkv_data is not None:
                return (x, cache_update_info, sink_qkv_data)
            return (x, cache_update_info)
        else:
            return x

class CausalHead(nn.Module):
    """AdaLN output head used by causal Wan graphs."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-06):
        """Init.

        Args:
            dim: The dim.
            out_dim: The out dim.
            patch_size: The patch size.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        """Forward.

        Args:
            x: The x.
            e: The e.
        """
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x

class CausalWanModel(nn.Module, NativeConfigMixin):
    """Causal Wan transformer (chunk KV cache, official or packed layout)."""
    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-06, dr_rope=False, tri_rope_cont=False, tri_rope_pmax=21, relative_rope=False, relative_rope_pmax=21):
        """Init.

        Args:
            model_type: The model type.
            patch_size: The patch size.
            text_len: The text len.
            in_dim: The in dim.
            dim: The dim.
            ffn_dim: The ffn dim.
            freq_dim: The freq dim.
            text_dim: The text dim.
            out_dim: The out dim.
            num_heads: The num heads.
            num_layers: The num layers.
            local_attn_size: The local attn size.
            sink_size: The sink size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
            dr_rope: The dr rope.
            tri_rope_cont: The tri rope cont.
            tri_rope_pmax: The tri rope pmax.
            relative_rope: The relative rope.
            relative_rope_pmax: The relative rope pmax.
        """
        super().__init__()
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.dr_rope = dr_rope
        self.tri_rope_cont = tri_rope_cont
        self.tri_rope_pmax = tri_rope_pmax
        self.relative_rope = relative_rope
        self.relative_rope_pmax = relative_rope_pmax
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)])
        for _blk_i, blk in enumerate(self.blocks):
            blk.self_attn.dr_rope = self.dr_rope
            blk.self_attn.tri_rope_cont = self.tri_rope_cont
            blk.self_attn.tri_rope_pmax = self.tri_rope_pmax
            blk.self_attn.relative_rope = self.relative_rope
            blk.self_attn.relative_rope_pmax = self.relative_rope_pmax
            blk.self_attn.num_frame_per_block_attr = 3
            blk.self_attn._layer_id = _blk_i
        if self.tri_rope_cont:
            first_attn = self.blocks[0].self_attn
            sink_s = first_attn.sink_size
            max_attn = first_attn.max_attention_size
            W_est = max(0, (max_attn - sink_s * 1560) // 1560)
            Q_max_est = 6
            B_f_assume = 3
            span = B_f_assume + W_est + Q_max_est + sink_s
            assert span <= tri_rope_pmax, f'tri_rope_cont invariant violated (init consistency check): B_f≤{B_f_assume} + W({W_est}) + Q≤{Q_max_est} + sink({sink_s}) = {span} > pmax({tri_rope_pmax})'
        if self.relative_rope:
            first_attn = self.blocks[0].self_attn
            sink_s = first_attn.sink_size
            N_L = first_attn.max_attention_size // 1560 - sink_s
            N_Q_est = 3
            pmax_m1 = self.relative_rope_pmax - 1
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask = None
        self._num_frame_per_block = 1
        self.independent_first_frame = False
        object.__setattr__(self, 'query_memory_encoder', None)
        object.__setattr__(self, 'sink_memory', None)
        object.__setattr__(self, '_ei_prev_window_start', None)

    @property
    def num_frame_per_block(self):
        """Num frame per block."""
        return self._num_frame_per_block

    @num_frame_per_block.setter
    def num_frame_per_block(self, value):
        """Num frame per block.

        Args:
            value: The value.
        """
        self._num_frame_per_block = int(value)
        for blk in self.blocks:
            blk.self_attn.num_frame_per_block_attr = int(value)

    def setup_memory_encoder(self, memory_kwargs):
        """Setup memory encoder.

        Args:
            memory_kwargs: The memory kwargs.
        """
        from .query_memory import QueryMemoryEncoder
        from types import SimpleNamespace
        if isinstance(memory_kwargs, dict):
            memory_kwargs = SimpleNamespace(**memory_kwargs)
        self.query_memory_encoder = QueryMemoryEncoder(memory_kwargs)

    def setup_sink_memory(self, memory_kwargs):
        """Setup sink memory.

        Args:
            memory_kwargs: The memory kwargs.
        """
        from .sink_memory import SinkMemory
        if isinstance(memory_kwargs, dict):
            from types import SimpleNamespace
            memory_kwargs = SimpleNamespace(**memory_kwargs)
        num_blocks = len(self.blocks)
        num_heads = self.blocks[0].self_attn.num_heads
        head_dim = self.blocks[0].self_attn.head_dim
        tokens_per_frame = getattr(memory_kwargs, 'tokens_per_frame', 1560)
        sink_size = self.blocks[0].self_attn.sink_size
        hidden_dim = self.blocks[0].dim
        sm = SinkMemory(num_blocks, num_heads, head_dim, tokens_per_frame, sink_size, hidden_dim)
        object.__setattr__(self, 'sink_memory', sm)

    def _set_gradient_checkpointing(self, module, value=False):
        """Helper function to set gradient checkpointing.

        Args:
            module: The module.
            value: The value.
        """
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(device: torch.device | str, num_frames: int=21, frame_seqlen: int=1560, num_frame_per_block=1, local_attn_size=-1) -> BlockMask:
        """Helper function to prepare blockwise causal attn mask.

        Args:
            device: The device.
            num_frames: The num frames.
            frame_seqlen: The frame seqlen.
            num_frame_per_block: The num frame per block.
            local_attn_size: The local attn size.

        Returns:
            The return value.
        """
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)
        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            """Attention mask.

            Args:
                b: The b.
                h: The h.
                q_idx: The q idx.
                kv_idx: The kv idx.
            """
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (kv_idx < ends[q_idx]) & (kv_idx >= ends[q_idx] - local_attn_size * frame_seqlen) | (q_idx == kv_idx)
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=False, device=device)
        import torch.distributed as dist
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(device: torch.device | str, num_frames: int=21, frame_seqlen: int=1560, num_frame_per_block=1) -> BlockMask:
        """Helper function to prepare teacher forcing mask.

        Args:
            device: The device.
            num_frames: The num frames.
            frame_seqlen: The frame seqlen.
            num_frame_per_block: The num frame per block.

        Returns:
            The return value.
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        clean_ends = num_frames * frame_seqlen
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(start=0, end=num_frames * frame_seqlen, step=attention_block_size, device=device, dtype=torch.long)
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size
        noisy_image_start_list = torch.arange(num_frames * frame_seqlen, total_length, step=attention_block_size, device=device, dtype=torch.long)
        noisy_image_end_list = noisy_image_start_list + attention_block_size
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            """Attention mask.

            Args:
                b: The b.
                h: The h.
                q_idx: The q idx.
                kv_idx: The kv idx.
            """
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=False, device=device)
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(device: torch.device | str, num_frames: int=21, frame_seqlen: int=1560, num_frame_per_block=4, local_attn_size=-1) -> BlockMask:
        """Helper function to prepare blockwise causal attn mask i2v.

        Args:
            device: The device.
            num_frames: The num frames.
            frame_seqlen: The frame seqlen.
            num_frame_per_block: The num frame per block.
            local_attn_size: The local attn size.

        Returns:
            The return value.
        """
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        ends[:frame_seqlen] = frame_seqlen
        frame_indices = torch.arange(start=frame_seqlen, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)
        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            """Attention mask.

            Args:
                b: The b.
                h: The h.
                q_idx: The q idx.
                kv_idx: The kv idx.
            """
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (kv_idx < ends[q_idx]) & (kv_idx >= ends[q_idx] - local_attn_size * frame_seqlen) | (q_idx == kv_idx)
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=False, device=device)
        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """Helper function to apply cache updates.

        Args:
            kv_cache: The kv cache.
            cache_update_infos: The cache update infos.
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                if update_info['action'] == 'roll_and_insert':
                    sink_tokens = update_info['sink_tokens']
                    num_rolled_tokens = update_info['num_rolled_tokens']
                    num_evicted_tokens = update_info['num_evicted_tokens']
                    local_start_index = update_info['local_start_index']
                    local_end_index = update_info['local_end_index']
                    write_start_index = update_info.get('write_start_index', local_start_index)
                    write_end_index = update_info.get('write_end_index', local_end_index)
                    new_k = update_info['new_k']
                    new_v = update_info['new_v']
                    cache['k'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['k'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache['v'][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache['v'][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    if write_end_index > write_start_index and new_k.shape[1] == write_end_index - write_start_index:
                        cache['k'][:, write_start_index:write_end_index] = new_k
                        cache['v'][:, write_start_index:write_end_index] = new_v
                elif update_info['action'] == 'direct_insert':
                    local_start_index = update_info['local_start_index']
                    local_end_index = update_info['local_end_index']
                    write_start_index = update_info.get('write_start_index', local_start_index)
                    write_end_index = update_info.get('write_end_index', local_end_index)
                    new_k = update_info['new_k']
                    new_v = update_info['new_v']
                    if write_end_index > write_start_index and new_k.shape[1] == write_end_index - write_start_index:
                        cache['k'][:, write_start_index:write_end_index] = new_k
                        cache['v'][:, write_start_index:write_end_index] = new_v
            is_recompute = False if update_info is None else update_info.get('is_recompute', False)
            if not is_recompute:
                kv_cache[block_index]['global_end_index'].fill_(current_end)
                kv_cache[block_index]['local_end_index'].fill_(local_end_index)

    def _forward_inference(self, x, t, context, seq_len, clip_fea=None, y=None, kv_cache: dict=None, crossattn_cache: dict=None, current_start: int=0, cache_start: int=0, sink_recache_after_switch=False):
        """Helper function to forward inference.

        Args:
            x: The x.
            t: The t.
            context: The context.
            seq_len: The seq len.
            clip_fea: The clip fea.
            y: The y.
            kv_cache: The kv cache.
            crossattn_cache: The crossattn cache.
            current_start: The current start.
            cache_start: The cache start.
            sink_recache_after_switch: The sink recache after switch.
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        if kv_cache is not None:
            first_global_end = kv_cache[0]['global_end_index'].item() if kv_cache else 0
            if first_global_end == 0 and current_start == 0:
                B = x[0].shape[0] if isinstance(x, list) else x.shape[0]
                if self.query_memory_encoder is not None:
                    self.query_memory_encoder.reset(batch_size=B, device=x[0].device if isinstance(x, list) else x.device, dtype=torch.bfloat16)
                if self.sink_memory is not None:
                    self.sink_memory.reset()
                self._ei_prev_window_start = None
        sm = self.sink_memory
        if sm is not None and (not sm.initialized) and (kv_cache is not None):
            sink_tok = self.blocks[0].self_attn.sink_size * 1560
            if current_start == 0 and sink_tok > 0:
                sm._pending_sink_hidden = x[:, :sink_tok].clone().detach()
            elif current_start > 0 and sm.sink_hidden is None:
                pending_hidden = getattr(sm, '_pending_sink_hidden', None)
                if pending_hidden is not None:
                    sm.initialize_hidden(pending_hidden)
                    sm._pending_sink_hidden = None
        enc = self.query_memory_encoder
        num_blocks = len(self.blocks)
        if sm is not None and sm.has_history:
            memory_kv_list = [sm.get_kv(i) for i in range(num_blocks)]
        elif enc is not None and enc.has_history:
            if enc.num_query_groups > 1:
                blocks_per_group = num_blocks // enc.num_query_groups
                group_kvs = [enc.get_kv(group_index=g) for g in range(enc.num_query_groups)]
                memory_kv_list = [group_kvs[min(i // blocks_per_group, enc.num_query_groups - 1)] for i in range(num_blocks)]
            else:
                kv = enc.get_kv()
                memory_kv_list = [kv] * num_blocks
        else:
            memory_kv_list = [None] * num_blocks
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens, block_mask=self.block_mask, sink_recache_after_switch=sink_recache_after_switch, memory_kv=None)

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """

            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward
        _need_sink_capture = sm is not None and (not sm.initialized) and (kv_cache is not None) and (current_start == 0)
        cache_update_info = None
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            kwargs['memory_kv'] = memory_kv_list[block_index]
            kwargs['capture_sink_qkv'] = _need_sink_capture
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update({'kv_cache': kv_cache[block_index], 'current_start': current_start, 'cache_start': cache_start})
                result = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, **kwargs, use_reentrant=False)
                if kv_cache is not None and isinstance(result, tuple):
                    if len(result) == 3:
                        x, block_cache_update_info, sink_qkv = result
                        if not hasattr(sm, '_pending_sink_captures'):
                            sm._pending_sink_captures = {}
                        sm._pending_sink_captures[block_index] = sink_qkv
                    else:
                        x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
            else:
                kwargs.update({'kv_cache': kv_cache[block_index], 'crossattn_cache': crossattn_cache[block_index], 'current_start': current_start, 'cache_start': cache_start})
                result = block(x, **kwargs)
                if kv_cache is not None and isinstance(result, tuple):
                    if len(result) == 3:
                        x, block_cache_update_info, sink_qkv = result
                        if not hasattr(sm, '_pending_sink_captures'):
                            sm._pending_sink_captures = {}
                        sm._pending_sink_captures[block_index] = sink_qkv
                    else:
                        x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
        if kv_cache is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            enc = getattr(self, 'query_memory_encoder', None)
        if kv_cache is not None and cache_update_infos:
            _has_memory = self.query_memory_encoder is not None or self.sink_memory is not None
            if _has_memory and cache_update_infos:
                last_block_idx = cache_update_infos[-1][0]
                last_update_info = cache_update_infos[-1][1]
                local_end_idx = last_update_info[1]
                update_dict = last_update_info[2] if len(last_update_info) > 2 else None
                sink_tok = self.blocks[0].self_attn.sink_size * 1560
                sink_frames = self.blocks[0].self_attn.sink_size
                frame_seqlen = 1560
                max_attn = self.blocks[0].self_attn.max_attention_size
                recent_window_frames = (max_attn - sink_tok) // frame_seqlen
                num_new_frames = (cache_update_infos[-1][1][0] - current_start) // frame_seqlen
                current_end_frame = current_start // frame_seqlen + num_new_frames
                oldest_recent_frame = max(sink_frames, current_end_frame - recent_window_frames)
                prev_oldest = getattr(self, '_ei_prev_window_start', None)
                if prev_oldest is None:
                    prev_oldest = sink_frames
                num_exited_frames = max(0, oldest_recent_frame - prev_oldest)
                self._ei_prev_window_start = oldest_recent_frame
                sm = self.sink_memory
                _pending = getattr(sm, '_pending_sink_captures', {}) if sm is not None else {}
                if sm is not None and (not sm.initialized) and (current_start > 0) and _pending:
                    mem_grid = grid_sizes.clone()
                    mem_grid[:, 0] = sm.sink_size
                    for blk_idx in range(len(self.blocks)):
                        if blk_idx in _pending:
                            q_pre, k_pre, v_sink = _pending[blk_idx]
                            q_roped = causal_rope_apply(q_pre, mem_grid, self.freqs, start_frame=0)
                            sm.initialize_block(blk_idx, q_roped, k_pre, v_sink)
                    sm._pending_sink_captures = {}
                if num_exited_frames > 0:
                    num_exited_tokens = num_exited_frames * frame_seqlen
                    if update_dict is not None and update_dict.get('action') == 'roll_and_insert':
                        capture_start = sink_tok
                    else:
                        capture_start = prev_oldest * frame_seqlen
                    if self.query_memory_encoder is not None:
                        cache = kv_cache[last_block_idx]
                        exited_k = cache['k'][:, capture_start:capture_start + num_exited_tokens].clone()
                        exited_v = cache['v'][:, capture_start:capture_start + num_exited_tokens].clone()
                        sink_k = cache['k'][:, :sink_tok].clone() if sink_tok > 0 else None
                        sink_v = cache['v'][:, :sink_tok].clone() if sink_tok > 0 else None
                        self.query_memory_encoder.update(exited_k, exited_v, sink_k, sink_v)
                    if sm is not None and sm.initialized:
                        evicted_kv_all = []
                        for blk_idx in range(len(self.blocks)):
                            cache_blk = kv_cache[blk_idx]
                            ek = cache_blk['k'][:, capture_start:capture_start + num_exited_tokens].clone()
                            ev = cache_blk['v'][:, capture_start:capture_start + num_exited_tokens].clone()
                            evicted_kv_all.append((ek, ev))
                        sm.update(self.blocks, evicted_kv_all, e0, context, context_lens, self.freqs, grid_sizes, evicted_k_is_pre_rope=self.dr_rope or self.tri_rope_cont or self.relative_rope)
            self._apply_cache_updates(kv_cache, cache_update_infos)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(self, x, t, context, seq_len, clean_x=None, aug_t=None, clip_fea=None, y=None):
        """Helper function to forward train.

        Args:
            x: The x.
            t: The t.
            context: The context.
            seq_len: The seq len.
            clean_x: The clean x.
            aug_t: The aug t.
            clip_fea: The clip fea.
            y: The y.
        """
        pass
        raise NotImplementedError()
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(device, num_frames=x.shape[2], frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]), num_frame_per_block=self.num_frame_per_block)
            elif self.independent_first_frame:
                self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(device, num_frames=x.shape[2], frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]), num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(device, num_frames=x.shape[2], frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]), num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1) for u in x])
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x])
            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens, block_mask=self.block_mask)

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """

            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        """Forward."""
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        """Unpatchify.

        Args:
            x: The x.
            grid_sizes: The grid sizes.
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Init weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
