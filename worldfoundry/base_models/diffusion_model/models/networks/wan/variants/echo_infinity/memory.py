"""Echo-Infinity memory-augmented causal Wan (hidden-state bank).

Stores and replays memory hidden states across clips.  Used with
:mod:`.infinity` / :mod:`.causal`.  Memory adapter only.
"""
import math
import types
from types import SimpleNamespace
import torch
import torch.distributed as dist
from .attention import attention
from .model import sinusoidal_embedding_1d
from .infinity import CausalWanModel as CausalWanModelInfinityBase
from .causal import CausalWanSelfAttention as _RelropeSelfAttention, causal_rope_apply
_FIRST_LONG_LOGGED = {'flag': False}
_FIRST_BULK_LOGGED = {'flag': False}

def _self_attn_inf_mem_forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, cache_start=None, sink_recache_after_switch=False, memory_kv=None, capture_sink_qkv=False):
    """Helper function to self attn inf mem forward.

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

    def qkv_fn(_x):
        """Qkv fn.

        Args:
            _x: The x.
        """
        q = self.norm_q(self.q(_x)).view(b, s, n, d)
        k = self.norm_k(self.k(_x)).view(b, s, n, d)
        v = self.v(_x).view(b, s, n, d)
        return (q, k, v)
    q, k, v = qkv_fn(x)
    if kv_cache is None:
        raise NotImplementedError('CausalWanModelInfinityMemory only supports inference-time kv_cache forward; training path is intentionally removed.')
    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    num_new_frames = grid_sizes[0][0].item()
    current_end = current_start + q.shape[1]
    sink_tokens = self.sink_size * frame_seqlen
    sink_size_frames = self.sink_size
    kv_cache_size = kv_cache['k'].shape[1]
    num_new_tokens = q.shape[1]
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
            temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        cache_update_info = {'action': 'roll_and_insert', 'sink_tokens': sink_tokens, 'num_rolled_tokens': num_rolled_tokens, 'num_evicted_tokens': num_evicted_tokens, 'local_start_index': local_start_index, 'local_end_index': local_end_index, 'write_start_index': write_start_index, 'write_end_index': local_end_index, 'new_k': k[:, roped_offset:roped_offset + write_len], 'new_v': v[:, roped_offset:roped_offset + write_len], 'current_end': current_end, 'is_recompute': is_recompute}
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
            temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        cache_update_info = {'action': 'direct_insert', 'local_start_index': local_start_index, 'local_end_index': local_end_index, 'write_start_index': write_start_index, 'write_end_index': local_end_index, 'new_k': k[:, roped_offset:roped_offset + write_len], 'new_v': v[:, roped_offset:roped_offset + write_len], 'current_end': current_end, 'is_recompute': is_recompute}
    num_cache_frames = local_end_index // frame_seqlen
    R_active = num_cache_frames - sink_size_frames
    N_Q_rr = memory_kv[0].shape[1] // frame_seqlen if memory_kv is not None else 0
    current_start_frame = current_start // frame_seqlen
    num_frame_per_block_attr = getattr(self, 'num_frame_per_block_attr', 3)
    relative_rope_pmax = getattr(self, 'relative_rope_pmax', 21)
    rr_pos = _RelropeSelfAttention._compute_relative_positions(current_start_frame=current_start_frame, B=num_new_frames, R=R_active, N_Q=N_Q_rr, N_S=sink_size_frames, pmax=relative_rope_pmax, num_frame_per_block=num_frame_per_block_attr)
    _diag = f"[InfMem-diag] layer={getattr(self, '_layer_id', -1)} cur_start_f={current_start_frame} B={num_new_frames} R={R_active} N_Q={N_Q_rr} N_S={sink_size_frames} pmax={relative_rope_pmax} is_bulk={rr_pos['is_bulk_forward']} use_mem={rr_pos['use_memory']} sink_s={rr_pos['sink_start']} mem_s={rr_pos['mem_start']} mem_e={rr_pos['mem_end']} local_s={rr_pos['local_start']} local_e={rr_pos['local_end']} q_s={rr_pos['q_start']} q_l={rr_pos['q_last']}"
    assert rr_pos['q_last'] <= relative_rope_pmax - 1, f'InfMem overflow: q_last > pmax-1 | {_diag}'
    assert rr_pos['q_start'] >= 0, f'InfMem underflow: q_start < 0 | {_diag}'
    if rr_pos['use_memory']:
        assert rr_pos['mem_start'] >= sink_size_frames, f'InfMem mem overlaps sink | {_diag}'
        assert rr_pos['mem_end'] + 1 == rr_pos['local_start'], f'InfMem mem-local not contiguous | {_diag}'
    if R_active > 0 and (not rr_pos['is_bulk_forward']):
        assert rr_pos['q_last'] == rr_pos['local_end'], f'InfMem Q-local tail mismatch | {_diag}'
    if sink_tokens > 0:
        k_sink_raw = temp_k[:, :sink_tokens]
        v_sink = temp_v[:, :sink_tokens]
        sink_grid = grid_sizes.clone()
        sink_grid[:, 0] = sink_size_frames
        k_sink = causal_rope_apply(k_sink_raw, sink_grid, freqs, start_frame=0).type_as(v)
    else:
        k_sink = None
        v_sink = None
    if R_active > 0:
        k_local_raw = temp_k[:, sink_tokens:local_end_index]
        v_local = temp_v[:, sink_tokens:local_end_index]
        local_grid = grid_sizes.clone()
        local_grid[:, 0] = R_active
        k_local = causal_rope_apply(k_local_raw, local_grid, freqs, start_frame=rr_pos['local_start']).type_as(v)
    else:
        k_local = None
        v_local = None
    q_grid = grid_sizes.clone()
    q_grid[:, 0] = num_new_frames
    roped_query = causal_rope_apply(q, q_grid, freqs, start_frame=rr_pos['q_start']).type_as(v)
    mem_k_roped = None
    mem_v = None
    if memory_kv is not None and rr_pos['use_memory'] and (not rr_pos['is_bulk_forward']):
        mem_k_raw, mem_v_raw = memory_kv
        N_Q = mem_k_raw.shape[1] // frame_seqlen
        mem_grid = grid_sizes.clone()
        mem_grid[:, 0] = N_Q
        mem_k_roped = causal_rope_apply(mem_k_raw, mem_grid, freqs, start_frame=rr_pos['mem_start']).type_as(v)
        mem_v = mem_v_raw
    parts_k, parts_v = ([], [])
    if k_sink is not None:
        parts_k.append(k_sink)
        parts_v.append(v_sink)
    if mem_k_roped is not None:
        parts_k.append(mem_k_roped)
        parts_v.append(mem_v)
    if k_local is not None:
        parts_k.append(k_local)
        parts_v.append(v_local)
    k_cat = torch.cat(parts_k, dim=1)
    v_cat = torch.cat(parts_v, dim=1)
    _layer_id = getattr(self, '_layer_id', -1)
    if _layer_id == 0:
        if not rr_pos['is_bulk_forward'] and rr_pos['q_last'] == relative_rope_pmax - 1 and (not _FIRST_LONG_LOGGED['flag']):
            mem_str = f"mem[{rr_pos['mem_start']}..{rr_pos['mem_end']}]" if rr_pos['use_memory'] else 'mem=inactive'
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[InfMem] FIRST LONG PHASE @ t={current_start_frame}: sink[0..{sink_size_frames - 1}] | {mem_str} | local[{rr_pos['local_start']}..{rr_pos['local_end']}] | Q[{rr_pos['q_start']}..{rr_pos['q_last']}]", flush=True)
            _FIRST_LONG_LOGGED['flag'] = True
        if rr_pos['is_bulk_forward'] and (not _FIRST_BULK_LOGGED['flag']):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[InfMem] FIRST BULK FORWARD @ t={current_start_frame}, B={num_new_frames}, R={R_active}: sink[0..{sink_size_frames - 1}] | gap | local[{rr_pos['local_start']}..{rr_pos['local_end']}] | Q[{rr_pos['q_start']}..{rr_pos['q_last']}] (memory skipped)", flush=True)
            _FIRST_BULK_LOGGED['flag'] = True
    x = attention(roped_query, k_cat, v_cat)
    x = x.flatten(2)
    x = self.o(x)
    return (x, (current_end, local_end_index, cache_update_info))

def _block_inf_mem_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, block_mask, kv_cache=None, crossattn_cache=None, current_start=0, cache_start=None, sink_recache_after_switch=False, memory_kv=None, capture_sink_qkv=False):
    """Helper function to block inf mem forward.

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
    e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
    self_attn_result = self.self_attn((self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e_mod[1]) + e_mod[0]).flatten(1, 2), seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, memory_kv=memory_kv, capture_sink_qkv=capture_sink_qkv)
    if kv_cache is not None:
        y, cache_update_info = self_attn_result
    else:
        y = self_attn_result
        cache_update_info = None
    x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[2]).flatten(1, 2)
    x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
    y = self.ffn((self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e_mod[4]) + e_mod[3]).flatten(1, 2))
    x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[5]).flatten(1, 2)
    if cache_update_info is not None:
        return (x, cache_update_info)
    else:
        return x

class CausalWanModelInfinityMemory(CausalWanModelInfinityBase):
    """Echo-Infinity causal Wan that reads/writes the hidden-state memory bank."""

    def enable_infmem(self, relative_rope: bool=True, relative_rope_pmax: int=21, num_frame_per_block_attr: int=3):
        """Enable infmem.

        Args:
            relative_rope: The relative rope.
            relative_rope_pmax: The relative rope pmax.
            num_frame_per_block_attr: The num frame per block attr.
        """
        self.relative_rope = relative_rope
        self.relative_rope_pmax = relative_rope_pmax
        self.num_frame_per_block_attr = num_frame_per_block_attr
        for i, block in enumerate(self.blocks):
            sa = block.self_attn
            sa.relative_rope = relative_rope
            sa.relative_rope_pmax = relative_rope_pmax
            sa.num_frame_per_block_attr = num_frame_per_block_attr
            sa._layer_id = i
            sa.forward = types.MethodType(_self_attn_inf_mem_forward, sa)
            block.forward = types.MethodType(_block_inf_mem_forward, block)
        object.__setattr__(self, 'query_memory_encoder', None)
        object.__setattr__(self, 'sink_memory', None)
        self._ei_prev_window_start = None
        _FIRST_LONG_LOGGED['flag'] = False
        _FIRST_BULK_LOGGED['flag'] = False

    def setup_memory_encoder(self, memory_kwargs):
        """Setup memory encoder.

        Args:
            memory_kwargs: The memory kwargs.
        """
        from .query_memory import QueryMemoryEncoder
        if isinstance(memory_kwargs, dict):
            memory_kwargs = SimpleNamespace(**memory_kwargs)
        self.query_memory_encoder = QueryMemoryEncoder(memory_kwargs)

    @property
    def num_frame_per_block(self):
        """Num frame per block."""
        return getattr(self, '_num_frame_per_block', 1)

    @num_frame_per_block.setter
    def num_frame_per_block(self, value):
        """Num frame per block.

        Args:
            value: The value.
        """
        self._num_frame_per_block = int(value)
        for blk in self.blocks:
            blk.self_attn.num_frame_per_block_attr = int(value)

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
                B = x.shape[0]
                if self.query_memory_encoder is not None:
                    self.query_memory_encoder.reset(batch_size=B, device=x.device, dtype=torch.bfloat16)
                self._ei_prev_window_start = None
        enc = self.query_memory_encoder
        num_blocks = len(self.blocks)
        if enc is not None and getattr(enc, 'has_history', False):
            num_query_groups = getattr(enc, 'num_query_groups', 1)
            if num_query_groups > 1:
                blocks_per_group = num_blocks // num_query_groups
                group_kvs = [enc.get_kv(group_index=g) for g in range(num_query_groups)]
                memory_kv_list = [group_kvs[min(i // blocks_per_group, num_query_groups - 1)] for i in range(num_blocks)]
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

            def custom_forward(*inputs, **_kwargs):
                """Custom forward."""
                return module(*inputs, **_kwargs)
            return custom_forward
        cache_update_info = None
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            kwargs['memory_kv'] = memory_kv_list[block_index]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update({'kv_cache': kv_cache[block_index], 'current_start': current_start, 'cache_start': cache_start})
                result = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, **kwargs, use_reentrant=False)
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
            else:
                kwargs.update({'kv_cache': kv_cache[block_index], 'crossattn_cache': crossattn_cache[block_index], 'current_start': current_start, 'cache_start': cache_start})
                result = block(x, **kwargs)
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    cache_update_info = block_cache_update_info[:2]
                else:
                    x = result
        if kv_cache is not None and cache_update_infos:
            if self.query_memory_encoder is not None:
                last_block_idx = cache_update_infos[-1][0]
                last_update_info = cache_update_infos[-1][1]
                update_dict = last_update_info[2] if len(last_update_info) > 2 else None
                sink_tok = self.blocks[0].self_attn.sink_size * 1560
                sink_frames = self.blocks[0].self_attn.sink_size
                frame_seqlen = 1560
                max_attn = self.blocks[0].self_attn.max_attention_size
                recent_window_frames = (max_attn - sink_tok) // frame_seqlen
                num_new_frames = (cache_update_infos[-1][1][0] - current_start) // frame_seqlen
                current_end_frame = current_start // frame_seqlen + num_new_frames
                oldest_recent_frame = max(sink_frames, current_end_frame - recent_window_frames)
                prev_oldest = self._ei_prev_window_start if self._ei_prev_window_start is not None else sink_frames
                num_exited_frames = max(0, oldest_recent_frame - prev_oldest)
                self._ei_prev_window_start = oldest_recent_frame
                if num_exited_frames > 0:
                    num_exited_tokens = num_exited_frames * frame_seqlen
                    if update_dict is not None and update_dict.get('action') == 'roll_and_insert':
                        capture_start = sink_tok
                    else:
                        capture_start = prev_oldest * frame_seqlen
                    cache_blk = kv_cache[last_block_idx]
                    exited_k = cache_blk['k'][:, capture_start:capture_start + num_exited_tokens].clone()
                    exited_v = cache_blk['v'][:, capture_start:capture_start + num_exited_tokens].clone()
                    sink_k = cache_blk['k'][:, :sink_tok].clone() if sink_tok > 0 else None
                    sink_v = cache_blk['v'][:, :sink_tok].clone() if sink_tok > 0 else None
                    self.query_memory_encoder.update(exited_k, exited_v, sink_k, sink_v)
            self._apply_cache_updates(kv_cache, cache_update_infos)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)
