"""Echo-1 Infinity long-horizon causal Wan (relativistic RoPE + memory).

Extends the causal graph with block-relativistic RoPE and hooks for
query / sink memory so rollouts can keep a compact history.

Adds causal attention and memory hooks.
"""
from .attention import attention
from .model import WanRMSNorm, rope_apply, WanLayerNorm, WAN_CROSSATTENTION_CLASSES, rope_params, MLPProj, sinusoidal_embedding_1d
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
import torch.nn as nn
import torch
import math
import torch.distributed as dist
flex_attention = torch.compile(flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs')

def block_relativistic_rope(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    """Block relativistic rope.

    Args:
        x: The x.
        grid_sizes: The grid sizes.
        freqs: The freqs.
        start_frame: The start frame.
        relative_frame_indices: The relative frame indices.
    """
    n, c = (x.size(2), x.size(3) // 2)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        if relative_frame_indices is not None:
            frame_indices = relative_frame_indices.long()
            freqs_temporal = freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
        else:
            freqs_temporal = freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_i = torch.cat([freqs_temporal, freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
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

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, cache_start=None, sink_recache_after_switch=False):
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
            num_new_frames = grid_sizes[0][0].item()
            current_end = current_start + q.shape[1]
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
                    temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
                query_relative_indices = torch.arange(self.local_attn_size - num_new_frames, self.local_attn_size, device=q.device)
                roped_query = block_relativistic_rope(q, grid_sizes, freqs, relative_frame_indices=query_relative_indices).type_as(v)
                num_cache_frames = local_end_index // frame_seqlen
                cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                cache_grid_sizes = grid_sizes.clone()
                cache_grid_sizes[0, 0] = num_cache_frames
                roped_temp_k = block_relativistic_rope(temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2), cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices).type_as(v)
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
                current_frame_in_window = local_start_index // frame_seqlen
                query_relative_indices = torch.arange(current_frame_in_window, current_frame_in_window + num_new_frames, device=q.device)
                roped_query = block_relativistic_rope(q, grid_sizes, freqs, relative_frame_indices=query_relative_indices).type_as(v)
                num_cache_frames = local_end_index // frame_seqlen
                cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                cache_grid_sizes = grid_sizes.clone()
                cache_grid_sizes[0, 0] = num_cache_frames
                roped_temp_k = block_relativistic_rope(temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2), cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices).type_as(v)
                cache_update_info = {'action': 'direct_insert', 'local_start_index': local_start_index, 'local_end_index': local_end_index, 'write_start_index': write_start_index, 'write_end_index': local_end_index, 'new_k': k[:, roped_offset:roped_offset + write_len], 'new_v': v[:, roped_offset:roped_offset + write_len], 'current_end': current_end, 'is_recompute': is_recompute}
            if sink_tokens > 0:
                local_budget = self.max_attention_size - sink_tokens
                k_sink = roped_temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = roped_temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                else:
                    k_cat = k_sink
                    v_cat = v_sink
                x = attention(roped_query, k_cat, v_cat)
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                x = attention(roped_query, roped_temp_k[:, window_start:local_end_index], temp_v[:, window_start:local_end_index])
        x = x.flatten(2)
        x = self.o(x)
        if kv_cache is not None:
            return (x, (current_end, local_end_index, cache_update_info))
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

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, block_mask, kv_cache=None, crossattn_cache=None, current_start=0, cache_start=None, sink_recache_after_switch=False):
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
        """
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        self_attn_result = self.self_attn((self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2), seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch)
        if kv_cache is not None:
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
    def __init__(self, model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-06):
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
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)])
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False

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
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens, block_mask=self.block_mask, sink_recache_after_switch=sink_recache_after_switch)

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """

            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward
        cache_update_info = None
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
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
