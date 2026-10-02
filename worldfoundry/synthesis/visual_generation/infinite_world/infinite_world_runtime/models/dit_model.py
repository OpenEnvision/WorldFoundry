# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from worldfoundry.core.distributed.model_parallel import context_state as context_parallel_util
from worldfoundry.core import autocast_context

from .checkpoint import auto_grad_checkpoint

try:
    from transformer_engine.pytorch.attention import DotProductAttention
    _TRANSFORMER_ENGINE_IMPORT_ERROR = None
except Exception as exc:
    DotProductAttention = None
    _TRANSFORMER_ENGINE_IMPORT_ERROR = exc

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

__all__ = ['WanModel']

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.nonlinearity = nn.SiLU()
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class TemporalDownsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 时序下采样: kernel=3, stride=(2,1,1), padding=(1,1,1)
        # T -> T/2, H, W 保持不变
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


class WanEncoderAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=(-1, -1), eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 内部使用 WanSelfAttention，保持与主干网络一致的 3D RoPE 和 FlashAttention
        self.attn = WanSelfAttention(
            dim, 
            num_heads, 
            window_size=window_size, 
            qk_norm=True, 
            eps=eps
        )
        
        # Pre-Norm
        self.norm = WanLayerNorm(dim, eps)

    def _build_freqs(self, device):
        # 构建 RoPE 频率参数
        d = self.head_dim
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        return freqs.to(device)

    def forward(self, x):
        # Input: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        # 1. 转换格式: (B, C, T, H, W) -> (B, L, C)
        # 先 permute 到 (B, T, H, W, C)，再 flatten
        x_in = x.permute(0, 2, 3, 4, 1).flatten(1, 3)
        
        # 2. Norm
        x_norm = self.norm(x_in)
        
        # 3. 构造 Metadata
        # grid_sizes: [B, 3] -> [[T, H, W], ...]
        grid_sizes = torch.tensor([T, H, W], device=x.device).unsqueeze(0).repeat(B, 1)
        
        # seq_lens: [B]
        seq_lens = torch.tensor([T * H * W] * B, device=x.device, dtype=torch.long)
        
        # freqs: RoPE (可以考虑缓存，这里为了独立性实时生成)
        freqs = self._build_freqs(x.device)
        
        # 4. Attention Forward
        # Encoder 内部通常不需要 causal mask 或 ignore mask
        x_out = self.attn(
            x_norm, 
            seq_lens=seq_lens, 
            grid_sizes=grid_sizes, 
            freqs=freqs, 
            token_ignore_mask=None
        )
        
        # 5. Residual + 恢复形状
        x_out = x_in + x_out
        
        # (B, L, C) -> (B, T, H, W, C) -> (B, C, T, H, W)
        x_out = x_out.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x_out


class TemporalLatentEncoder(nn.Module):
    def __init__(self, in_channels=16, hidden_dim=256, num_heads=8, use_checkpoint=True):
        """
        高配版时序 Encoder
        结构: ConvIn -> ResBlock*2 -> Down -> ResBlock*2 -> Down -> ResBlock -> WanAttn -> ResBlock -> ConvOut
        输入输出: (B, 16, T, H, W) -> (B, 16, T/4, H, W)
        
        Args:
            use_checkpoint: 是否使用 gradient checkpointing 节省显存（默认开启）
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # 1. Initial Conv
        self.conv_in = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        # 2. Down Block 1 (T -> T/2)
        self.down_block1 = nn.Sequential(
            ResnetBlock3D(hidden_dim, hidden_dim),
            ResnetBlock3D(hidden_dim, hidden_dim),
            TemporalDownsample(hidden_dim)
        )
        
        # 3. Down Block 2 (T/2 -> T/4)
        self.down_block2 = nn.Sequential(
            ResnetBlock3D(hidden_dim, hidden_dim),
            ResnetBlock3D(hidden_dim, hidden_dim),
            TemporalDownsample(hidden_dim)
        )
        
        # 4. Mid Block (Res + WanAttention + Res)
        self.mid_block = nn.Sequential(
            ResnetBlock3D(hidden_dim, hidden_dim),
            WanEncoderAttentionBlock(dim=hidden_dim, num_heads=num_heads), # 使用 Wanx 风格 Attention
            ResnetBlock3D(hidden_dim, hidden_dim),
        )
        
        # 5. Output Projection
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6, affine=True)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv3d(hidden_dim, in_channels, kernel_size=3, stride=1, padding=1)

    def _forward_down_block1(self, x):
        return self.down_block1(x)
    
    def _forward_down_block2(self, x):
        return self.down_block2(x)
    
    def _forward_mid_block(self, x):
        return self.mid_block(x)

    def forward(self, x):
        # x: (B, C, T, H, W)
        from torch.utils.checkpoint import checkpoint
        
        x = self.conv_in(x)
        
        # 🔴 使用 gradient checkpointing 节省显存
        if self.use_checkpoint and self.training:
            x = checkpoint(self._forward_down_block1, x, use_reentrant=False)
            x = checkpoint(self._forward_down_block2, x, use_reentrant=False)
            x = checkpoint(self._forward_mid_block, x, use_reentrant=False)
        else:
            x = self.down_block1(x)
            x = self.down_block2(x)
            x = self.mid_block(x)
        
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        
        return x

def temporal_sample(x: torch.Tensor, rate: int, dim: int = 2) -> torch.Tensor:
    """
    在指定维度采样，首尾必保留
    
    Args:
        x (torch.Tensor): 输入张量，默认 shape = (B, C, T, H, W)
        rate (int): 采样率（步长）
        dim (int): 采样的维度，默认=2 (T维)
        
    Returns:
        torch.Tensor: 采样后的张量
    """
    assert x.dim() >= dim + 1, f"输入维度 {x.dim()} 小于 dim={dim}"
    N = x.shape[dim]

    # 初步采样下标
    indices = torch.arange(0, N, step=rate, device=x.device)
    
    # 确保首尾都在
    if indices[0] != 0:
        indices = torch.cat([torch.tensor([0], device=x.device), indices])
    if indices[-1] != N - 1:
        indices = torch.cat([indices, torch.tensor([N - 1], device=x.device)])
    
    # 去重并排序
    indices = torch.unique(indices, sorted=True)

    return torch.index_select(x, dim, indices)

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif not FLASH_ATTN_2_AVAILABLE:
        # CUDA stacks. Fall back to PyTorch SDPA so small demos can still run.
        q_offset = 0
        k_offset = 0
        outputs = q.new_zeros((b, lq, q.size(1), v.size(-1)))
        for batch_idx in range(b):
            q_len = int(q_lens[batch_idx].item())
            k_len = int(k_lens[batch_idx].item())
            qi = q[q_offset:q_offset + q_len].permute(1, 0, 2).unsqueeze(0)
            ki = k[k_offset:k_offset + k_len].permute(1, 0, 2).unsqueeze(0)
            vi = v[k_offset:k_offset + k_len].permute(1, 0, 2).unsqueeze(0)
            if qi.size(1) != ki.size(1):
                repeat = qi.size(1) // ki.size(1)
                ki = ki.repeat_interleave(repeat, dim=1)
                vi = vi.repeat_interleave(repeat, dim=1)
            oi = _worldfoundry_scaled_dot_product_attention(
                qi,
                ki,
                vi,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            outputs[batch_idx, :q_len] = oi.squeeze(0).permute(1, 0, 2)
            q_offset += q_len
            k_offset += k_len
        x = outputs
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    with autocast_context("cuda", enabled=False):
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta,
                            torch.arange(0, dim, 2).to(torch.float64).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


def rope_apply(x, grid_sizes, freqs, enable_context_parallel=False):
    with autocast_context("cuda", enabled=False):
        s, n, c = x.size(1), x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
                s, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

            if enable_context_parallel:
                freqs_i = rearrange(freqs_i, "(T S) B C -> T S B C", T=f)
                freqs_i = context_parallel_util.split_cp(freqs_i, seq_dim=1)
                freqs_i = rearrange(freqs_i, "T S B C -> (T S) B C")

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class ActionEncoder(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=256, hidden_dim=512, out_dim=1536):
        super().__init__()
        # 将整数映射到向量
        self.embedding_move = nn.Embedding(vocab_size, embed_dim)
        self.embedding_view = nn.Embedding(vocab_size, embed_dim)

        self.encode_1 = nn.Sequential(
            nn.Conv1d(embed_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, hidden_dim),
            nn.ReLU(),
        )

        self.encode_2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, hidden_dim),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, move, view):
        # x: (B, L+1)，整数输入
        x_move = self.embedding_move(move).transpose(1, 2)
        x_view = self.embedding_view(view).transpose(1, 2)
        x = torch.cat([x_move, x_view], dim=1)

        x = self.encode_2(self.encode_1(x))       # (B, out_dim, (L+1)/4)

        x = x.transpose(1, 2)             # (B, (L/4)+1, out_dim)
        x = self.proj(x)
        return x

class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(), 
            self.normalized_shape, 
            None if self.weight is None else self.weight.float(), 
            None if self.bias is None else self.bias.float() ,
            self.eps
        ).to(origin_dtype)
        return out


class WanSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        enable_context_parallel=False,
        fp32_infer=False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.enable_context_parallel = enable_context_parallel

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        if self.enable_context_parallel:
            if DotProductAttention is None:
                raise ImportError(
                    "transformer_engine is required when enable_context_parallel=True"
                ) from _TRANSFORMER_ENGINE_IMPORT_ERROR
            qkv_format = "bshd"
            attn_mask_type = "no_mask"
            os.environ["NVTE_FUSED_ATTN"] = "0"
            os.environ["NVTE_FLASH_ATTN"] = "1"
            self.core_attn = DotProductAttention(
                self.num_heads,
                self.head_dim,
                num_gqa_groups=self.num_heads,
                qkv_format=qkv_format,
                attn_mask_type=attn_mask_type,
            )
            self.core_attn.set_context_parallel_group(context_parallel_util.get_cp_group(), 
                                                      context_parallel_util.get_cp_rank_list(), 
                                                      context_parallel_util.get_cp_stream())
            
        self.fp32_infer = fp32_infer
        self.out_c = None

    def forward(self, x, seq_lens, grid_sizes, freqs, token_ignore_mask=None, dtype=torch.bfloat16):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            token_ignore_mask: [B, N]; bool tensor indicating tokens to be ignored
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs, enable_context_parallel=self.enable_context_parallel)
        k = rope_apply(k, grid_sizes, freqs, enable_context_parallel=self.enable_context_parallel)

        # maks implementation by setting KV to zero
        # this is a hack for the sake of cp support
        if token_ignore_mask is not None:
            select_mask = ~token_ignore_mask
            expanded_select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim) # [B, N, H, D]
            expanded_select_mask = expanded_select_mask.to(k.dtype)
            k = k * expanded_select_mask
            v = v * expanded_select_mask

        if self.enable_context_parallel:
            # cp_size = context_parallel_util.get_cp_size()
            # half_dtypes = (torch.float16, torch.bfloat16)
            # def half(x):
            #     return x if x.dtype in half_dtypes else x.to(dtype)

            # max_seqlen_q = s * cp_size
            # max_seqlen_kv = max_seqlen_q
            # x = self.core_attn(
            #     half(q) if self.fp32_infer else q.type_as(x),
            #     half(k) if self.fp32_infer else k.type_as(x),
            #     half(v) if self.fp32_infer else v.type_as(x),
            #     core_attention_bias_type="no_bias",
            #     core_attention_bias=None,
            #     cu_seqlens_q=None,
            #     cu_seqlens_kv=None,
            #     max_seqlen_q=max_seqlen_q,
            #     max_seqlen_kv=max_seqlen_kv,
            # )
            # x = rearrange(x, "B S (H D) -> B S H D", H=self.num_heads)
            raise(NotImplementedError)
        else:
            B, S, H, D = q.shape
            # 👉 你需要提前传入 num_c（或在这里根据场景算出）
            num_c = getattr(self, "num_c", 0)
            if num_c > 0 and num_c < S:
                # 2️⃣ 当前 noisy 帧 Qz 看 [Kc; Kz]
                q_z, k_z, v_z = q[:, num_c:], k, v
                x = flash_attention(q_z, k_z, v_z, window_size=self.window_size).type_as(x)
            else:
                # 没有分段信息，默认用标准路径
                x = flash_attention(q, k, v, k_lens=seq_lens, window_size=self.window_size).type_as(x)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        enable_context_parallel=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.enable_context_parallel = enable_context_parallel

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, enable_context_parallel=enable_context_parallel)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.hist = None
        self.hist_cross = None

    def forward(
        self,
        x,
        e_all,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        token_ignore_mask=None,
        training=True
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            token_ignore_mask: [B, N]; bool tensor indicating tokens to be ignored in self attention
        """
        dtype = x.dtype
        e, e_no_noise = e_all[0], e_all[1]
        assert e.dtype == torch.float32
        assert e_no_noise.dtype == torch.float32
        with autocast_context(x, dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
            e_no_noise = (self.modulation + e_no_noise).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        num_hist = getattr(self.self_attn, "num_c", 0)
        hist, noisy = x[:, :num_hist], x[:, num_hist:]
        _, H, W = grid_sizes[0].tolist()  # 假设所有样本一致
        B = grid_sizes.shape[0]
        T_noisy = noisy.shape[1] // (H * W)
        T_hist= hist.shape[1] // (H * W)

        grid_sizes_noisy = torch.tensor([T_noisy, H, W], device=grid_sizes.device).unsqueeze(0).repeat(B, 1)
        grid_sizes_hist = torch.tensor([T_hist, H, W], device=grid_sizes.device).unsqueeze(0).repeat(B, 1)

        # print(x.shape, e[1].shape, e[0].shape)
        # self-attention
        
        seq_len_hist = torch.tensor([u.size(0) for u in hist], dtype=torch.long)
        if training or self.hist is None or self.hist.shape[1] != num_hist:
            if token_ignore_mask is not None:
                hist_token_ignore_mask = token_ignore_mask[:, :num_hist]
            else:
                hist_token_ignore_mask = token_ignore_mask
            y = self.self_attn(
                (self.norm1(hist).float() * (1 + e_no_noise[1]) + e_no_noise[0]).type_as(x), seq_len_hist, grid_sizes_hist,
                freqs, hist_token_ignore_mask)
            with autocast_context(x, dtype=torch.float32):
                self.hist = hist + y * e_no_noise[2]

        # print('recompute condition', x.shape)
        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, grid_sizes,
            freqs, token_ignore_mask)
        with autocast_context(x, dtype=torch.float32):
            noisy = noisy + y * e[2]

        x = torch.cat([self.hist, noisy], dim=1)
        x = x.to(dtype)
        # print('after self attn', x.shape)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # print('before cross attn', x.shape)
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            # print('after cross attn', x.shape)
            hist, noisy = x[:, :num_hist], x[:, num_hist:]
            
            y = self.ffn((self.norm2(noisy).float() * (1 + e[4]) + e[3]).to(dtype))
            with autocast_context(x, dtype=torch.float32):
                noisy = noisy + y * e[5]
            
            if training or self.hist_cross is None or self.hist_cross.shape[1] != num_hist:
                y = self.ffn((self.norm2(hist).float() * (1 + e_no_noise[4]) + e_no_noise[3]).to(dtype))
                with autocast_context(x, dtype=torch.float32):
                    self.hist_cross = hist + y * e_no_noise[5]
                # print('compute hist cross', self.hist_cross.shape, hist.shape, noisy.shape, x.shape)
            x = torch.cat([self.hist_cross, noisy], dim=1)
            # print('after ffn', self.hist_cross.shape, hist.shape, noisy.shape, x.shape)

            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        x = x.to(dtype)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with autocast_context(x, dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        model_max_length=512,
        in_channels=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        caption_channels=4096,
        out_channels=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        enable_context_parallel=False,
        use_convenc=True,  # 🔴 新增参数：是否使用卷积编码器进行时序压缩
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            model_max_length (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_channels (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            caption_channels (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_channels (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.model_max_length = model_max_length
        self.in_channels = in_channels
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.caption_channels = caption_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.enable_context_parallel = enable_context_parallel
        self.use_convenc = use_convenc  # 🔴 保存参数

        # hack y_embedder, not support uncond training now, pls use negative prompt for uncond
        self.y_embedder = None

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(caption_channels, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.action_encoder = ActionEncoder()
        # 🔴 只在 use_convenc=True 时创建时序编码器
        if self.use_convenc:
            self.latent_encoder = TemporalLatentEncoder()
        else:
            self.latent_encoder = None
        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              enable_context_parallel=enable_context_parallel,)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_channels, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        y,
        y_mask=None,
        x_ignore_mask=None,
        clip_fea=None,
        image_cond=None,
        move=None,
        view=None
    ):
        r"""
        Forward pass through the diffusion model
        """

        COMPRESSION_RATE = 4
        MAX_T_OUT = 20                 
        TARGET_T_MID = MAX_T_OUT * COMPRESSION_RATE # 80
        W_IN = 64
        W_OUT_PER_CHUNK = W_IN // COMPRESSION_RATE # 16
        TARGET_N_CHUNKS = 5 # 确保 T_mid = 80

        dtype = self.patch_embedding.weight.dtype
        B, _, T, H, W = x.shape
        device = x.device # 获取当前设备
        T_in = image_cond.shape[2] # 原始输入的时间维度长度

        # 1. 提取局部记忆 (Last Frame Memory) - 必须在压缩前进行
        loc_mem = image_cond[:,:,-1:,:,:].to(dtype) 

        # 2. 确保输入数据类型正确
        image_cond = image_cond.to(dtype)
        
        # ----------------- [NEW LOGIC START] 时序压缩逻辑 -----------------

        # 🔴 只在 use_convenc=True 时执行时序压缩
        if T_in <= TARGET_T_MID: 
            # 情况 A: T_in <= 80，直接一次编码
            image_cond = self.latent_encoder(image_cond)
            
        else:
            # 情况 B: T_in > 80，滑动窗口 + 二次压缩
            
            # --- Step 1: 滑动窗口分块编码 (T_in -> T_mid=80) ---
            
            # 计算步长 S，确保 5 个 Chunk 覆盖 T_in
            S_denom = TARGET_N_CHUNKS - 1
            # S = floor( (T_in - W_IN) / (N_chunks - 1) )
            S = math.floor((T_in - W_IN) / S_denom)
            S = max(1, S) # 最小步长为 1

            latent_chunks = []
            
            for i in range(TARGET_N_CHUNKS):
                start = i * S
                end = start + W_IN
                
                chunk = image_cond[:, :, start:end, :, :]
                
                # 处理填充：如果 end > T_in，则需要填充
                padding_len = W_IN - chunk.shape[2]
                if padding_len > 0:
                    # 在时序维度 (dim=2) 末尾填充 0
                    # F.pad 参数: (W_pad_start, W_pad_end, H_pad_start, H_pad_end, T_pad_start, T_pad_end)
                    chunk = F.pad(chunk, (0, 0, 0, 0, 0, padding_len))
                
                # 编码块 (W_IN -> W_OUT_PER_CHUNK=16)
                # 第一次编码通常冻结
                # with torch.no_grad():
                    # self.latent_encoder.eval() 
                encoded_chunk = self.latent_encoder(chunk)
                    # self.latent_encoder.train() 
                
                # 裁剪到预期的输出长度 (防止 padding 导致的额外输出)
                encoded_chunk = encoded_chunk[:, :, :W_OUT_PER_CHUNK, :, :]
                latent_chunks.append(encoded_chunk)

            # 拼接中间序列 T_mid (T_mid = 80)
            image_cond = torch.cat(latent_chunks, dim=2)
            T_mid = image_cond.shape[2] 
            
            # --- Step 2: 二次压缩 (T_mid=80 -> T_out=20) ---
            
            if T_mid > MAX_T_OUT:
                # 此时 T_mid = 80，是 4 的倍数，直接编码即可
                image_cond = self.latent_encoder(image_cond)
                # T_out = 20

        # ----------------- [NEW LOGIC END] -----------------

        # 3. 拼接压缩后的 Condition 和 Loc_Mem
        image_cond = torch.cat((image_cond, loc_mem), dim=2) 
        
        # 4. 拼接 Condition 和 Noisy Input
        x = torch.cat((image_cond, x.to(dtype)), dim=2) # B, C, T_all, H, W
        # print("x init shape: ", x.shape)
        # print("image_cond init shape: ", image_cond.shape)

        T_all = x.shape[2]
        mask = torch.ones(B, T_all, H, W, device=x.device, dtype=x.dtype) # B, T_all, H, W
        mask[:, -T:] = 0
        mask = mask.unsqueeze(1).expand(-1, 4, -1, -1, -1) # B, 4, T_all, H, W

        x = torch.cat((x, mask), dim=1) # B, C+4, T_all, H, W
        T_x = T
        T = T_all
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]
        T_cond = image_cond.shape[2] # 新的 T_cond 约为 21 (20 + 1 loc_mem)
        num_c = (T_cond // self.patch_size[0]) * (H // self.patch_size[1]) * (W // self.patch_size[2])
        for block in self.blocks:
            block.self_attn.num_c = num_c
        dtype = self.patch_embedding.weight.dtype
        x = x.to(dtype)
        t = t.to(dtype)
        y = y.to(dtype)

        if self.model_type == 'i2v':
            assert clip_fea is not None and image_cond is not None
            # clip_fea = clip_fea.to(dtype)
            

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.model_type == 'i2v' and image_cond is not None:
            # image_cond = image_cond.to(dtype)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, image_cond)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] # fp32 -> bf16
        
        # *******************************************************************
        # 注意：这里的 action_encoder 调用已经更新为 move 和 view
        # 假设 self.action_encoder 现在接收 move 和 view 两个参数
        # *******************************************************************
        
        # Action Embedding Logic
        action_embedding_2 = self.action_encoder(move[:, -81:], view[:, -81:]).to(dtype).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

    
        # padding action embedding2 with a tensor of all zeros, the tensor has a same time length of image cond
        action_shape = list(action_embedding_2.shape)
        action_shape[2] = T_cond
        padding_embedding = torch.zeros(action_shape, device=device)

        # make data type and device right with action embedding 1
        padding_embedding = padding_embedding.to(dtype).to(device)

        # concat action embedding 1 and 2
        action_embedding = torch.cat((padding_embedding, action_embedding_2), dim=2)

        # 切片 action embedding to meet the length of x (the last action)
        action_embedding = action_embedding[:, :, -T_all:]
        # print("action", action_embedding.shape)
        # print("u shape 1", x[0].shape)
        x = [u + action_embedding for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        # print("u shape", x[0].shape)
        # hack seq_len
        seq_len = seq_lens.max()
        x = torch.cat([
            torch.cat([u, u.new_zeros(u.size(0), seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # print("x now", x.shape)
        # time embeddings
        with autocast_context(x, dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        t_no_noise = torch.zeros_like(t)  # 对应 t = 0

        with autocast_context(x, dtype=torch.float32):
            e_no_noise = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_no_noise).float())
            e0_no_noise = self.time_projection(e_no_noise).unflatten(1, (6, self.dim))
            assert e_no_noise.dtype == torch.float32 and e0_no_noise.dtype == torch.float32
        y = y[:,0]
        y = y * y_mask[...,None]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.model_max_length - u.size(0), u.size(1))])  for u in y] #padding
            )
        )

        # # sync context among cp ranks to avoid the following situation:
        # # cp_rank 0 dropped the context but cp_rank 1 did not, then they have different y embeeding in a forward pass
        # if context_parallel_util.get_cp_size() > 1:
        #     context_parallel_util.cp_broadcast(context)

        if self.model_type == 'i2v' and clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1) # bf16 --> tf32

        if self.enable_context_parallel:
            x = rearrange(x, "B (T S) C -> B T S C", T=N_t)
            x = context_parallel_util.split_cp(x, seq_dim=2)
            x = rearrange(x, "B T S C -> B (T S) C")
        
        # convert x_mask to token_ignore_mask
        token_ignore_mask = None
        if x_ignore_mask is not None:
            x_ignore_mask = x_ignore_mask.to(torch.float32) # [B, T, H, W]; cast for interpolation
            # x_ignore_mask_temp_sample_cond = temporal_sample(x_ignore_mask[:, :-T_x], rate=2, dim=1)
            # print(x_ignore_mask_temp_sample_cond.shape)
            x_ignore_mask_temp_sample = torch.cat((x_ignore_mask, x_ignore_mask[:, -T_x:]), dim=1)
            token_ignore_mask = nn.functional.interpolate(x_ignore_mask_temp_sample, size=(N_h, N_w), mode='nearest')[:, -T_all:] # [B, T, N_h, N_w]
            token_ignore_mask = token_ignore_mask.reshape(B, T * N_h * N_w) # [B, N]
            token_ignore_mask = (token_ignore_mask > 0)
            
        if self.enable_context_parallel and x_ignore_mask is not None:
            token_ignore_mask = rearrange(token_ignore_mask, "B (T S) -> B T S", T=T)
            token_ignore_mask = context_parallel_util.split_cp(token_ignore_mask, seq_dim=2)
            token_ignore_mask = rearrange(token_ignore_mask, "B T S -> B (T S)")

        for block in self.blocks:
            # support grad checkpointing
            x = auto_grad_checkpoint(block, x, [e0, e0_no_noise], seq_lens, grid_sizes, self.freqs, context, context_lens, token_ignore_mask)
        
        if self.enable_context_parallel:
            x = context_parallel_util.gather_cp(x, N_t)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return torch.stack(x).float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
