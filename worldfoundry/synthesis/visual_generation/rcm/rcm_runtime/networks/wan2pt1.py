# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from typing import Optional, List

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange, repeat

from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, CheckpointWrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

from worldfoundry.core.distributed.runtime.logging import log
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.a2a_cp import MinimalA2AAttnOp
from worldfoundry.core.nn.checkpointing.activation_checkpointing import CheckpointMode, SACConfig
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.context_parallel import split_inputs_cp, cat_outputs_cp, cat_outputs_cp_with_grad, broadcast
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.kv_cache import KVCache, AttnContext, CausalInferenceState, KVCacheMode
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.blockmask import AttnMaskSpec, FlexOrSdpaLocalAttention
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.rope import RopeCache

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def _uses_append_cache(attn_ctx: Optional[AttnContext]) -> bool:
    return attn_ctx is not None and attn_ctx.mode == KVCacheMode.APPEND and attn_ctx.kv_cache is not None


class CacheAwareCheckpointWrapper(CheckpointWrapper):
    def __init__(self, module: nn.Module, context_fn, preserve_rng_state: bool):
        super().__init__(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=context_fn,
            preserve_rng_state=preserve_rng_state,
        )

    def forward(self, *args, **kwargs):
        if _uses_append_cache(kwargs.get("attn_ctx")):
            return self._checkpoint_wrapped_module(*args, **kwargs)
        return super().forward(*args, **kwargs)


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        self.seq = None
        self.dim_spatial_range = None
        self.dim_temporal_range = None

    def cache_parameters(self, device: torch.device, max_t_needed: int) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        required_len = max(self.max_h, self.max_w, max(self.max_t, max_t_needed))
        needs_refresh = (
            self.seq is None
            or self.seq.device != device
            or self.seq.numel() < required_len
            or self.dim_spatial_range is None
            or self.dim_temporal_range is None
        )
        if not needs_refresh:
            return

        self.seq = torch.arange(required_len, device=device, dtype=torch.float32)
        self.dim_spatial_range = torch.arange(0, dim_h, 2, device=device, dtype=torch.float32)[: (dim_h // 2)] / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2, device=device, dtype=torch.float32)[: (dim_t // 2)] / dim_t

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        t_start: int = 0,
        t_indices: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        total_tokens = T * H * W

        _t_indices_provided = t_indices is not None
        if t_indices is None:
            t_indices = torch.arange(t_start, t_start + T, device=self.patch_device(), dtype=torch.long)
        else:
            t_indices = t_indices.to(device=self.patch_device(), dtype=torch.long)

        if t_indices.ndim != 1:
            raise ValueError(f"t_indices must be 1D, got {tuple(t_indices.shape)}")

        if t_indices.numel() == T:
            t_token = torch.repeat_interleave(t_indices, repeats=H * W)
        elif t_indices.numel() == total_tokens:
            t_token = t_indices
        else:
            raise ValueError(f"t_indices must have {T} or {total_tokens} elements, got {t_indices.numel()}")

        if _t_indices_provided:
            max_t_needed = int(t_token.max().item()) + 1 if t_token.numel() > 0 else t_start + T
        else:
            max_t_needed = t_start + T
        self.cache_parameters(device=t_token.device, max_t_needed=max_t_needed)

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        frame_tokens = H * W
        repeats = total_tokens // frame_tokens
        h_idx = torch.arange(H, device=t_token.device).repeat_interleave(W).repeat(repeats)
        w_idx = torch.arange(W, device=t_token.device).repeat(H).repeat(repeats)

        freqs_t = t_token.to(torch.float32).unsqueeze(-1) * temporal_freqs.unsqueeze(0)
        freqs_h = h_idx.to(torch.float32).unsqueeze(-1) * h_spatial_freqs.unsqueeze(0)
        freqs_w = w_idx.to(torch.float32).unsqueeze(-1) * w_spatial_freqs.unsqueeze(0)
        return torch.cat([freqs_t, freqs_h, freqs_w], dim=-1).float()

    @property
    def seq_dim(self):
        return 0

    def patch_device(self) -> torch.device:
        if self.seq is not None:
            return self.seq.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.qk_norm = qk_norm

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op = MinimalA2AAttnOp(local_attn=FlexOrSdpaLocalAttention())

    def init_weights(self):
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.q.weight, std=std)
        torch.nn.init.trunc_normal_(self.k.weight, std=std)
        torch.nn.init.trunc_normal_(self.v.weight, std=std)
        torch.nn.init.trunc_normal_(self.o.weight, std=std)
        # zero out bias
        self.q.bias.data.zero_()
        self.k.bias.data.zero_()
        self.v.bias.data.zero_()
        self.o.bias.data.zero_()
        # reset norm weights
        if self.qk_norm:
            self.norm_q.reset_parameters()
            self.norm_k.reset_parameters()

    def forward(self, x, seq_lens, attn_ctx: Optional[AttnContext] = None, attn_meta: Optional[AttnMaskSpec] = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            attn_ctx: Per-layer attention context (KV cache, RoPE freqs, observer).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = self.attn_op(q, k, v, attn_ctx=attn_ctx, attn_meta=attn_meta)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


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
        x = self.attn_op(q, k, v)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op_image = MinimalA2AAttnOp()

    def init_weights(self):
        super().init_weights()
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.k_img.weight, std=std)
        torch.nn.init.trunc_normal_(self.v_img.weight, std=std)
        # zero out bias
        self.k_img.bias.data.zero_()
        self.v_img.bias.data.zero_()
        # reset norm weights
        if self.qk_norm:
            self.norm_k_img.reset_parameters()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = self.attn_op_image(q, k_img, v_img)
        # compute attention
        x = self.attn_op(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {"t2v_cross_attn": WanT2VCrossAttention, "i2v_cross_attn": WanI2VCrossAttention}


class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        self.self_attn.init_weights()
        self.cross_attn.init_weights()

        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)

    def forward(
        self,
        x,
        e,
        seq_lens,
        context,
        context_lens,
        attn_ctx: Optional[AttnContext] = None,
        attn_meta: Optional[AttnMaskSpec] = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            attn_ctx: Per-layer attention context (KV cache, RoPE freqs, observer).
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).unbind(dim=2)

        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x),
            seq_lens,
            attn_ctx=attn_ctx,
            attn_meta=attn_meta,
        )
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].type_as(x)

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).type_as(x))
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5].type_as(x)
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

    def init_weights(self):
        self.norm.reset_parameters()

        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)
        torch.nn.init.trunc_normal_(self.head.weight, std=std)
        self.head.bias.data.zero_()

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).unbind(dim=2)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def init_weights(self):
        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()
        self.proj[3].reset_parameters()
        self.proj[4].reset_parameters()

        if hasattr(self, "emb_pos"):
            self.emb_pos.data.zero_()

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        sac_config: SACConfig = SACConfig(),
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v"]
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
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_crossattn_projection = False

        # embeddings
        self.patch_embedding = nn.Linear(in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0

        d = dim // num_heads

        self.rope_position_embedding = VideoRopePosition3DEmb(head_dim=d, len_h=128, len_w=128, len_t=32)
        self._rope_cache: dict = {}

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

        self.enable_selective_checkpoint(sac_config)

    def get_spatial_patch_size(self):
        return self.patch_size[1] * self.patch_size[2]

    def _compute_rope(
        self, B: int, T: int, H: int, W: int, inference_state: Optional[CausalInferenceState], attn_meta: Optional[AttnMaskSpec], use_fused: bool
    ) -> RopeCache:
        """Compute RoPE frequencies and populate inference_state.

        **Extrapolation path** (``current_t_indices`` set): pre-RoPE caching,
        explicit per-token temporal indices.

        **Original causal path** (``inference_state`` set, no ``current_t_indices``):
        post-RoPE caching, sequential positions.

        **Bidirectional / teacher-forcing** (``inference_state is None``):
        no caching, sequential positions.
        """
        gen = self.rope_position_embedding.generate_embeddings

        if inference_state is not None and inference_state.current_t_indices is not None:
            query_freqs = gen(torch.Size([B, T, H, W, self.dim]), t_indices=inference_state.current_t_indices)
            if inference_state.cached_t_indices is not None:
                key_t = torch.cat([inference_state.cached_t_indices, inference_state.current_t_indices], dim=0)
            else:
                key_t = inference_state.current_t_indices
            key_freqs = gen(torch.Size([B, key_t.numel() // (H * W), H, W, self.dim]), t_indices=key_t)
            current_key_freqs = gen(torch.Size([B, T, H, W, self.dim]), t_indices=inference_state.current_t_indices)

            shared_rope = RopeCache(
                query_freqs=query_freqs, key_freqs=key_freqs, current_key_freqs=current_key_freqs, use_fused=use_fused, cached_k_rotated=False
            )
            inference_state.rope = shared_rope
            return shared_rope

        t_offset = 0 if inference_state is None else int(inference_state.t_offset)
        attn_mode = attn_meta.mode if attn_meta is not None else None
        fast_infer = inference_state is not None and inference_state.fast_infer
        has_kv_cache = inference_state is not None and inference_state.mode != KVCacheMode.DISABLED

        cache_key = (T, H, W, t_offset, attn_mode, fast_infer, has_kv_cache, use_fused)
        cached_rope = self._rope_cache.get(cache_key)
        if cached_rope is not None:
            if inference_state is not None:
                inference_state.rope = cached_rope
            return cached_rope

        if attn_mode == "teacher_forcing":
            query_freqs = gen(torch.Size([B, T // 2, H, W, self.dim]), t_start=t_offset).repeat(2, 1)
            key_freqs = query_freqs
            current_key_freqs = query_freqs
        else:
            query_freqs = gen(torch.Size([B, T, H, W, self.dim]), t_start=t_offset)
            if has_kv_cache:
                if fast_infer:
                    current_key_freqs = query_freqs
                    key_freqs = None
                else:
                    key_freqs = gen(torch.Size([B, t_offset + T, H, W, self.dim]), t_start=0)
                    current_key_freqs = gen(torch.Size([B, T, H, W, self.dim]), t_start=t_offset)
            else:
                key_freqs = query_freqs
                current_key_freqs = query_freqs

        rope = RopeCache(
            query_freqs=query_freqs,
            key_freqs=key_freqs,
            current_key_freqs=current_key_freqs,
            use_fused=use_fused,
            cached_k_rotated=True,
        )
        self._rope_cache[cache_key] = rope
        if inference_state is not None:
            inference_state.rope = rope
        return rope

    def forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        frame_cond_crossattn_emb_B_L_D=None,
        y_B_C_T_H_W=None,
        inference_state: Optional[CausalInferenceState] = None,
        attn_meta: Optional[AttnMaskSpec] = None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x_B_C_T_H_W (Tensor):
                Input video tensor with shape [B, C_in, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            frame_cond_crossattn_emb_B_L_D (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y_B_C_T_H_W (Tensor, *optional*):
                Conditional video inputs for image-to-video mode, shape [B, C_in, T, H, W]

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, T, H / 8, W / 8]
        """

        cp_group = getattr(self, "_cp_group", None)
        cp_enabled = (cp_group is not None) and (cp_group.size() > 1)
        if cp_enabled:
            x_B_C_T_H_W = broadcast(x_B_C_T_H_W, cp_group)
            timesteps_B_T = broadcast(timesteps_B_T, cp_group)
            crossattn_emb = broadcast(crossattn_emb, cp_group)
            if frame_cond_crossattn_emb_B_L_D is not None:
                frame_cond_crossattn_emb_B_L_D = broadcast(frame_cond_crossattn_emb_B_L_D, cp_group)
            if y_B_C_T_H_W is not None:
                y_B_C_T_H_W = broadcast(y_B_C_T_H_W, cp_group)

        assert timesteps_B_T.ndim == 2, f"timesteps_B_T must be 2D [B, T], got {timesteps_B_T.shape}"
        del kwargs
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert frame_cond_crossattn_emb_B_L_D is not None and y_B_C_T_H_W is not None

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        kt, kh, kw = self.patch_size
        assert kt == 1, "Currently only support temporal patch size of 1."
        B, _, T_in, H_in, W_in = x_B_C_T_H_W.shape
        assert (T_in % kt) == 0 and (H_in % kh) == 0 and (W_in % kw) == 0
        T, H, W = T_in // kt, H_in // kh, W_in // kw
        L = T * H * W

        uniform_time = timesteps_B_T.shape[1] == 1
        if not uniform_time:
            assert timesteps_B_T.shape[1] == T, f"timesteps_B_T.shape[1]={timesteps_B_T.shape[1]} != T={T}"

        # patchify and flatten
        x_B_L_Din = rearrange(
            x_B_C_T_H_W,
            "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
            kt=kt,
            kh=kh,
            kw=kw,
        ).contiguous()

        if cp_enabled:
            assert (L % cp_group.size()) == 0, f"L=T*H*W must be divisible by cp_size. Got L={L}, cp={cp_group.size()}."
            x_B_L_Din = split_inputs_cp(x_B_L_Din, seq_dim=1, cp_group=cp_group)

        # embeddings
        x_B_L_D = self.patch_embedding(x_B_L_Din)
        seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)

        # time embeddings
        if uniform_time:
            t_flat = timesteps_B_T.reshape(-1)
            with amp.autocast("cuda", dtype=torch.float32):
                e_BT_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_flat).float())
                e0_B_L_6_D = self.time_projection(e_BT_D).view(B, 1, 6, self.dim)
            e_B_L_D = e_BT_D.view(B, 1, self.dim)
        else:
            with amp.autocast("cuda", dtype=torch.float32):
                e_BT_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timesteps_B_T.reshape(-1)).float())
                e0_B_T_6_D = self.time_projection(e_BT_D).view(B, T, 6, self.dim).contiguous()
            e_B_T_D = e_BT_D.view(B, T, self.dim).contiguous()
            e_B_L_D = repeat(e_B_T_D, "b t d -> b (t hw) d", hw=H * W).contiguous()
            e0_B_L_6_D = repeat(e0_B_T_6_D, "b t m d -> b (t hw) m d", hw=H * W).contiguous()
            assert e_B_L_D.dtype == torch.float32 and e0_B_L_6_D.dtype == torch.float32

        if cp_enabled and not uniform_time:
            e_B_L_D = split_inputs_cp(e_B_L_D, seq_dim=1, cp_group=cp_group)
            e0_B_L_6_D = split_inputs_cp(e0_B_L_6_D, seq_dim=1, cp_group=cp_group)

        # context
        context_lens = None
        context_B_L_D = self.text_embedding(crossattn_emb)

        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)  # bs x 257 (x2) x dim
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        shared_rope = self._compute_rope(B, T, H, W, inference_state, attn_meta, use_fused=True)

        # arguments
        kwargs = dict(
            e=e0_B_L_6_D,
            seq_lens=seq_lens,
            context=context_B_L_D,
            context_lens=context_lens,
        )

        for block_idx, block in enumerate(self.blocks):
            if inference_state is not None:
                attn_ctx = inference_state.attn_ctx(block_idx)
            else:
                attn_ctx = AttnContext(rope=shared_rope)
            x_B_L_D = block(x_B_L_D, **kwargs, attn_ctx=attn_ctx, attn_meta=attn_meta)

        # head
        x_B_L_Dout = self.head(x_B_L_D, e_B_L_D)

        if cp_enabled:
            if torch.is_grad_enabled():
                x_B_L_Dout = cat_outputs_cp_with_grad(x_B_L_Dout, seq_dim=1, cp_group=cp_group)
            else:
                x_B_L_Dout = cat_outputs_cp(x_B_L_Dout, seq_dim=1, cp_group=cp_group)

        # unpatchify
        x_B_C_T_H_W = rearrange(
            x_B_L_Dout,
            "b (t h w) (kt kh kw d) -> b d (t kt) (h kh) (w kw)",
            kt=kt,
            kh=kh,
            kw=kw,
            t=T,
            h=H,
            w=W,
            d=self.out_dim,
        )
        return x_B_C_T_H_W

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

        for block in self.blocks:
            block.init_weights()
        self.head.init_weights()

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    def fully_shard(self, mesh, mp_policy):
        for i, block in enumerate(self.blocks):
            fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self.head, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
        fully_shard(self.text_embedding, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self.time_embedding, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self.patch_embedding, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)

    def disable_context_parallel(self):
        # attention
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False
        self._cp_group = None

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(process_group=process_group, ranks=cp_ranks, stream=torch.cuda.Stream())

        self._is_context_parallel_enabled = True
        self._cp_group = process_group

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            return self

        log.info(f"Enable selective checkpoint with mm_only, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}")
        _context_fn = sac_config.get_context_fn()
        for block_id, block in self.blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                block = CacheAwareCheckpointWrapper(block, context_fn=_context_fn, preserve_rng_state=False)
                self.blocks.register_module(block_id, block)
        self.register_module("head", ptd_checkpoint_wrapper(self.head, context_fn=_context_fn, preserve_rng_state=False))

        return self

    def allocate_kv_caches(self, max_len: int, cache_factory=None) -> List[KVCache]:
        """Build one KV cache per layer.

        ``cache_factory`` lets WorldFoundry substitute a policy-driven cache for
        bounded-memory rollouts; it receives ``max_len`` and returns a cache with
        the same surface. Omitting it keeps the upstream unbounded behaviour.
        """
        if cache_factory is not None:
            return [cache_factory(max_len) for _ in range(self.num_layers)]
        return [KVCache(max_len) for _ in range(self.num_layers)]
