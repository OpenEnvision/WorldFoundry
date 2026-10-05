"""DreamX-World Wan transformer (official packing + ProPE option).

Wan 2.x Fun-style stem with optional :class:`PropeSelfAttention`.
Causal rollout is :mod:`.causal`.  Adds optional pose RoPE; no VACE.
"""

import glob
import json
import math
import os
import types
import warnings
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.distributed as dist

from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from worldfoundry.core.nn import ModuleDeviceDtypeMixin

from worldfoundry.base_models.diffusion_model.models.networks.wan.mixins import (
    WanTransformerMethodsMixin,
)
from worldfoundry.core.distributed.sequence_parallel.runtime import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.camera_attention import attention
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm.prope import prope_qkv
from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_22 import (
    WanLayerNorm,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_qk(q, k, grid_sizes, freqs):
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        x = attention(
            q.to(dtype),
            k.to(dtype),
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class PropeSelfAttention(nn.Module):
    """
    Lightweight parallel PRoPE self-attention module.

    """

    def __init__(self,
                 dim,
                 attn_dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        # independent q/k/v projections (dim -> attn_dim)
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm_q = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()

        # Preserve the checkpoint architecture's zero-initialized residual projection.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, cam_emb, seq_lens=None, sp_group=None):
        """
        Args:
            x (Tensor): Shape [B, S, D] — input tokens (after norm + modulation).
                        S may be padded to max_seq_len; actual lengths are in seq_lens.
            cam_emb (dict): Camera embedding dict with keys 'viewmats' and 'K'
            seq_lens (Tensor, optional): Shape [B], actual token count per sample
        Returns:
            Tensor: Shape [B, S, D] — prope attention output (zero-padded to match input)
        """
        batch_size, seq_len, dim = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim


        # q/k/v projections: [B, L, N, D]
        q = self.norm_q(self.q_proj(x)).view(batch_size, seq_len, num_heads, head_dim)
        k = self.norm_k(self.k_proj(x)).view(batch_size, seq_len, num_heads, head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, num_heads, head_dim)

        # transpose to [B, N, L, D] for prope_qkv
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply PRoPE positional encoding
        q, k, v, apply_fn_o = prope_qkv(
            q, k, v,
            viewmats=cam_emb['viewmats'],
            Ks=cam_emb['K'],
        )

        # attention: expects [B, L, N, D]
        out = attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=seq_lens,
        )

        # apply inverse PRoPE transform
        out = apply_fn_o(out.transpose(1, 2)).transpose(1, 2)

        # project back to dim: [B, L_valid, D]
        out = out.flatten(2)
        out = self.out_proj(out)


        return out

class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # compute attention
        x = attention(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            k_lens=context_lens
        )
        x = x.to(dtype)

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

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
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
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype),
            k_img.to(dtype),
            v_img.to(dtype),
            k_lens=None
        )
        img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # compute attention
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        # output
        x = x.flatten(2)
        x = self.o(x.to(dtype))
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.add_control_adapter = kwargs.get('add_control_adapter', False)
        self.cam_method = kwargs.get('cam_method', None)
        self.attn_compress = kwargs.get('attn_compress', 1)
        self.layer_idx = kwargs.get('layer_idx', None)
        cam_self_attn_layers = kwargs.get('cam_self_attn_layers', None)

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
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

        # parallel PRoPE self-attention branch
        add_cam_attn = self.add_control_adapter and self.cam_method == 'prope'
        if add_cam_attn and cam_self_attn_layers is not None:
            add_cam_attn = self.layer_idx in cam_self_attn_layers
        if add_cam_attn:
            self.cam_self_attn = PropeSelfAttention(
                dim,
                dim // self.attn_compress,
                num_heads // self.attn_compress,
                qk_norm=qk_norm,
                eps=eps,
            )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
        group=None,
        cam_emb=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        # self-attention (RoPE branch)
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype)

        # parallel PRoPE branch (v2 design)
        if hasattr(self, 'cam_self_attn') and cam_emb is not None:
            # print(f'temp_x: {temp_x.shape}')
            # print(f'cam_emb["viewmats"]: {cam_emb["viewmats"].shape}')
            # print(f'cam_emb["K"]: {cam_emb["K"].shape}')
            y = y + self.cam_self_attn(temp_x, cam_emb, seq_lens=seq_lens)

        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
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
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
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



class WanTransformer3DModel(
    WanTransformerMethodsMixin,
    ModuleDeviceDtypeMixin,
    nn.Module,
    NativeConfigMixin,
):
    r"""DreamX-World Wan transformer (optional ProPE self-attention)."""

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
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
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        cross_attn_type=None,
        add_control_adapter=False,
        cam_method='prope',
        attn_compress=1,
        cam_self_attn_layers=None,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
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

        # assert model_type in ['t2v', 'i2v', 'ti2v']
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
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if cross_attn_type is None:
            if model_type == "t2v":
                cross_attn_type = 't2v_cross_attn'
            else:
                cross_attn_type = 'i2v_cross_attn'
            # else:
                # cross_attn_type = "cross_attn"

        attn_class = WanAttentionBlock # by default use WanAttentionBlock

        self.control_adapter = None
        self.camera_projection = None

        self.blocks = nn.ModuleList([
            attn_class(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              add_control_adapter=add_control_adapter,
                              cam_method=cam_method,
                              attn_compress=attn_compress,
                              layer_idx=layer_idx,
                              cam_self_attn_layers=cam_self_attn_layers)
            for layer_idx in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.cam_method = cam_method


        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.sp_group = None

        self.customize_initialize_weights()

    def customize_initialize_weights(self):
        self.init_weights()

        for blk in self.blocks:
            if hasattr(blk, 'cam_self_attn'):
                nn.init.zeros_(blk.cam_self_attn.out_proj.weight)
                nn.init.zeros_(blk.cam_self_attn.out_proj.bias)


    def enable_multi_gpus_inference(self,):
        from worldfoundry.base_models.diffusion_model.models.networks.wan.parallel_adapter import sp_prope_forward, usp_attn_forward

        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        self.sp_group = None # None for inference

        # For RoPE self-attention.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

            # For PRoPE parallel branch (cam_self_attn).
            if hasattr(block, 'cam_self_attn'):
                block.cam_self_attn.forward = types.MethodType(
                    sp_prope_forward, block.cam_self_attn)

    # @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        dtype=torch.bfloat16,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        device = self.patch_embedding.weight.device
        dtype = x[0].dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # print(f"seq_len: {seq_len} t.shape: {t.shape}")
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:

            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]


        if y_camera is not None and isinstance(y_camera, dict):
            cam_viewmats = y_camera['viewmats']
            cam_K = y_camera['K']
        else:
            cam_viewmats = None
            cam_K = None

        cam_emb = None
        if cam_viewmats is not None:
            cam_emb = {'viewmats': cam_viewmats, 'K': cam_K}
        block_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t,
            group=self.sp_group,
            cam_emb=cam_emb,
        )
        for block in self.blocks:
            x = block(x, **block_kwargs)

        # head
        x = self.head(x, e)



        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        return x

class Wan2_2Transformer3DModel(WanTransformer3DModel):
    r"""DreamX-World Wan 2.2 transformer subclass."""

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    def __init__(
        self,
        model_type='t2v',
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
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        cam_method='prope',
        attn_compress=1,
        cam_self_attn_layers=None,
    ):
        r"""
        Initialize the diffusion model backbone.
        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
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
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            cross_attn_type="cross_attn",
            cam_method=cam_method,
            attn_compress=attn_compress,
            cam_self_attn_layers=cam_self_attn_layers,
        )

        if hasattr(self, "img_emb"):
            del self.img_emb
