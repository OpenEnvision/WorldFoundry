# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# X-WAM joint world-action extensions copyright the X-WAM contributors.
# WorldFoundry modifications: inference-only path, core attention, and explicit mixed-dtype boundaries.
import math
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_21 import (
    WanRMSNorm,
    sinusoidal_embedding_1d,
)

from .attention import attention

__all__ = ["XWAMModel"]


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000, scale=1.0):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(0, max_seq_len, 1 / scale),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_1d(x, freqs):
    """
    x:     [B, L, H, D]  (real, float)
    freqs: [L, D//2]  (complex, from rope_params)
    """
    B, L, H, D = x.shape

    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(B, L, H, D // 2, 2))

    return torch.view_as_real(x_complex * freqs).flatten(3).float()


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def prepare_qkv(self, x, freqs):
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        x = x.type_as(self.q.weight)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(x)).view(b, -1, n, d)
        v = self.v(x).view(b, -1, n, d)

        rope_q = rope_apply_1d(q, freqs)
        rope_k = rope_apply_1d(k, freqs)

        return rope_q, rope_k, v

    def forward(self, x, freqs, save_kv_cache=False, cache_k=None, cache_v=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, D] where L = T*V*HW + Ta + Tp
            freqs(Tensor): RoPE frequencies, shape [L, D / num_heads / 2]
            save_kv_cache(bool): Whether to return K/V for extra modality branches
            cache_k(Tensor): Optional cached keys [B, L', num_heads, head_dim]
            cache_v(Tensor): Optional cached values [B, L', num_heads, head_dim]
        """
        rope_q, rope_k, v = self.prepare_qkv(x, freqs)

        if cache_k is not None and cache_v is not None:
            use_k = torch.cat([rope_k, cache_k], dim=1)
            use_v = torch.cat([v, cache_v], dim=1)
        elif cache_k is not None or cache_v is not None:
            raise ValueError("cache_k and cache_v must all be None or not None")
        else:
            use_k = rope_k
            use_v = v

        x = attention(q=rope_q, k=use_k, v=use_v).flatten(2)
        x = self.o(x.type_as(self.o.weight))

        if save_kv_cache:
            return x, rope_k, v

        return x


class WanCrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, qk_norm=True, eps=1e-6):
        assert kv_dim % num_heads == 0
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = kv_dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(q_dim, kv_dim)
        self.k = nn.Linear(kv_dim, kv_dim)
        self.v = nn.Linear(kv_dim, kv_dim)
        self.o = nn.Linear(kv_dim, q_dim)
        self.norm_q = WanRMSNorm(kv_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(kv_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        x = x.type_as(self.q.weight)

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
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
        self.cross_attn = WanCrossAttention(dim, dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, freqs, context, save_kv_cache=False, cache_k=None, cache_v=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, D] where L = T*V*HW (+ Ta + Tp for main branch)
            e(Tensor): Per-token modulation [B, L, 6, D] from timestep embedding
            freqs(Tensor): RoPE frequencies [L, D / num_heads / 2]
            context(Tensor): Text embeddings [B, text_len, D]
            save_kv_cache(bool): Return K/V for use by extra modality branches
            cache_k(Tensor): Cached keys from main branch [B, L_main, num_heads, head_dim]
            cache_v(Tensor): Cached values from main branch [B, L_main, num_heads, head_dim]
        """
        T = e.shape[1]
        assert e.dtype == torch.float32
        e = (self.modulation.float().unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        x_norm = (self.norm1(x.unflatten(1, (T, -1))).float() * (1 + e[1]) + e[0]).flatten(1, 2)
        if save_kv_cache:
            y, rope_k, v = self.self_attn(x_norm, freqs, True, cache_k, cache_v)
        else:
            y = self.self_attn(x_norm, freqs, False, cache_k, cache_v)
        x = x.float() + (y.unflatten(1, (T, -1)).float() * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(
            (self.norm2(x.unflatten(1, (T, -1))).float() * (1 + e[4]) + e[3]).type_as(self.ffn[0].weight)
        ).flatten(1, 2)
        x = x.float() + (y.unflatten(1, (T, -1)).float() * e[5]).flatten(1, 2)

        if save_kv_cache:
            return x, rope_k, v

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
            e(Tensor): Shape [B, L1, C]
        """
        T = e.shape[1]
        assert e.dtype == torch.float32
        e = (self.modulation.float().unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.norm(x.unflatten(1, (T, -1))).float() * (1 + e[1]) + e[0]
        x = self.head(x.type_as(self.head.weight)).flatten(1, 2)
        return x


class XWAMModel(ModelMixin, ConfigMixin):
    r"""
    X-WAM: Joint video-action-proprioception diffusion transformer for multi-view robotic manipulation.
    Extends the Wan video backbone with action/proprio encoding, multi-view support, and
    optional extra modality branches (e.g. depth).
    """

    ignore_for_config: ClassVar[list] = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules: ClassVar[list] = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        num_modalities=2,
        num_views=3,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        action_dim=20,
        action_num=4,
        proprio_dim=20,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        num_extra_layers=10,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v', 'i2v', 'ti2v', or 's2v'
            num_modalities (`int`, *optional*, defaults to 2):
                Number of output modalities (e.g. 2 = RGB + depth). Extra modalities
                beyond the first get their own branched blocks and prediction heads.
            num_views (`int`, *optional*, defaults to 3):
                Number of camera views processed jointly via view embeddings and
                shared attention across views within each frame.
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input latent channels (VAE latent dim)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for per-token sinusoidal timestep embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text encoder output
            action_dim (`int`, *optional*, defaults to 20):
                Per-step action vector dimension
            action_num (`int`, *optional*, defaults to 4):
                Number of action steps between consecutive video frames
            proprio_dim (`int`, *optional*, defaults to 20):
                Per-step proprioception vector dimension
            out_dim (`int`, *optional*, defaults to 16):
                Output latent channels (should match in_dim for flow matching)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Total number of transformer blocks (shared backbone)
            num_extra_layers (`int`, *optional*, defaults to 10):
                Number of trailing blocks that are duplicated for extra modality branches.
                The last `num_extra_layers` blocks of the backbone also serve as the
                primary (RGB) branch; the extra branches attend to the primary branch's
                KV cache for cross-referencing.
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key RMSNorm
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable LayerNorm before cross-attention
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        self.num_modalities = num_modalities
        self.num_views = num_views
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.action_dim = action_dim
        self.action_num = action_num
        self.proprio_dim = proprio_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_extra_layers = num_extra_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.view_embedding = nn.Embedding(self.num_views, dim)

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)]
        )

        self.extra_blocks = nn.ModuleList()
        for _ in range(num_modalities - 1):
            self.extra_blocks.append(
                nn.ModuleList(
                    [
                        WanAttentionBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps)
                        for _ in range(num_extra_layers)
                    ]
                )
            )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        self.extra_heads = nn.ModuleList()
        for _ in range(num_modalities - 1):
            self.extra_heads.append(Head(dim, out_dim, patch_size, eps))

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, action_dim)
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.proprio_decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, proprio_dim)
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.video_freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        self.action_freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6), scale=action_num * 4),
                self.video_freqs[1][0:1].repeat(1024 * action_num * 4, 1),
                self.video_freqs[2][0:1].repeat(1024 * action_num * 4, 1),
            ],
            dim=-1,
        )
        self.proprio_freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6), scale=4),
                self.video_freqs[1][0:1].repeat(1024 * 4, 1),
                self.video_freqs[2][0:1].repeat(1024 * 4, 1),
            ],
            dim=-1,
        )

        # initialize weights
        self.init_weights()

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int = 0):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.video_freqs):
            self.video_freqs = [freq.to(device) for freq in self.video_freqs]
        if self.action_freqs.device != device:
            self.action_freqs = self.action_freqs.to(device)
        if self.proprio_freqs.device != device:
            self.proprio_freqs = self.proprio_freqs.to(device)
        f, h, w = grid_size
        v = self.num_views
        freqs = torch.cat(
            [
                self.video_freqs[0][start_frame : start_frame + f].view(f, 1, 1, 1, -1).expand(f, v, h, w, -1),
                self.video_freqs[1][:h].view(1, 1, h, 1, -1).expand(f, v, h, w, -1),
                self.video_freqs[2][:w].view(1, 1, 1, w, -1).expand(f, v, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * v * h * w, -1)

        return freqs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor | None = None,
        t_actions: torch.Tensor | None = None,
        proprios: torch.Tensor | None = None,
        t_proprios: torch.Tensor | None = None,
        cfg: float = 0.0,
        run_depth: bool = True,
    ):
        r"""
        Joint flow-matching velocity prediction for video, actions, and proprioception.

        Args:
            x (Tensor):
                Noisy video latents [B, C, T, V, H, W] (V = num_views)
            t (Tensor):
                Per-token video timesteps [B] (broadcast) or [B, T] (per-frame)
            context (Tensor):
                Text embeddings [B, L, text_dim]. When cfg > 0, must be
                [2B, L, text_dim] with cond in first B and uncond in last B.
            actions (Tensor):
                Noisy actions [B, Ta, action_dim]
            t_actions (Tensor):
                Action timesteps [B] or [B, Ta]
            proprios (Tensor):
                Noisy proprioceptions [B, Tp, proprio_dim]
            t_proprios (Tensor):
                Proprioception timesteps [B] or [B, Tp]
            cfg (float):
                Classifier-free guidance scale. 0 = no guidance.
            run_depth (bool):
                Whether to run extra modality branches (e.g. depth prediction).

        Returns:
            tuple of:
                - Predicted video velocity [B, C, T, V, H, W]
                - Predicted action velocity [B, Ta, action_dim]
                - Predicted proprio velocity [B, Tp, proprio_dim]
                - Extra modality outputs (list of Tensors or [None])
        """
        if cfg > 0.0:
            # context: [2B, L, D] — first B are cond, last B are uncond
            # x, t, actions, t_actions are [B, ...], need to be doubled
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            actions_in = torch.cat([actions, actions], dim=0)
            t_actions_in = torch.cat([t_actions, t_actions], dim=0)
            proprios_in = torch.cat([proprios, proprios], dim=0)
            t_proprios_in = torch.cat([t_proprios, t_proprios], dim=0)

            out_x, out_actions, out_proprios, out_extras = self._forward_single(
                x_in, t_in, context, actions_in, t_actions_in, proprios_in, t_proprios_in, run_depth
            )
            B = x.shape[0]
            cond_x, uncond_x = out_x[:B], out_x[B:]
            cond_actions, uncond_actions = out_actions[:B], out_actions[B:]
            cond_proprios, uncond_proprios = out_proprios[:B], out_proprios[B:]
            # Depth (extra_outs) is not CFG-guided — use conditional prediction only
            out_x = uncond_x + cfg * (cond_x - uncond_x)
            out_actions = uncond_actions + cfg * (cond_actions - uncond_actions)
            out_proprios = uncond_proprios + cfg * (cond_proprios - uncond_proprios)
            out_extras = [e[:B] for e in out_extras]
            return out_x, out_actions, out_proprios, out_extras
        else:
            return self._forward_single(x, t, context, actions, t_actions, proprios, t_proprios, run_depth)

    def _forward_single(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        t_actions: torch.Tensor,
        proprios: torch.Tensor,
        t_proprios: torch.Tensor,
        run_depth: bool = True,
    ):
        """Core forward pass without CFG doubling.

        Patchifies video, encodes actions/proprios, concatenates into a single
        sequence [video | actions | proprios], applies shared transformer blocks,
        then splits outputs and decodes each modality through its own head.
        Extra modality branches (depth) fork from the last `num_extra_layers` blocks
        and attend to the primary branch's KV cache.
        """
        # embeddings
        T = x.shape[2]
        Ta = actions.shape[1]
        Tp = proprios.shape[1]

        x = x.type_as(self.patch_embedding.weight)
        x = self.patch_embedding(rearrange(x, "b c t v h w -> (b v) c t h w"))  # B*V, D, T, H, W
        grid_sizes = list(x.shape[2:])  # [3]: THW
        video_freqs = self._create_freqs(grid_sizes)

        x = x.flatten(2).transpose(1, 2)
        x = rearrange(x, "(b v) l d-> b v l d", v=self.num_views)

        # view embeddings
        view_ids = torch.arange(self.num_views, device=x.device)
        view_embeddings = self.view_embedding(view_ids).view(1, self.num_views, 1, -1)  # [1, V, 1, D]
        x = x + view_embeddings
        x = rearrange(x, "b v (t hw) d-> b (t v hw) d", v=self.num_views, t=T)

        actions = self.action_encoder(actions.type_as(self.action_encoder[0].weight))
        proprios = self.proprio_encoder(proprios.type_as(self.proprio_encoder[0].weight))

        # concat as a single sequence: [B, T*V*H*W + Ta + Tp, D]
        input_seq: torch.Tensor = torch.cat([x, actions, proprios], dim=1)
        freqs = torch.cat([video_freqs, self.action_freqs[:Ta], self.proprio_freqs[:Tp]], dim=0)

        # context: [B, L, D]
        context = self.text_embedding(context.type_as(self.text_embedding[0].weight))

        # timestep embedding
        assert t.ndim == 1 or t.ndim == 2, f"t must be 1D or 2D, but got {t.ndim}"
        if t.ndim == 1:
            t = t.view(t.size(0), 1).repeat(1, T)
        t = t.repeat_interleave(x.shape[1] // T, dim=1)
        assert t.shape[1] == x.shape[1]  # T*V*H*W

        assert t_actions.ndim == 1 or t_actions.ndim == 2, f"t_actions must be 1D or 2D, but got {t_actions.ndim}"
        if t_actions.ndim == 1:
            t_actions = t_actions.view(t_actions.size(0), 1).repeat(1, Ta)

        assert t_proprios.ndim == 1 or t_proprios.ndim == 2, f"t_proprios must be 1D or 2D, but got {t_proprios.ndim}"
        if t_proprios.ndim == 1:
            t_proprios = t_proprios.view(t_proprios.size(0), 1).repeat(1, Tp)

        t_seq: torch.Tensor = torch.cat([t, t_actions, t_proprios], dim=1)
        time_features = sinusoidal_embedding_1d(self.freq_dim, t_seq.flatten()).unflatten(0, t_seq.shape)
        e = self.time_embedding(time_features.type_as(self.time_embedding[0].weight)).float()
        e0 = self.time_projection(e.type_as(self.time_projection[1].weight)).float()
        e0 = e0.unflatten(2, (6, self.dim))  # [B, T_all, 6, D]
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if run_depth:
            for bi in range(self.num_layers - self.num_extra_layers):
                input_seq = self.blocks[bi](input_seq, e=e0, freqs=freqs, context=context)

            # ========== extra modalities ==========
            extra_seq_intermediate = input_seq[:, : e.shape[1] - Ta - Tp]
            extra_e0 = e0[:, : e.shape[1] - Ta - Tp]
            extra_outs = [extra_seq_intermediate for _ in range(self.num_modalities - 1)]
            # ======================================

            for bi in range(self.num_extra_layers):
                block = self.blocks[bi + self.num_layers - self.num_extra_layers]
                input_seq, cache_k, cache_v = block(input_seq, e=e0, freqs=freqs, context=context, save_kv_cache=True)
                for hi in range(self.num_modalities - 1):
                    extra_outs[hi] = self.extra_blocks[hi][bi](
                        extra_outs[hi],
                        e=extra_e0,
                        freqs=video_freqs,
                        context=context,
                        cache_k=cache_k,
                        cache_v=cache_v,
                    )
        else:
            for bi in range(self.num_layers):
                input_seq = self.blocks[bi](input_seq, e=e0, freqs=freqs, context=context)

        x, actions, proprios = input_seq.split([input_seq.shape[1] - Ta - Tp, Ta, Tp], dim=1)

        # head
        x = self.head(x, e[:, : e.shape[1] - Ta - Tp])
        if run_depth:
            for hi in range(self.num_modalities - 1):
                extra_outs[hi] = self.extra_heads[hi](extra_outs[hi], e=e[:, : e.shape[1] - Ta - Tp])

        actions = self.action_decoder(actions.type_as(self.action_decoder[0].weight))
        proprios = self.proprio_decoder(proprios.type_as(self.proprio_decoder[0].weight))

        # unpatchify
        x = rearrange(x, "b (t v hw) c -> (b v) (t hw) c", t=T, v=self.num_views)
        x = self.unpatchify(x, grid_sizes)
        x = rearrange(x, "(b v) c t h w -> b c t v h w", v=self.num_views)

        if run_depth:
            for hi in range(self.num_modalities - 1):
                extra_outs[hi] = rearrange(extra_outs[hi], "b (t v hw) c -> (b v) (t hw) c", t=T, v=self.num_views)
                extra_outs[hi] = self.unpatchify(extra_outs[hi], grid_sizes)
                extra_outs[hi] = rearrange(extra_outs[hi], "(b v) c t h w -> b c t v h w", v=self.num_views).float()
        else:
            extra_outs = [None]

        return x.float(), actions.float(), proprios.float(), extra_outs

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct latent tensors from patch tokens.

        Args:
            x (Tensor):
                Patch tokens [B*V, F*H*W, C_out * prod(patch_size)]
            grid_sizes (list[int]):
                Patch grid dimensions [F_patches, H_patches, W_patches]

        Returns:
            Tensor: Reconstructed latents [B*V, C_out, F*p, H*q, W*r]
        """

        x = rearrange(
            x,
            "b (f h w) (p q r c) -> b c (f p) (h q) (w r)",
            f=grid_sizes[0],
            h=grid_sizes[1],
            w=grid_sizes[2],
            p=self.patch_size[0],
            q=self.patch_size[1],
            r=self.patch_size[2],
            c=self.out_dim,
        )
        return x

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
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        for head in self.extra_heads:
            nn.init.zeros_(head.head.weight)

    def init_new_weights(self):
        r"""
        Re-initialize weights for newly added modules that are NOT in the
        pretrained checkpoint.  Must be called AFTER from_pretrained() because
        diffusers creates the model on a meta device, making the init_weights()
        call inside __init__ a no-op for these parameters.
        """
        nn.init.normal_(self.view_embedding.weight, std=0.02)

        for m in self.action_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.action_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.proprio_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.proprio_decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for head in self.extra_heads:
            head.modulation.data.copy_(torch.randn(1, 2, self.dim) / self.dim**0.5)
            nn.init.normal_(head.head.weight, std=0.02)
            if head.head.bias is not None:
                nn.init.zeros_(head.head.bias)

    def copy_weights_to_extra_blocks(self):
        r"""
        Copy weights from main blocks to extra blocks.
        """
        for hi in range(self.num_modalities - 1):
            for bi in range(self.num_extra_layers):
                self.extra_blocks[hi][bi].load_state_dict(
                    self.blocks[bi + self.num_layers - self.num_extra_layers].state_dict()
                )
