# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat

from .attention import flash_attention

__all__ = ['StateAdapterWanModel']


def modulate(x, shift, scale):
    """
    Apply affine transformation for AdaLN.
    Args:
        x: input tensor [B, L, C]
        shift: shift parameter [B, L, C]
        scale: scale parameter [B, L, C]
    Returns:
        Modulated tensor [B, L, C]
    """
    return x * (1 + scale) + shift

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


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.camera_21 import rope_apply


def rope_params_with_offset(positions, dim, theta=10000):
    """
    Generate RoPE frequencies for arbitrary position indices (can be negative).

    Args:
        positions: 1D tensor of position indices (can include negative values)
        dim: dimension of the RoPE embedding
        theta: base for the exponential

    Returns:
        Complex frequencies tensor of shape [len(positions), dim//2]
    """
    assert dim % 2 == 0
    positions = positions.to(torch.float64)
    inv_freq = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=positions.device).to(torch.float64) / dim)
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply_state_adapter(x, h, w, num_t, num_p, num_r, freqs, max_seq_len=1024, num_r_scene=None, num_r_inst=None):
    """
    Apply segment-aware RoPE for LiveWorld [T, P, R_scene, R_inst] frame order.

    Temporal RoPE positions:
    - T frames: 0, 1, 2, ..., T-1 (target frames being generated)
    - P frames: -P, -P+1, ..., -1 (preceding frames, temporally before T)
    - R_scene frames: 0 for all (scene reference, no temporal relationship)
    - R_inst frames: num_t for all (instance reference, distinct from scene)

    Spatial RoPE (H, W) is applied normally for all frames.

    Args:
        x: input tensor [B, seq_len, num_heads, head_dim]
        h: spatial height (after patch embedding)
        w: spatial width (after patch embedding)
        num_t: number of target frames
        num_p: number of preceding frames
        num_r: number of reference frames (legacy)
        num_r_scene: number of scene reference frames
        num_r_inst: number of instance reference frames
        freqs: precomputed base frequencies [max_seq_len, head_dim//2]
        max_seq_len: maximum sequence length for RoPE

    Returns:
        x with RoPE applied [B, seq_len, num_heads, head_dim]
    """
    device = x.device
    n, c = x.size(2), x.size(3) // 2  # num_heads, half of head_dim

    # Split freqs for temporal and spatial dimensions
    # freqs layout: [temporal_dim, h_dim, w_dim]
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_t_base, freqs_h, freqs_w = freqs_split

    if num_r_scene is None or num_r_inst is None:
        total_frames = num_t + num_p + num_r
        num_r_scene = num_r
        num_r_inst = 0
    else:
        total_frames = num_t + num_p + num_r_scene + num_r_inst
    tokens_per_frame = h * w
    total_seq_len = total_frames * tokens_per_frame

    # Build temporal position indices for each segment
    # T frames: positions 0, 1, ..., T-1
    t_positions = torch.arange(0, num_t, device=device)
    # P frames: positions -P, -P+1, ..., -1
    p_positions = torch.arange(-num_p, 0, device=device)
    # R_scene frames: all position 0 (no temporal relationship)
    r_scene_positions = torch.zeros(num_r_scene, device=device, dtype=torch.long)
    # R_inst frames: fixed position after target frames (distinct from scene)
    r_inst_positions = torch.full((num_r_inst,), num_t, device=device, dtype=torch.long)

    # Concatenate in [T, P, R_scene, R_inst] order
    frame_positions = torch.cat([t_positions, p_positions, r_scene_positions, r_inst_positions])

    # Generate temporal frequencies for these positions (can be negative)
    # We need to compute RoPE for arbitrary positions, not just 0..N-1
    dim_t = c - 2 * (c // 3)
    freqs_t = rope_params_with_offset(frame_positions, dim_t * 2).to(device)  # [total_frames, dim_t]

    # Build the full frequency tensor for all tokens
    # Each frame has h*w tokens, all with same temporal position but different spatial positions
    output = []
    for b in range(x.size(0)):
        x_b = x[b, :total_seq_len].to(torch.float64).reshape(total_seq_len, n, -1, 2)
        x_b = torch.view_as_complex(x_b)  # [seq_len, n, c]

        # Build freqs for each token
        freqs_list = []
        for f_idx in range(total_frames):
            # Temporal freq for this frame (same for all h*w tokens in this frame)
            freq_t = freqs_t[f_idx:f_idx+1].expand(h * w, -1)  # [h*w, dim_t]
            # LiveWorldl freqs
            freq_h = freqs_h[:h].view(h, 1, -1).expand(h, w, -1).reshape(h * w, -1)  # [h*w, dim_h]
            freq_w = freqs_w[:w].view(1, w, -1).expand(h, w, -1).reshape(h * w, -1)  # [h*w, dim_w]
            # Concatenate
            freq_frame = torch.cat([freq_t, freq_h, freq_w], dim=-1)  # [h*w, c]
            freqs_list.append(freq_frame)

        freqs_full = torch.cat(freqs_list, dim=0).unsqueeze(1)  # [seq_len, 1, c]

        # Apply RoPE
        x_b = torch.view_as_real(x_b * freqs_full).flatten(2)  # [seq_len, n, head_dim]

        # Append any remaining tokens (padding)
        if x.size(1) > total_seq_len:
            x_b = torch.cat([x_b, x[b, total_seq_len:]], dim=0)

        output.append(x_b)

    return torch.stack(output).type_as(x)


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


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


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

    def forward(self, x, seq_lens, grid_sizes, freqs, backbone_segment_info=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            backbone_segment_info(dict, optional): For LiveWorld segment-aware RoPE
                - num_t: number of target frames
                - num_p: number of preceding frames
                - num_r: number of reference frames
                - h: spatial height (after patch embedding)
                - w: spatial width (after patch embedding)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Apply RoPE (segment-aware for LiveWorld, standard otherwise)
        num_t = num_p = num_r = num_r_scene = num_r_inst = None
        h = w = None
        if backbone_segment_info is not None:
            num_t = backbone_segment_info['num_t']
            num_p = backbone_segment_info['num_p']
            num_r = backbone_segment_info['num_r']
            num_r_scene = backbone_segment_info.get('num_r_scene', None)
            num_r_inst = backbone_segment_info.get('num_r_inst', None)
            h = backbone_segment_info['h']
            w = backbone_segment_info['w']
            q = rope_apply_state_adapter(q, h, w, num_t, num_p, num_r, freqs, num_r_scene=num_r_scene, num_r_inst=num_r_inst)
            k = rope_apply_state_adapter(k, h, w, num_t, num_p, num_r, freqs, num_r_scene=num_r_scene, num_r_inst=num_r_inst)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x



class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

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
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, **kwargs):
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
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 sp_block_id=None,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.sp_block_id = sp_block_id

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

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
        sp_hints=None,
        sp_context_scale=1.0,
        sp_hint_offset=0,
        backbone_segment_info=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            sp_hints(List[Tensor], optional): State Adapter hint features for injection
            sp_context_scale(float): Scale factor for State Adapter hint features
            sp_hint_offset(int): Token offset for State Adapter hint injection (for LiveWorld: skip R reference tokens)
            backbone_segment_info(dict, optional): For LiveWorld segment-aware RoPE
                - num_t: number of target frames
                - num_p: number of preceding frames
                - num_r: number of reference frames
                - h: spatial height (after patch embedding)
                - w: spatial width (after patch embedding)
        """
        # Handle per-frame timestep [B, F, 6, C] vs uniform [B, 6, C]
        if e.dim() == 4:
            # Per-frame timestep: e is [B, F, 6, dim]
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)  # 6 x [B, F, 1, dim]

            # Self-attention: unflatten x to [B, F, frame_seqlen, dim], apply modulation, flatten back
            x_norm = self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            y = self.self_attn((x_norm * (1 + e[1]) + e[0]).flatten(1, 2), seq_lens, grid_sizes, freqs, backbone_segment_info)
            # Unflatten y, multiply by e[2], then flatten before adding to x
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

            # Cross-attention & FFN
            x = x + self.cross_attn(x, context, context_lens)
            x_norm2 = self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            y = self.ffn((x_norm2 * (1 + e[4]) + e[3]).flatten(1, 2))
            # Unflatten y, multiply by e[5], then flatten before adding to x
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        else:
            # Uniform timestep: e is [B, 6, dim]
            e = (self.modulation + e).chunk(6, dim=1)

            # Self-attention
            y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs, backbone_segment_info)
            x = x + y * e[2]

            # Cross-attention & FFN
            x = x + self.cross_attn(x, context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]

        # State Adapter hint injection for LiveWorld [T, P, R] frame order
        # hint has T+P tokens (from sp_context), x has T+P+R tokens
        # With [T, P, R] order, State Adapter hints go to first T+P tokens (sp_hint_offset=0)
        if self.sp_block_id is not None and sp_hints is not None:
            hint = sp_hints[self.sp_block_id] * sp_context_scale
            hint_len = hint.shape[1]
            # Add hint to first hint_len tokens (T+P portion)
            x[:, :hint_len, :] = x[:, :hint_len, :] + hint

        return x


class SPControlBlock(WanAttentionBlock):
    """
    State Adapter Control Block that processes control signals.
    Extends WanAttentionBlock with additional projection layers for feature stacking.
    """
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
            sp_ctrl_block_id=0,
            **kwargs
    ):
        # Don't pass sp_block_id to parent (control blocks don't inject hints)
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps, sp_block_id=None, **kwargs)
        self.sp_ctrl_block_id = sp_ctrl_block_id

        # Zero-initialized projection layers for State Adapter feature stacking
        if sp_ctrl_block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, e, **kwargs):
        """
        Forward pass for State Adapter control block.

        Args:
            c: Control features (stacked tensor or single tensor)
            x: Input features from main model (for block 0 initialization)
            e: Timestep embedding for attention block
            **kwargs: Additional arguments for parent forward

        Returns:
            Updated control features stack
        """
        if self.sp_ctrl_block_id == 0:
            # First block: initialize control features from main features
            c = self.before_proj(c) + x
            all_c = []
        else:
            # Subsequent blocks: unstack and process last feature
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Process through attention block (pass c as x, e as e)
        c = super().forward(c, e, **kwargs)

        # Create skip connection and stack
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


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
            e(Tensor): Shape [B, C] (uniform) or [B, F, C] (per-frame)
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if e.dim() == 3:
            # Per-frame timestep: e is [B, F, dim]
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).chunk(2, dim=2)  # 2 x [B, F, 1, dim]
            x_norm = self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            x = self.head((x_norm * (1 + e[1]) + e[0]).flatten(1, 2))
        else:
            # Uniform timestep: e is [B, dim]
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
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


class StateAdapterWanModel(ModelMixin, ConfigMixin):
    r"""
    LiveWorld model for camera-controlled video generation.

    Frame order: [T, P, R] where T=target (noisy), P=preceding (clean), R=reference (clean)
    - T frames: noisy target frames with actual diffusion timestep
    - P frames: clean preceding frames with timestep=0
    - R frames: clean reference frames with timestep=0

    State Adapter processes [T_scene, P_scene] (scene projections for target and preceding frames)
    while main model has [T, P, R] frames.

    With [T, P, R] order, State Adapter hints are added to the FIRST T tokens (sp_hint_offset=0).
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
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
                 add_control_adapter=False,
                 in_dim_control_adapter=24,
                 downscale_factor_control_adapter=8,
                 sp_layers=None,
                 sp_in_dim=None,
                 **kwargs):
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
            add_control_adapter (`bool`, *optional*, defaults to False):
                Enable camera control adapter
            in_dim_control_adapter (`int`, *optional*, defaults to 24):
                Input dimension for camera control adapter (e.g., 24 for camera parameters)
            downscale_factor_control_adapter (`int`, *optional*, defaults to 8):
                Downscale factor for control adapter
            sp_layers (`list`, *optional*, defaults to None):
                Layer indices where State Adapter hints are injected. If None, State Adapter is disabled.
                Example: [0, 2, 4, 6, ...] for every other layer
            sp_in_dim (`int`, *optional*, defaults to None):
                Input channels for State Adapter control signals. If None, uses in_dim.
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        # State Adapter configuration
        self.sp_layers = sp_layers
        self.sp_in_dim = in_dim if sp_in_dim is None else sp_in_dim
        self.enable_sp = sp_layers is not None and len(sp_layers) > 0

        if self.enable_sp:
            assert 0 in sp_layers, "State Adapter layer 0 must be included for proper initialization"
            self.sp_layers_mapping = {i: n for n, i in enumerate(sp_layers)}

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
        self.local_attn_size = 21

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # ti2v uses t2v_cross_attn (Wan2.2-5B structure, no separate k_img/v_img layers)
        if model_type in ['t2v', 'ti2v']:
            cross_attn_type = 't2v_cross_attn'
        else:  # i2v
            cross_attn_type = 'i2v_cross_attn'

        # Create blocks with optional State Adapter block IDs
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                sp_block_id=self.sp_layers_mapping[i] if self.enable_sp and i in self.sp_layers else None,
                **kwargs
            )
            for i in range(num_layers)
        ])

        # State Adapter control blocks
        if self.enable_sp:
            self.sp_blocks = nn.ModuleList([
                SPControlBlock(
                    't2v_cross_attn', dim, ffn_dim, num_heads, window_size,
                    qk_norm, cross_attn_norm, eps,
                    sp_ctrl_block_id=i, **kwargs
                )
                for i in range(len(self.sp_layers))
            ])

            # Zero-initialize ALL State Adapter parameters (sp_blocks + sp_patch_embedding)
            # This ensures that if pretrained weights have shape mismatch and are skipped,
            # State Adapter starts as zero and doesn't inject random noise into the model.
            # Pretrained weights will be loaded afterwards in _load_sp_weights().
            for param in self.sp_blocks.parameters():
                nn.init.zeros_(param)

            # State Adapter patch embedding for control signals
            self.sp_patch_embedding = nn.Conv3d(
                self.sp_in_dim, dim, kernel_size=patch_size, stride=patch_size
            )
            nn.init.zeros_(self.sp_patch_embedding.weight)
            nn.init.zeros_(self.sp_patch_embedding.bias)
        else:
            self.sp_blocks = None
            self.sp_patch_embedding = None

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

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

        self.control_adapter = None

        # initialize weights
        # #! dzc
        # self.init_weights()

        self.gradient_checkpointing = False

    # 在 CausalWanModel 内部添加或替换该方法
    def _set_gradient_checkpointing(self, *args, **kwargs):
        """
        兼容 Diffusers 新旧两个调用方式：
        - 旧: _set_gradient_checkpointing(value: bool)
        - 新: _set_gradient_checkpointing(enable: bool = True, gradient_checkpointing_func: Optional[Callable] = None)
        """
        # 1) 解析参数，兼容两种风格
        if "enable" in kwargs or "gradient_checkpointing_func" in kwargs:
            enable = kwargs.get("enable", True)
            grad_ckpt_func = kwargs.get("gradient_checkpointing_func", None)
        elif len(args) >= 1 and isinstance(args[0], bool):
            enable = args[0]
            grad_ckpt_func = None
        else:
            enable = True
            grad_ckpt_func = None

        # 2) 递归地把设置应用到子模块（如果它们支持）
        def _apply(module):
            # 一些自定义模块用这个 flag
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable

            # Diffusers/Transformers 常见写法：提供 set_gradient_checkpointing 方法
            if enable and hasattr(module, "set_gradient_checkpointing"):
                # 新接口希望传函数；旧接口通常不需要
                try:
                    module.set_gradient_checkpointing(grad_ckpt_func)  # 新式
                except TypeError:
                    module.set_gradient_checkpointing(enable=True)     # 旧式兜底

        # 把设置应用到整棵模型
        for m in self.modules():
            _apply(m)

    def forward(
        self,
        *args,
        **kwargs
    ):
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        y_camera=None,
        sp_context=None,
        sp_context_scale=1.0,
        sp_hint_offset=0,
        num_t=None,
        num_p=None,
        num_r=None,
        num_r_scene=None,
        num_r_inst=None,
        **kwargs,
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
            y_camera (Tensor, *optional*):
                Camera control signals [B, in_dim_control_adapter, F, H, W]
            sp_context (List[Tensor], *optional*):
                State Adapter control signals, list of [C_sp, F_sp, H, W] where F_sp may be < F_main
            sp_context_scale (float):
                Scale factor for State Adapter hint injection
            sp_hint_offset (int):
                Token offset for State Adapter hints (for LiveWorld: skip R*H*W reference tokens)
            num_t (int, optional):
                Number of target frames for segment-aware RoPE
            num_p (int, optional):
                Number of preceding frames for segment-aware RoPE
            num_r (int, optional):
                Number of reference frames for segment-aware RoPE
            num_r_scene (int, optional):
                Number of scene reference frames for segment-aware RoPE
            num_r_inst (int, optional):
                Number of instance reference frames for segment-aware RoPE

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # ti2v uses y (first frame conditioning) but not clip_fea
        if self.model_type == 'ti2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        # Apply camera control adapter if available
        if self.control_adapter is not None and y_camera is not None and (any(p.requires_grad for p in self.control_adapter.parameters()) or not x[0].requires_grad):
            # Apply camera adapter to transform camera control input
            y_camera = self.control_adapter(y_camera)
            # Split y_camera along batch dimension to match x (which is a list of tensors)
            y_camera = [y_camera[i:i+1] for i in range(y_camera.size(0))]
            # Element-wise addition of camera features to visual features
            x = [u + v for u, v in zip(x, y_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # Wan2.2-5B
        if self.model_type == "ti2v":
            # time embeddings
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            # time embeddings
            # Support per-frame timestep [B, F] like causal model
            if t.dim() == 2:
                # Per-frame timestep: t is [B, F]
                t_shape = t.shape
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
                e0 = self.time_projection(e).unflatten(
                    1, (6, self.dim)).unflatten(dim=0, sizes=t_shape)  # [B, F, 6, dim]
                # Reshape e to [B, F, dim] for head
                e = e.unflatten(dim=0, sizes=t_shape)  # [B, F, dim]
            else:
                # Single timestep: t is [B]
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, dim]


        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)


        # arguments
        base_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
            )

        # State Adapter hint generation for LiveWorld [T, P, R] frame order
        sp_hints = None
        if self.enable_sp and sp_context is not None:
            # Embed State Adapter context: each u is [C, T+P, H, W] for LiveWorld (scene projections)
            c = [self.sp_patch_embedding(u.unsqueeze(0)) for u in sp_context]
            c = [u.flatten(2).transpose(1, 2) for u in c]  # list of [1, (T+P)*h*w, dim]
            c = torch.cat(c)  # [B, sp_seq_len, dim] where sp_seq_len = (T+P) * tokens_per_frame

            # For LiveWorld [T, P, R]: State Adapter processes T+P frames while main model has T+P+R frames
            # With [T, P, R] order, T+P tokens are at the BEGINNING of x
            sp_seq_len = c.shape[1]  # (T+P) * tokens_per_frame
            if sp_seq_len < seq_len:
                # LiveWorld case: main model has more frames than State Adapter
                # Slice first T+P tokens from x (sp_hint_offset=0 in [T, P, R] order)
                x_for_sp = x[:, :sp_seq_len, :]
            else:
                # Original State Adapter case: State Adapter and main model have same frames
                x_for_sp = x

            # Prepare kwargs for State Adapter blocks
            # For per-frame timestep, also slice e0 to T+P frames
            sp_e0 = e0
            if e0.dim() == 4 and sp_seq_len < seq_len:
                # Per-frame timestep with LiveWorld: e0 is [B, T+P+R, 6, dim]
                # Need to slice to [B, T+P, 6, dim] for State Adapter
                total_frames = e0.shape[1]
                tokens_per_frame = seq_len // total_frames
                num_sp_frames = sp_seq_len // tokens_per_frame
                # With [T, P, R] order, T+P frames are first, R frames are last
                sp_e0 = e0[:, :num_sp_frames]  # [B, T+P, 6, dim]

            # Build backbone_segment_info for State Adapter blocks (only T+P frames, no R)
            sp_backbone_segment_info = None
            h = grid_sizes[0, 1].item()
            w = grid_sizes[0, 2].item()
            if num_t is not None and num_p is not None:
                sp_backbone_segment_info = {
                    'num_t': num_t,
                    'num_p': num_p,
                    'num_r': 0,  # State Adapter has no reference frames
                    'h': h,
                    'w': w,
                }

            # Build grid_sizes for State Adapter blocks (only T+P frames)
            # grid_sizes is [B, 3] with [F, H, W], need to adjust F for State Adapter
            tokens_per_frame = h * w
            num_sp_frames = sp_seq_len // tokens_per_frame
            sp_grid_sizes = grid_sizes.clone()
            sp_grid_sizes[:, 0] = num_sp_frames  # Set F to T+P for State Adapter

            # Build seq_lens for State Adapter blocks
            sp_seq_lens = torch.full_like(seq_lens, sp_seq_len)

            # SPControlBlock.forward(c, x, e, **kwargs) - c, x, e are positional args
            sp_kwargs = dict(seq_lens=sp_seq_lens, grid_sizes=sp_grid_sizes,
                               freqs=self.freqs, context=context, context_lens=context_lens,
                               backbone_segment_info=sp_backbone_segment_info)
            # Process through State Adapter control blocks
            for block in self.sp_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    # Create a closure that captures the current block and kwargs
                    def create_sp_forward(blk, kw):
                        def sp_forward(c_in, x_in, e_in):
                            return blk(c_in, x_in, e_in, **kw)
                        return sp_forward
                    c = torch.utils.checkpoint.checkpoint(
                        create_sp_forward(block, sp_kwargs),
                        c, x_for_sp, sp_e0,
                        use_reentrant=False,
                    )
                else:
                    c = block(c, x_for_sp, sp_e0, **sp_kwargs)

            # Extract hints (all except the last stacked feature)
            sp_hints = torch.unbind(c)[:-1]

        # Add State Adapter hints to forward kwargs
        forward_kwargs = base_kwargs.copy()
        forward_kwargs['sp_hints'] = sp_hints
        forward_kwargs['sp_context_scale'] = sp_context_scale
        forward_kwargs['sp_hint_offset'] = sp_hint_offset

        # Build backbone_segment_info for segment-aware RoPE if segment counts are provided
        if num_t is not None and num_p is not None and num_r is not None:
            # Get spatial dimensions from grid_sizes (after patch embedding)
            # grid_sizes is [B, 3] where dims are [F, H, W]
            # For batched inputs with same spatial size, take from first sample
            h = grid_sizes[0, 1].item()
            w = grid_sizes[0, 2].item()
            forward_kwargs['backbone_segment_info'] = {
                'num_t': num_t,
                'num_p': num_p,
                'num_r': num_r,
                'num_r_scene': num_r_scene,
                'num_r_inst': num_r_inst,
                'h': h,
                'w': w,
            }
            # Debug print (remove after verification)
            # print(f"[DEBUG] backbone_segment_info: num_t={num_t}, num_p={num_p}, num_r={num_r}, h={h}, w={w}")
        else:
            forward_kwargs['backbone_segment_info'] = None
            # Debug print (remove after verification)
            # print(f"[DEBUG] backbone_segment_info is None - num_t={num_t}, num_p={num_p}, num_r={num_r}")

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **forward_kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **forward_kwargs)

        # head
        x = self.head(x, e) # [1, 32760, 1536], [1, 1536]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return torch.stack(x)



    def unpatchify(self, x, grid_sizes, c=None):
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

        c = self.out_dim if c is None else c
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
