"""Matrix Game 3.5 Wan DiT with optional projective camera RoPE.

Architecture
------------
``patch_embedding`` is Conv3d: ``B C F H W`` → ``B C F' H' W'``, then
tokens are ``B (F'H'W') C``.  Factorized 3D RoPE splits each head into
(T, H, W) bands.  When PRoPE is on, camera viewmats replace (or zero)
the temporal band so cross-frame addressing comes from extrinsics.
Subject-ref memory adds learned embeddings concatenated into the stream.

Reuses Wan ``MLP`` / ``CrossAttention`` / ``Head`` from ``../wan.model``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.core.attention import (
    apply_complex_rotary_embedding as rope_apply,
)
from worldfoundry.core.attention import (
    complex_rotary_frequencies,
    flattened_multihead_attention,
)
from worldfoundry.core.attention.rotary.projective_rope import prope_dot_product_attention
from worldfoundry.core.nn import RMSNorm, sinusoidal_embedding_1d
from worldfoundry.core.nn import scale_shift as modulate

from ..wan.adapter import SimpleAdapter
from ..wan.model import (
    MLP,
    CrossAttention,
    GateModule,
    Head,
)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Build complex RoPE tables for T/H/W (plus finer QL spatial tables)."""
    temporal_dim = dim - 2 * (dim // 3)
    spatial_dim = dim // 3
    f_freqs_cis = complex_rotary_frequencies(temporal_dim, end, theta)
    h_freqs_cis = complex_rotary_frequencies(spatial_dim, end, theta)
    w_freqs_cis = complex_rotary_frequencies(spatial_dim, end, theta)
    ql_h_freqs_cis = complex_rotary_frequencies(spatial_dim, end, theta, subdivisions=16)
    ql_w_freqs_cis = complex_rotary_frequencies(spatial_dim, end, theta, subdivisions=16)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis, ql_h_freqs_cis, ql_w_freqs_cis


class AttentionModule(nn.Module):
    """Flattened multi-head SDPA used when PRoPE is off."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None):
        """Attend ``B S (H·D)`` QKV; optional mask is passed through."""
        x = flattened_multihead_attention(
            q,
            k,
            v,
            self.num_heads,
            attn_mask=attn_mask,
        )
        return x


class SelfAttention(nn.Module):
    """Wan self-attn with 3D RoPE and optional PRoPE camera attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        use_prope: bool = False,
        prope_disable_native_rope: bool = False,
        prope_disable_t_rope: bool = False,
        prope_camera_layout: str = "full",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        # Camera tiling for PRoPE attention (see PROPE_CAMERA_LAYOUTS):
        # "full" = legacy tiling over all head_dim; "sf13" = contiguous
        # 2-camera (sub-frames {1,3}) layout in the rope-free t-band.
        self.prope_camera_layout = prope_camera_layout
        # Complex-pair count of the temporal RoPE band; mirrors the
        # head_dim split in `precompute_freqs_cis_3d` (t gets
        # head_dim - 2*(head_dim//3) real dims, x/y get head_dim//3 each).
        self.rope_t_pairs = (self.head_dim - 2 * (self.head_dim // 3)) // 2

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attn_mask=None, camera_info=None):
        """Self-attend ``B S C``; ``freqs`` is complex RoPE, ``camera_info`` optional PRoPE."""
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        use_prope_attention = self.use_prope and camera_info is not None
        if not (use_prope_attention and self.prope_disable_native_rope):
            rope_freqs = freqs
            if use_prope_attention and self.prope_disable_t_rope:
                # PRoPE blocks keep x/y RoPE (intra-frame spatial structure)
                # but drop the temporal band: cross-frame addressing is
                # delegated to the camera matrices, and the t-band channels
                # become a RoPE-phase-free subspace for them. 1+0j is the
                # identity rotation.
                rope_freqs = freqs.clone()
                rope_freqs[..., : self.rope_t_pairs] = 1
            q = rope_apply(q, rope_freqs, self.num_heads)
            k = rope_apply(k, rope_freqs, self.num_heads)
        if use_prope_attention:
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            view_change_positions = camera_info[2] if len(camera_info) > 2 else None
            x = prope_dot_product_attention(
                q,
                k,
                v,
                viewmats=camera_info[1],
                view_change_positions=view_change_positions,
                camera_layout=getattr(self, "prope_camera_layout", "full"),
                attn_mask=attn_mask,
            )
            x = rearrange(x, "b n s d -> b s (n d)")
        else:
            x = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o(x)


class DiTBlock(nn.Module):
    """AdaLN-modulated self-attn + text CA + FFN (Wan block with PRoPE hooks)."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        use_prope: bool = False,
        prope_disable_native_rope: bool = False,
        prope_disable_t_rope: bool = False,
        prope_camera_layout: str = "full",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        self.prope_camera_layout = prope_camera_layout

        self.self_attn = SelfAttention(
            dim,
            num_heads,
            eps,
            use_prope=use_prope,
            prope_disable_native_rope=prope_disable_native_rope,
            prope_disable_t_rope=prope_disable_t_rope,
            prope_camera_layout=prope_camera_layout,
        )
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_mask=None,
        camera_info=None,
        cross_attn_keep_mask=None,
    ):
        """One residual block on ``B S C`` tokens; ``t_mod`` is ``6×C`` AdaLN (or per-token)."""
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        camera_info = camera_info if self.use_prope else None
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(input_x, freqs, attn_mask=attn_mask, camera_info=camera_info),
        )
        ca = self.cross_attn(self.norm3(x), context)
        if cross_attn_keep_mask is not None:
            ca = ca * cross_attn_keep_mask.to(device=ca.device, dtype=ca.dtype).view(1, -1, 1)
        x = x + ca
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class WanModel(torch.nn.Module):
    """Matrix Game 3.5 backbone: 3D patch DiT with optional camera PRoPE and ref memory."""

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        use_prope: bool = False,
        prope_disable_native_rope: bool = False,
        prope_disable_t_rope: bool = False,
        prope_camera_layout: str = "full",
        subject_ref_memory_enabled: bool = False,
        subject_ref_memory_max_refs: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        self.prope_camera_layout = prope_camera_layout
        self.subject_ref_memory_enabled = False

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    has_image_input,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    use_prope=use_prope,
                    prope_disable_native_rope=prope_disable_native_rope,
                    prope_disable_t_rope=prope_disable_t_rope,
                    prope_camera_layout=prope_camera_layout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

        if subject_ref_memory_enabled:
            self.enable_subject_ref_memory(subject_ref_memory_max_refs)

    def enable_subject_ref_memory(self, max_refs: int = 2):
        """Allocate learned index/type/local-pos embeddings for subject-reference tokens."""
        max_refs = max(1, int(max_refs))
        if self.subject_ref_memory_enabled:
            if int(self.subject_ref_index_embedding.shape[0]) != max_refs:
                raise ValueError(
                    "subject_ref_memory is already enabled with "
                    f"{int(self.subject_ref_index_embedding.shape[0])} refs, "
                    f"cannot re-enable with {max_refs} refs."
                )
            return
        self.subject_ref_memory_enabled = True
        self.subject_ref_memory_max_refs = max_refs
        self.subject_ref_memory_local_pos_size = 64
        ref_param = next(self.parameters(), None)
        device = ref_param.device if ref_param is not None else None
        dtype = ref_param.dtype if ref_param is not None else None
        self.subject_ref_index_embedding = nn.Parameter(torch.zeros(max_refs, self.dim, device=device, dtype=dtype))
        self.subject_ref_type_embedding = nn.Parameter(torch.zeros(1, self.dim, device=device, dtype=dtype))
        self.subject_ref_local_h_embedding = nn.Parameter(
            torch.zeros(
                self.subject_ref_memory_local_pos_size,
                self.dim,
                device=device,
                dtype=dtype,
            )
        )
        self.subject_ref_local_w_embedding = nn.Parameter(
            torch.zeros(
                self.subject_ref_memory_local_pos_size,
                self.dim,
                device=device,
                dtype=dtype,
            )
        )

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ):
        """Conv3d-patch ``B C F H W``; optionally add a spatial camera-control adapter."""
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """Fold ``B (F'H'W') (pt·ph·pw·C)`` back to ``B C F H W``."""
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Denoise ``B C F H W`` latents; return residual in the same layout."""
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
