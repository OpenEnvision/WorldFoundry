# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Pyramid UCPE with the reference FP32 attention and residual semantics.

Storage dtype is independent from compute dtype. In particular, compact UCPE
must not downcast camera embeddings or Q/K/V to its BF16 storage dtype.
"""

import torch
from einops import rearrange, repeat
from ..ucpe.camera import UcpeSelfAttention
from ..ucpe.bridge import _flash_attention_sdpa as flash_attention


class FastUcpeSelfAttention(UcpeSelfAttention):
    def forward(self, x: torch.Tensor, control_camera_dit_input: dict):
        """``x`` is ``[B, T, D]`` over the current-chunk tokens only."""
        B, T, D = x.shape
        num_cameras = control_camera_dit_input["viewmats"].shape[1]
        grid_h = control_camera_dit_input.get("patches_y", self.patches_y)
        grid_w = control_camera_dit_input.get("patches_x", self.patches_x)
        expected = num_cameras * grid_h * grid_w
        assert T == expected or T == num_cameras, (
            f"Expected token count {expected} ({num_cameras}x{grid_h}x{grid_w}) or {num_cameras}, got {T}"
        )

        if hasattr(self, "cam_encoder") and "cam_emb" in control_camera_dit_input:
            cam_emb = control_camera_dit_input["cam_emb"]
            y = self.cam_encoder(cam_emb)
            if y.shape[1] != T:
                hw = T // cam_emb.shape[1]
                y = repeat(y, "b f d -> b (f hw) d", hw=hw)
            x = x + y

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        self.prope_attn._precompute_and_cache_apply_fns(
            viewmats=control_camera_dit_input["viewmats"],
            Ks=control_camera_dit_input.get("K", None),
            coeffs_x=control_camera_dit_input.get("coeffs_x", None),
            coeffs_y=control_camera_dit_input.get("coeffs_y", None),
        )

        q = self.prope_attn._apply_to_q(q)
        k = self.prope_attn._apply_to_kv(k)
        v = self.prope_attn._apply_to_kv(v)

        q = rearrange(q, "b h t d -> b t (h d)")
        k = rearrange(k, "b h t d -> b t (h d)")
        v = rearrange(v, "b h t d -> b t (h d)")

        out = flash_attention(q, k, v, num_heads=self.num_heads)

        out = rearrange(out, "b t (h d) -> b h t d", h=self.num_heads)
        out = self.prope_attn._apply_to_o(out)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)
