"""DualCamCtrl gated 3D residual fusion (main tokens × control tokens).

Used by :class:`WanControlNet` when ``use_gate_3d_linear`` is set.
Packed ``[B, F*H*W, C]`` streams are reshaped to 3D, reduced, mixed with
depthwise temporal convs, then sigmoid-gated back onto the main stream.
``proj_out`` is zero-initialized so the adapter starts as an identity.

Camera-control fusion only.  No VACE, action, causal, or linear attn.
"""

import torch
import torch.nn as nn
from einops import rearrange


class ResidualFusion3D_Gating(nn.Module):
    """Per-voxel gated fusion of main and control hidden states."""

    def __init__(self, C, reduction=4):
        super().__init__()
        mid = max(8, C // reduction)

        # 1x1x1 channel reduction for the main and control volumes.
        self.main_embed = nn.Conv3d(C, mid, kernel_size=1, bias=False)
        self.control_embed = nn.Conv3d(C, mid, kernel_size=1, bias=False)

        # Temporal depthwise conv (3,1,1)
        self.fuse_temporal = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=mid,
                    out_channels=mid,
                    kernel_size=(3, 3, 3),
                    padding=(1, 1, 1),
                    groups=mid,
                    bias=False,
                )
                for _ in range(3)
            ]
        )

        # pointwise convs before/after factorized convs
        self.fuse_pw1 = nn.Conv3d(mid * 2, mid, kernel_size=1, bias=False)
        self.fuse_pw2 = nn.Conv3d(mid, C, kernel_size=1, bias=False)

        self.act = nn.SiLU()

        # gating per-pixel (B,C,F,H,W)
        self.gate_conv = nn.Conv3d(C, C, kernel_size=1, bias=False)

        # proj_out zero-init
        self.proj_out = nn.Conv3d(C, C, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj_out.weight)

        # init
        for m in [
            self.main_embed,
            self.control_embed,
            self.fuse_pw1,
            self.fuse_pw2,
            self.fuse_temporal,
            self.gate_conv,
        ]:
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, r, f, h, w):
        """
        x, r: [B, F*H*W, C]
        """
        B = x.shape[0]
        C = x.shape[-1]

        # Fold packed tokens back to a 3D volume for the conv mixer.
        x = rearrange(x, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
        r = rearrange(r, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        # embed
        x_ = self.main_embed(x)
        r_ = self.control_embed(r)

        fused = torch.cat([x_, r_], dim=1)  # B,2*mid,F,H,W
        z = self.fuse_pw1(fused)
        z = self.act(z)

        # temporal conv
        for conv in self.fuse_temporal:
            z = conv(z)
        z = self.act(z)

        # channel restore
        z = self.fuse_pw2(z)

        # per-pixel gating
        gate = torch.sigmoid(self.gate_conv(z))  # B,C,F,H,W
        v = x + z * gate

        v = self.proj_out(v)  # zero-init projection (residual safe)

        v = rearrange(v, "b c f h w -> b (f h w) c")
        return v


__all__ = ["ResidualFusion3D_Gating"]
