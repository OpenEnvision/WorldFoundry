"""DualCamCtrl stacked 3D fusion with per-block frame gates.

Alternative to :mod:`.dualcamctrl_fusion` when DualCamCtrl wants a deeper
mixer.  Each :class:`Conv3DBlock` is depthwise 3x3x3 → pointwise → a
spatial-pooled frame gate.  ``proj_out`` is zero-init for residual safety.

Camera-control fusion only.  No VACE, action, causal, or linear attn.
"""

import torch
import torch.nn as nn
from einops import rearrange


class Conv3DBlock(nn.Module):
    """
    Single 3D conv block:
      - depthwise 3D conv (kernel=3, groups=channels)
      - SiLU
      - pointwise 3D conv (kernel=1)
      - frame gate: spatial avg pool -> conv1x1x1 -> sigmoid -> broadcast
      - residual connection
    Input/Output shape: (B, C, F, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # depthwise conv
        self.dw = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            groups=channels,
            bias=False,
        )
        # pointwise conv
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        # frame gate: operate on (B, C, F, 1, 1) after spatial avgpool
        # we use a small conv to allow mixing across channels if needed
        self.frame_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()

        # init
        nn.init.xavier_uniform_(self.dw.weight)
        nn.init.xavier_uniform_(self.pw.weight)
        nn.init.zeros_(self.frame_conv.weight)

        if self.frame_conv.bias is not None:
            nn.init.zeros_(self.frame_conv.bias)

    def forward(self, x):
        # x: B,C,F,H,W
        identity = x

        z = self.dw(x)
        z = self.act(z)
        z = self.pw(z)
        # frame gate: spatial average
        pooled = z.mean(dim=[3, 4], keepdim=True)  # B,C,F,1,1
        gate = torch.sigmoid(self.frame_conv(pooled))  # B,C,F,1,1
        z = z * gate  # broadcast over H,W

        # residual
        out = identity + z
        return out


class ResidualFusion3D_Stack(nn.Module):
    """
    Fusion module similar to your original ResidualFusion3D_Gating but:
      - use main/control 1x1 embed -> fuse_pw1
      - stack N Conv3DBlock (each has frame gate)
      - fuse_pw2 -> proj_out (zero-init) for residual safety
    Inputs x, r are expected as [B, F*H*W, C]
    """
    def __init__(self, C, reduction=4, num_blocks=4):
        super().__init__()
        mid = max(4, C // reduction)
        self.C = C
        self.mid = mid
        self.num_blocks = num_blocks

        # main / control embed 1x1x1
        self.main_embed = nn.Conv3d(C, mid, kernel_size=1, bias=False)
        self.control_embed = nn.Conv3d(C, mid, kernel_size=1, bias=False)

        # fuse pointwise before blocks
        self.fuse_pw1 = nn.Conv3d(mid * 2, mid, kernel_size=1, bias=False)

        # stack of Conv3DBlock
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Conv3DBlock(mid))
        self.blocks = nn.ModuleList(blocks)

        # restore channel
        self.fuse_pw2 = nn.Conv3d(mid, C, kernel_size=1, bias=False)

        # zero-init proj_out for residual safety
        self.proj_out = nn.Conv3d(C, C, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj_out.weight)

        self.act = nn.SiLU()

        # init other convs
        for m in [self.main_embed, self.control_embed, self.fuse_pw1, self.fuse_pw2]:
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, r, f, h, w):
        """
        x, r: [B, F*H*W, C]
        returns v: [B, F*H*W, C]
        """
        B = x.shape[0]
        C = x.shape[-1]
        assert C == self.C, f"Expected channel {self.C}, got {C}"

        # Fold packed tokens back to a 3D volume for the stacked mixer.
        x_3d = rearrange(x, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
        r_3d = rearrange(r, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        # embed
        x_e = self.main_embed(x_3d)  # B,mid,F,H,W
        r_e = self.control_embed(r_3d)  # B,mid,F,H,W

        fused = torch.cat([x_e, r_e], dim=1)  # B, 2*mid, F, H, W
        z = self.fuse_pw1(fused)
        z = self.act(z)

        # stacked blocks (each block has its own frame gate and residual)
        for blk in self.blocks:
            z = blk(z)

        z = self.act(z)
        z = self.fuse_pw2(z)  # B,C,F,H,W

        # residual safe projection
        v = self.proj_out(x_3d + z)  # apply proj_out after adding residual

        v = rearrange(v, "b c f h w -> b (f h w) c")
        return v


__all__ = ["Conv3DBlock", "ResidualFusion3D_Stack"]
