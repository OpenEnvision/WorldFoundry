"""
Conv head is from MoGe (https://github.com/microsoft/moge)
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.depth.moge.model.v1 import ResidualConvBlock
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.heads.utils import (
    create_uv_grid as normalized_view_plane_uv,
)


class ConvHead(nn.Module):
    def __init__(
        self,
        num_features: int,
        dim_in: int,
        dim_out: list[int],
        dim_proj: int = 512,
        dim_upsample: list[int] = [256, 128, 128],  # noqa: B006 - Keeps the upstream default object and signature.
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal["group_norm", "layer_norm"] = "group_norm",
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        projects: nn.Module = None,
        using_uv: bool = True,
    ):
        super().__init__()

        self.using_uv = using_uv
        if projects is not None:
            self.projects = projects

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_ch + 2 if using_uv else in_ch, out_ch),
                    *(
                        ResidualConvBlock(
                            out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample, strict=False)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
            ]
        )

        self.output_block = nn.ModuleList(
            [
                self._make_output_block(
                    dim_upsample[-1] + 2 if using_uv else dim_upsample[-1],
                    dim_out_,
                    dim_times_res_block_hidden,
                    last_res_blocks,
                    last_conv_channels,
                    last_conv_size,
                    res_block_norm,
                )
                for dim_out_ in dim_out
            ]
        )

    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        dim_times_res_block_hidden: int,
        last_res_blocks: int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: Literal["group_norm", "layer_norm"],
    ):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            *(
                ResidualConvBlock(
                    last_conv_channels,
                    last_conv_channels,
                    dim_times_res_block_hidden * last_conv_channels,
                    activation="relu",
                    norm=res_block_norm,
                )
                for _ in range(last_res_blocks)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=last_conv_size,
                stride=1,
                padding=last_conv_size // 2,
                padding_mode="replicate",
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image: torch.Tensor = None,
        patch_h: int | None = None,
        patch_w: int | None = None,
    ):
        if image is not None:
            img_h, img_w = image.shape[-2:]
            patch_h, patch_w = img_h // 14, img_w // 14
        else:
            assert patch_h is not None and patch_w is not None
            img_h = patch_h * 14
            img_w = patch_w * 14

        # Process the hidden states
        if self.projects is not None:
            x = self.projects(hidden_states).permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()
        else:
            x = hidden_states

        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for _i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            if self.using_uv:
                uv = normalized_view_plane_uv(
                    width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
                )
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        if self.using_uv:
            uv = normalized_view_plane_uv(
                width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)

        return output
