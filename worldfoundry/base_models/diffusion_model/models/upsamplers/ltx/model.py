"""LTX Conv residual CNN that spatially (or temporally) upsamples VAE latents.

Used by :class:`~.component.LTXSpatialLatentProcessor` between multi-stage
passes.  Input / output are un-normalized BCTHW tensors.  Spatial 2× uses
pixel-shuffle; non-integer scales use :class:`SpatialRationalResampler`.
Temporal upsample drops the first output frame (it encodes one pixel
frame).  Not a :class:`~...contracts.LatentProcessor` by itself.
"""

import torch
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video import VideoEncoder
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.pixel_shuffle import PixelShuffleND
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.res_block import (
    ResBlock,
    _can_defer_conv_bias,
    _can_fuse_norm_act_module,
    _conv_no_bias,
)
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.spatial_rational_resampler import (
    SpatialRationalResampler,
)
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.triton_upsampler_ops import (
    _is_channels_last_nd,
    can_use_triton_upsampler_group_norm_silu,
    triton_upsampler_group_norm_silu,
)


class LatentUpsampler(torch.nn.Module):
    """
    Model to upsample VAE latents spatially and/or temporally.
    Args:
        in_channels (`int`): Number of channels in the input latent
        mid_channels (`int`): Number of channels in the middle layers
        num_blocks_per_stage (`int`): Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`): Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`): Whether to spatially upsample the latent
        temporal_upsample (`bool`): Whether to temporally upsample the latent
        spatial_scale (`float`): Scale factor for spatial upsampling
        rational_resampler (`bool`): Whether to use a rational resampler for spatial upsampling
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        self.res_blocks = torch.nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=self.spatial_scale)
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)
        self._channels_last_3d: bool = False

    def enable_channels_last_3d(self) -> int:
        """Convert Conv3d/Conv2d weights to channels-last layouts for Hopper Tensor Cores."""
        from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

        report = convert_convolution_weight_layouts(self, conv2d=True, conv3d=True)
        converted = report.conv3d_converted + report.conv2d_converted
        self._channels_last_3d = report.conv3d_converted > 0
        return converted

    def _run_initial(self, x: torch.Tensor) -> torch.Tensor:
        if _can_defer_conv_bias(self.initial_conv, x) and _can_fuse_norm_act_module(
            self.initial_norm, self.initial_activation, x.device
        ):
            h = _conv_no_bias(self.initial_conv, x)
            if can_use_triton_upsampler_group_norm_silu(
                h,
                self.initial_norm.num_groups,
                self.initial_norm.weight,
                self.initial_norm.bias,
                pre_bias=self.initial_conv.bias,
            ):
                return triton_upsampler_group_norm_silu(
                    h,
                    self.initial_norm.num_groups,
                    self.initial_norm.weight,
                    self.initial_norm.bias,
                    self.initial_norm.eps,
                    pre_bias=self.initial_conv.bias,
                    apply_silu=True,
                )
            if self.initial_conv.bias is not None:
                h = h + self.initial_conv.bias.view((1, -1) + (1,) * (h.ndim - 2)).to(h)
            return self.initial_activation(self.initial_norm(h))
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        return self.initial_activation(x)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample a BCTHW latent; 2-D path folds time into the batch."""
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self._run_initial(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = latent
            if self._channels_last_3d and x.ndim == 5 and x.device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last_3d)
            x = self._run_initial(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                # remove the first frame after upsampling.
                # This is done because the first frame encodes one pixel frame.
                x = x[:, :, 1:, :, :]
            elif isinstance(self.upsampler, SpatialRationalResampler):
                x = self.upsampler(x)
            elif _is_channels_last_nd(x):
                # Zero-copy 5-D NDHWC <-> 4-D NHWC view around the 2-D spatial upsampler.
                _, c_mid, _, h_in, w_in = x.shape
                x_4d = x.permute(0, 2, 3, 4, 1).reshape(b * f, h_in, w_in, c_mid).permute(0, 3, 1, 2)
                x_4d = self.upsampler(x_4d)
                _, c_up, h_up, w_up = x_4d.shape
                x = x_4d.permute(0, 2, 3, 1).contiguous().reshape(b, f, h_up, w_up, c_up).permute(0, 4, 1, 2, 3)
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


def upsample_video(latent: torch.Tensor, video_encoder: VideoEncoder, upsampler: "LatentUpsampler") -> torch.Tensor:
    """
    Apply upsampling to the latent representation using the provided upsampler,
    with normalization and un-normalization based on the video encoder's per-channel statistics.
    Args:
        latent: Input latent tensor of shape [B, C, F, H, W].
        video_encoder: VideoEncoder with per_channel_statistics for normalization.
        upsampler: LatentUpsampler module to perform upsampling.
    Returns:
        torch.Tensor: Upsampled and re-normalized latent tensor.
    """
    latent = video_encoder.per_channel_statistics.un_normalize(latent)
    latent = upsampler(latent)
    latent = video_encoder.per_channel_statistics.normalize(latent)
    return latent
