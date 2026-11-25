"""3-D residual / mid blocks for the LTX video VAE.

:class:`ResnetBlock3D` and :class:`UNetMidBlock3D` are the
checkpoint-compatible residual stack used by
:mod:`.video_vae`.  Convs come from :mod:`.convolution`.

Tensors are BCTHW.  Header-level docs only — do not comment
every inner conv.

Sibling of the audio :mod:`~..audio.resnet`.
"""

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.common.normalization import PixelNorm
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.convolution import (
    CausalConv3d,
    make_conv_nd,
    make_linear_nd,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import NormLayerType, PaddingModeType
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.triton_vae_ops import (
    can_use_triton_ltx_residual_bias_add,
    can_use_triton_pixel_norm_adaln_silu_pad_3d,
    triton_ltx_residual_bias_add,
    triton_pixel_norm_adaln_silu,
    triton_pixel_norm_adaln_silu_pad_3d,
)
from worldfoundry.core.nn.diffusion.timestep import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)


def _has_hooks(module: nn.Module) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def _can_fuse_pixel_norm_silu(norm: nn.Module, act: nn.Module, x: torch.Tensor) -> bool:
    return (
        type(norm) is PixelNorm
        and norm.dim in (1, 1 - x.ndim)
        and type(act) is nn.SiLU
        and not _has_hooks(norm)
        and not _has_hooks(act)
    )


def _can_fuse_pixel_norm_silu_conv(
    norm: nn.Module,
    act: nn.Module,
    conv: nn.Module,
    x: torch.Tensor,
    causal: bool,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    pre_bias: torch.Tensor | None = None,
) -> bool:
    if not _can_fuse_pixel_norm_silu(norm, act, x) or type(conv) is not CausalConv3d:
        return False
    if not conv._can_use_fast_conv_path(x):
        return False
    pad, s_mode, _ = conv._fused_pad_spec(causal)
    return can_use_triton_pixel_norm_adaln_silu_pad_3d(
        x,
        pad,
        spatial_mode=s_mode,
        temporal_mode="replicate",
        scale=scale,
        shift=shift,
        pre_bias=pre_bias,
    )


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.
    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        dims: int | tuple[int, int],
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm1 = PixelNorm()

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm2 = PixelNorm()

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        self.conv_shortcut = (
            make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Using GroupNorm with 1 group is equivalent to LayerNorm but works with (B, C, ...) layout
        # avoiding the need for dimension rearrangement used in standard nn.LayerNorm
        self.norm3 = (
            nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps, affine=True)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.zeros(4, in_channels))

    def _feed_spatial_noise(
        self,
        hidden_states: torch.Tensor,
        per_channel_scale: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # similar to the "explicit noise inputs" method in style-gan
        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype, generator=generator)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise

        return hidden_states

    def forward(
        self,
        input_tensor: torch.Tensor,
        causal: bool = True,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        shift1 = scale1 = shift2 = scale2 = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            ada_values = self.scale_shift_table[None, ..., None, None, None].to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) + timestep.reshape(
                batch_size,
                4,
                -1,
                timestep.shape[-3],
                timestep.shape[-2],
                timestep.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)

        dropout_is_noop = (
            type(self.dropout) is nn.Dropout
            and not _has_hooks(self.dropout)
            and (not self.training or self.dropout.p == 0.0)
        )
        can_fuse_stage2_base = (
            not self.inject_noise
            and dropout_is_noop
            and type(self.conv2) is CausalConv3d
            and _can_fuse_pixel_norm_silu(self.norm2, self.non_linearity, hidden_states)
        )

        bias1: torch.Tensor | None = None
        if isinstance(self.conv1, CausalConv3d) and _can_fuse_pixel_norm_silu_conv(
            self.norm1, self.non_linearity, self.conv1, hidden_states, causal, scale1, shift1
        ):
            pad1, s_mode1, conv_pad1 = self.conv1._fused_pad_spec(causal)
            x_pad1 = triton_pixel_norm_adaln_silu_pad_3d(
                hidden_states,
                scale1,
                shift1,
                self.norm1.eps,
                pad1,
                spatial_mode=s_mode1,
                temporal_mode="replicate",
                apply_silu=True,
            )
            hidden_states, bias1 = self.conv1._forward_conv_only(x_pad1, conv_pad1, defer_bias=can_fuse_stage2_base)
        else:
            if _can_fuse_pixel_norm_silu(self.norm1, self.non_linearity, hidden_states):
                hidden_states = triton_pixel_norm_adaln_silu(
                    hidden_states,
                    scale1,
                    shift1,
                    self.norm1.eps,
                    apply_silu=True,
                )
            else:
                hidden_states = self.norm1(hidden_states)
                if self.timestep_conditioning and scale1 is not None and shift1 is not None:
                    hidden_states = hidden_states * (1 + scale1) + shift1
                hidden_states = self.non_linearity(hidden_states)
            hidden_states = self.conv1(hidden_states, causal=causal)

        if self.inject_noise:
            if bias1 is not None:
                hidden_states = triton_ltx_residual_bias_add(None, hidden_states, bias1)
                bias1 = None
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale1.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        bias2: torch.Tensor | None = None
        if (
            dropout_is_noop
            and isinstance(self.conv2, CausalConv3d)
            and _can_fuse_pixel_norm_silu_conv(
                self.norm2,
                self.non_linearity,
                self.conv2,
                hidden_states,
                causal,
                scale2,
                shift2,
                pre_bias=bias1,
            )
        ):
            pad2, s_mode2, conv_pad2 = self.conv2._fused_pad_spec(causal)
            x_pad2 = triton_pixel_norm_adaln_silu_pad_3d(
                hidden_states,
                scale2,
                shift2,
                self.norm2.eps,
                pad2,
                spatial_mode=s_mode2,
                temporal_mode="replicate",
                apply_silu=True,
                pre_bias=bias1,
            )
            hidden_states, bias2 = self.conv2._forward_conv_only(x_pad2, conv_pad2, defer_bias=not self.inject_noise)
        else:
            if bias1 is not None:
                hidden_states = triton_ltx_residual_bias_add(None, hidden_states, bias1)
            if _can_fuse_pixel_norm_silu(self.norm2, self.non_linearity, hidden_states):
                hidden_states = triton_pixel_norm_adaln_silu(
                    hidden_states,
                    scale2,
                    shift2,
                    self.norm2.eps,
                    apply_silu=True,
                )
            else:
                hidden_states = self.norm2(hidden_states)
                if self.timestep_conditioning and scale2 is not None and shift2 is not None:
                    hidden_states = hidden_states * (1 + scale2) + shift2
                hidden_states = self.non_linearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            if bias2 is not None:
                hidden_states = triton_ltx_residual_bias_add(None, hidden_states, bias2)
                bias2 = None
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale2.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        input_tensor = self.norm3(input_tensor)
        input_tensor = self.conv_shortcut(input_tensor)

        if can_use_triton_ltx_residual_bias_add(input_tensor, hidden_states, bias2):
            return triton_ltx_residual_bias_add(input_tensor, hidden_states, bias2)
        if bias2 is not None:
            hidden_states = hidden_states + bias2.view(1, -1, 1, 1, 1).to(hidden_states)
        output_tensor = input_tensor + hidden_states

        return output_tensor


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.
    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        inject_noise (`bool`, *optional*, defaults to `False`):
            Whether to inject noise into the hidden states.
        timestep_conditioning (`bool`, *optional*, defaults to `False`):
            Whether to condition the hidden states on the timestep.
    Returns:
        `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.
    """

    def __init__(
        self,
        dims: int | tuple[int, int],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.GROUP_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=in_channels * 4, size_emb_dim=0
            )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        timestep_embed = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(batch_size, timestep_embed.shape[-1], 1, 1, 1)

        for resnet in self.res_blocks:
            hidden_states = resnet(
                hidden_states,
                causal=causal,
                timestep=timestep_embed,
                generator=generator,
            )

        return hidden_states
