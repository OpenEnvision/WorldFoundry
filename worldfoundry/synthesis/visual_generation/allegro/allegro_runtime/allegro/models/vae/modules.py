import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.downsampling import Downsample2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D
from einops import rearrange


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, dropout=0.0, up_sample=False, down_sample=False, spa_stride=1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        spa_pad = int((spa_stride - 1) * 0.5)
        temp_pad = 0
        self.temp_pad = temp_pad

        if down_sample:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim),
                nn.SiLU(),
                nn.Conv3d(
                    in_dim, out_dim, (2, spa_stride, spa_stride), stride=(2, 1, 1), padding=(0, spa_pad, spa_pad)
                ),
            )
        elif up_sample:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim * 2, (1, spa_stride, spa_stride), padding=(0, spa_pad, spa_pad)),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim),
                nn.SiLU(),
                nn.Conv3d(in_dim, out_dim, (3, spa_stride, spa_stride), padding=(temp_pad, spa_pad, spa_pad)),
            )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, spa_stride, spa_stride), padding=(temp_pad, spa_pad, spa_pad)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, spa_stride, spa_stride), padding=(temp_pad, spa_pad, spa_pad)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv3d(out_dim, in_dim, (3, spa_stride, spa_stride), padding=(temp_pad, spa_pad, spa_pad)),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

        self.down_sample = down_sample
        self.up_sample = up_sample

    def forward(self, hidden_states):
        identity = hidden_states

        if self.down_sample:
            identity = identity[:, :, ::2]
        elif self.up_sample:
            hidden_states_new = torch.cat((hidden_states, hidden_states), dim=2)
            hidden_states_new[:, :, 0::2] = hidden_states
            hidden_states_new[:, :, 1::2] = hidden_states
            identity = hidden_states_new
            del hidden_states_new

        if self.down_sample or self.up_sample:
            hidden_states = self.conv1(hidden_states)
        else:
            hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
            hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
            hidden_states = self.conv1(hidden_states)

        if self.up_sample:
            hidden_states = rearrange(hidden_states, "b (d c) f h w -> b c (f d) h w", d=2)

        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        hidden_states = self.conv2(hidden_states)
        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        hidden_states = self.conv3(hidden_states)
        hidden_states = torch.cat((hidden_states[:, :, 0:1], hidden_states), dim=2)
        hidden_states = torch.cat((hidden_states, hidden_states[:, :, -1:]), dim=2)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        return hidden_states


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        add_temp_downsample=False,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_temp_downsample:
            self.temp_convs_down = TemporalConvBlock(
                out_channels, out_channels, dropout=0.1, down_sample=True, spa_stride=3
            )
        self.add_temp_downsample = add_temp_downsample

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)
        if self.downsamplers:
            for down_layer in self.downsamplers:
                down_layer.requires_grad_(True)

    def forward(self, hidden_states):
        bz = hidden_states.shape[0]

        for resnet, temp_conv in zip(self.resnets, self.temp_convs, strict=False):
            hidden_states = rearrange(hidden_states, "b c n h w -> (b n) c h w")
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = rearrange(hidden_states, "(b n) c h w -> b c n h w", b=bz)
            hidden_states = temp_conv(hidden_states)
        if self.add_temp_downsample:
            hidden_states = self.temp_convs_down(hidden_states)

        if self.downsamplers is not None:
            hidden_states = rearrange(hidden_states, "b c n h w -> (b n) c h w")
            for upsampler in self.downsamplers:
                hidden_states = upsampler(hidden_states)
            hidden_states = rearrange(hidden_states, "(b n) c h w -> b c n h w", b=bz)
        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        add_temp_upsample=False,
        temb_channels=None,
    ):
        super().__init__()
        self.add_upsample = add_upsample

        resnets = []
        temp_convs = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvBlock(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        self.add_temp_upsample = add_temp_upsample
        if add_temp_upsample:
            self.temp_conv_up = TemporalConvBlock(out_channels, out_channels, dropout=0.1, up_sample=True, spa_stride=3)

        if self.add_upsample:
            # self.upsamplers = nn.ModuleList([PSUpsample2D(out_channels, use_conv=True, use_pixel_shuffle=True, out_channels=out_channels)])
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)
        if self.add_upsample:
            self.upsamplers.requires_grad_(True)

    def forward(self, hidden_states):
        bz = hidden_states.shape[0]

        for resnet, temp_conv in zip(self.resnets, self.temp_convs, strict=False):
            hidden_states = rearrange(hidden_states, "b c n h w -> (b n) c h w")
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = rearrange(hidden_states, "(b n) c h w -> b c n h w", b=bz)
            hidden_states = temp_conv(hidden_states)
        if self.add_temp_upsample:
            hidden_states = self.temp_conv_up(hidden_states)

        if self.upsamplers is not None:
            hidden_states = rearrange(hidden_states, "b c n h w -> (b n) c h w")
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
            hidden_states = rearrange(hidden_states, "(b n) c h w -> b c n h w", b=bz)
        return hidden_states


class UNetMidBlock3DConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvBlock(
                in_channels,
                in_channels,
                dropout=0.1,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            temp_convs.append(
                TemporalConvBlock(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)

    def _set_partial_grad(self):
        for temp_conv in self.temp_convs:
            temp_conv.requires_grad_(True)

    def forward(
        self,
        hidden_states,
    ):
        bz = hidden_states.shape[0]
        hidden_states = rearrange(hidden_states, "b c n h w -> (b n) c h w")

        hidden_states = self.resnets[0](hidden_states, temb=None)
        hidden_states = rearrange(hidden_states, "(b n) c h w -> b c n h w", b=bz)
        hidden_states = self.temp_convs[0](hidden_states)
        hidden_states = rearrange(hidden_states, "b c n h w -> (b n) c h w")

        for attn, resnet, temp_conv in zip(self.attentions, self.resnets[1:], self.temp_convs[1:], strict=False):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = rearrange(hidden_states, "(b n) c h w -> b c n h w", b=bz)
            hidden_states = temp_conv(hidden_states)
        return hidden_states
