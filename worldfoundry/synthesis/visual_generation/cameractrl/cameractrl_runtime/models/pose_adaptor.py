import math

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.encoders.lvdm.adapter import Downsample
from worldfoundry.core.nn.blocks.module_properties import get_parameter_dtype as get_parameter_dtype

from .motion_module import TemporalTransformerBlock


class PoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]  # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)  # bf c h w
        pose_embedding_features = [rearrange(x, "(b f) c h w -> b c f h w", b=bs) for x in pose_embedding_features]
        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states, pose_embedding_features=pose_embedding_features
        ).sample
        return noise_pred


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2, ...] = torch.sin(position * div_term)
        pe[0, :, 1::2, ...] = torch.cos(position * div_term)
        pe.unsqueeze_(-1).unsqueeze_(-1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), ...]
        return self.dropout(x)


class CameraPoseEncoder(nn.Module):
    def __init__(
        self,
        downscale_factor,
        channels=[320, 640, 1280, 1280],  # noqa: B006 - Keeps the upstream default object and signature.
        nums_rb=3,
        cin=64,
        ksize=3,
        sk=False,
        use_conv=True,
        compression_factor=1,
        temporal_attention_nhead=8,
        attention_block_types=("Temporal_Self",),
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=16,
        rescale_output_factor=1.0,
    ):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                temporal_attention_layer = TemporalTransformerBlock(
                    dim=out_dim,
                    num_attention_heads=temporal_attention_nhead,
                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                    attention_block_types=attention_block_types,
                    dropout=0.0,
                    cross_attention_dim=None,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        bs = x.shape[0]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block in zip(
            self.encoder_down_conv_blocks, self.encoder_down_attention_blocks, strict=False
        ):
            for res_layer, attention_layer in zip(res_block, attention_block, strict=False):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, "(b f) c h w -> (b h w) f c", b=bs)
                x = attention_layer(x)
                x = rearrange(x, "(b h w) f c -> (b f) c h w", h=h, w=w)
            features.append(x)
        return features
