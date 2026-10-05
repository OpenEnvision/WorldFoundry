# Adapted from Tencent Hunyuan-GameCraft-1.0, under the Tencent Hunyuan Community License.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .patch import PatchEmbed


class CameraNet(ModelMixin):
    """
    Camera state encoding network that processes camera parameters into feature embeddings.

    This network converts camera state information into suitable feature representations
    for video generation models through downsampling, convolutional encoding, and
    temporal dimension compression. Supports loading from pretrained weights.
    """

    def __init__(
        self,
        in_channels,
        downscale_coef,
        out_channels,
        patch_size,
        hidden_size,
    ):
        super().__init__()
        # Calculate initial channels: PixelUnshuffle moves spatial info to channel dimension
        # resulting in channels = in_channels * (downscale_coef^2)
        start_channels = in_channels * (downscale_coef**2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]
        self.input_channels = input_channels
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)

        self.encode_first = nn.Sequential(
            nn.Conv2d(input_channels[0], input_channels[1], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[1]),
            nn.ReLU(),
        )
        self._initialize_weights(self.encode_first)
        self.encode_second = nn.Sequential(
            nn.Conv2d(input_channels[1], input_channels[2], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(2, input_channels[2]),
            nn.ReLU(),
        )
        self._initialize_weights(self.encode_second)

        self.final_proj = nn.Conv2d(input_channels[2], out_channels, kernel_size=1)
        self.zeros_init_linear(self.final_proj)

        self.scale = nn.Parameter(torch.ones(1))

        self.camera_in = PatchEmbed(patch_size=patch_size, in_chans=out_channels, embed_dim=hidden_size)

    def zeros_init_linear(self, linear: nn.Module):
        """
        Zero-initializes weights and biases of linear or convolutional layers.

        Args:
            linear (nn.Module): Linear or convolutional layer to initialize
        """
        if isinstance(linear, (nn.Linear, nn.Conv2d)):
            if hasattr(linear, "weight"):
                nn.init.zeros_(linear.weight)
            if hasattr(linear, "bias"):
                nn.init.zeros_(linear.bias)

    def _initialize_weights(self, block):
        """
        Initializes convolutional layer weights using He initialization,
        with biases initialized to zero.

        Args:
            block (nn.Sequential): Sequential block containing convolutional layers
        """
        for m in block:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2.0 / n))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def compress_time(self, x, num_frames):
        """
        Temporal dimension compression: reduces number of frames using average pooling
        while preserving key temporal information.

        Handling logic:
        - Special frame counts (66 or 34): split into two segments, keep first frame of each
          segment then pool remaining frames
        - Odd frame counts: keep first frame, pool remaining frames
        - Even frame counts: directly pool all frames

        Args:
            x (torch.Tensor): Input tensor with shape (b*f, c, h, w)
            num_frames (int): Number of frames in temporal dimension

        Returns:
            torch.Tensor: Temporally compressed tensor with shape (b*f', c, h, w) where f' < f
        """
        # Reshape: (b*f, c, h, w) -> (b, f, c, h, w)
        x = rearrange(x, "(b f) c h w -> b f c h w", f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, "b f c h w -> (b h w) c f")

        # print(x.shape)
        # raise Exception
        # Handle special frame counts (66 or 34)
        if x.shape[-1] == 66 or x.shape[-1] == 34:
            x_len = x.shape[-1]
            # Process first segment: keep first frame, pool remaining
            x_clip1 = x[..., : x_len // 2]
            x_clip1_first, x_clip1_rest = x_clip1[..., 0].unsqueeze(-1), x_clip1[..., 1:]
            x_clip1_rest = F.avg_pool1d(x_clip1_rest, kernel_size=2, stride=2)

            # Process second segment: keep first frame, pool remaining
            x_clip2 = x[..., x_len // 2 : x_len]
            x_clip2_first, x_clip2_rest = x_clip2[..., 0].unsqueeze(-1), x_clip2[..., 1:]
            x_clip2_rest = F.avg_pool1d(x_clip2_rest, kernel_size=2, stride=2)

            # Concatenate results from both segments
            x = torch.cat([x_clip1_first, x_clip1_rest, x_clip2_first, x_clip2_rest], dim=-1)

        elif x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = rearrange(x, "(b h w) c f -> (b f) c h w", b=batch_size, h=height, w=width)
        return x

    def forward(
        self,
        camera_states: torch.Tensor,
    ):
        """
        Forward pass: encodes camera states into feature embeddings.

        Args:
            camera_states (torch.Tensor): Camera state tensor with dimensions
                (batch, frames, channels, height, width)

        Returns:
            torch.Tensor: Encoded feature embeddings after patch embedding and scaling
        """
        # import pdb;pdb.set_trace()
        batch_size, num_frames, channels, height, width = camera_states.shape
        camera_states = rearrange(camera_states, "b f c h w -> (b f) c h w")
        camera_states = self.unshuffle(camera_states)
        camera_states = self.encode_first(camera_states)
        camera_states = self.compress_time(camera_states, num_frames=num_frames)
        num_frames = camera_states.shape[0] // batch_size
        camera_states = self.encode_second(camera_states)
        camera_states = self.compress_time(camera_states, num_frames=num_frames)
        # camera_states = rearrange(camera_states, '(b f) c h w -> b f c h w', b=batch_size)
        camera_states = self.final_proj(camera_states)
        camera_states = rearrange(camera_states, "(b f) c h w -> b c f h w", b=batch_size)
        camera_states = self.camera_in(camera_states)
        return camera_states * self.scale
