try:
    import torch_npu

    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.npu_config import (
        npu_config,
    )
except BaseException:
    torch_npu = None
    npu_config = None

from collections import deque
from typing import Tuple as Tuple  # noqa: UP035 - Re-exported to modules that star-import this namespace.
from typing import Union as Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 - Re-exported under this name.
from torch.utils.checkpoint import checkpoint

from .block import Block
from .ops import cast_tuple, video_to_image


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int] = 3,
        stride: int | tuple[int] = 1,
        padding: str | int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    @video_to_image
    def forward(self, x):
        return super().forward(x)


class CausalConv3d(Block):
    def __init__(
        self, chan_in, chan_out, kernel_size: int | tuple[int, int, int], enable_cached=False, bias=True, **kwargs
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        self.conv = nn.Conv3d(chan_in, chan_out, self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)
        self.enable_cached = enable_cached

        self.is_first_chunk = True

        self.causal_cached = deque()
        self.cache_offset = 0

    def forward(self, x):
        if self.is_first_chunk:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
        else:
            first_frame_pad = self.causal_cached.popleft()

        x = torch.concatenate((first_frame_pad, x), dim=2)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0:
                    self.causal_cached.append(x[:, :, -(self.time_kernel_size - 1) // self.stride[0] :].clone())
                else:
                    self.causal_cached.append(
                        x[:, :, : -self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0] :].clone()
                    )
            else:
                self.causal_cached.append(x[:, :, 0:0, :, :].clone())
        elif self.enable_cached:
            self.causal_cached.append(x[:, :, 0:0, :, :].clone())

        x = self.conv(x)
        return x


class CausalConv3d_GC(CausalConv3d):
    def __init__(self, chan_in, chan_out, kernel_size: int | tuple[int], init_method="random", **kwargs):
        super().__init__(chan_in, chan_out, kernel_size, init_method, **kwargs)

    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))  # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return checkpoint(self.conv, x)
