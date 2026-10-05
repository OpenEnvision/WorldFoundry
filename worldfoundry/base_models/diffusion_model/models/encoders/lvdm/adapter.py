"""Adapter that exposes LVDM OpenCLIP as a Diffusers-shaped encoder.

:class:`Adapter` plus residual / downsample helpers let
legacy VideoCrafter CLIP towers look like a Diffusers text
encoder to loaders that expect that API.

Token layout stays OpenCLIP ``context`` (B, L, C).
The native graph is :mod:`.condition`.

Compatibility wrapper only — T2V-Turbo uses
:class:`~..t2v_turbo.T2VTurboConditioner`.
"""

import torch.nn as nn

from worldfoundry.core.nn.diffusion.convolution import avg_pool_nd, conv_nd


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        """Init.

        Args:
            channels: The channels.
            use_conv: The use conv.
            dims: The dims.
            out_channels: The out channels.
            padding: The padding.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    """Resnet block implementation."""
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        """Init.

        Args:
            in_c: The in c.
            out_c: The out c.
            down: The down.
            ksize: The ksize.
            sk: The sk.
            use_conv: The use conv.
        """
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            # self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps) # edit by zhouxiawang
            self.skep = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        if self.down == True:
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


class Adapter(nn.Module):
    """Adapter implementation."""
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        """Init.

        Args:
            channels: The channels.
            nums_rb: The nums rb.
            cin: The cin.
            ksize: The ksize.
            sk: The sk.
            use_conv: The use conv.
        """
        super(Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features
