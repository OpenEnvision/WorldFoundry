# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot -> networks -> encoders -> resnet.py functionality."""

import math

import torch.nn as nn

from ...utils.learning import freeze_params


class Bottleneck(nn.Module):
    """Bottleneck implementation."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        """Init.

        Args:
            inplanes: The inplanes.
            planes: The planes.
            stride: The stride.
            dilation: The dilation.
            downsample: The downsample.
            BatchNorm: The batchnorm.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Res net implementation."""
    def __init__(self, block, layers, output_stride, BatchNorm, freeze_at=0):
        """Init.

        Args:
            block: The block.
            layers: The layers.
            output_stride: The output stride.
            BatchNorm: The batchnorm.
            freeze_at: The freeze at.
        """
        self.inplanes = 64
        super(ResNet, self).__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=strides[0],
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=strides[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=strides[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.stem = [self.conv1, self.bn1]
        self.stages = [self.layer1, self.layer2, self.layer3]

        self._init_weight()
        self.freeze(freeze_at)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        """Helper function to make layer.

        Args:
            block: The block.
            planes: The planes.
            blocks: The blocks.
            stride: The stride.
            dilation: The dilation.
            BatchNorm: The batchnorm.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                max(dilation // 2, 1),
                downsample,
                BatchNorm,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        """Forward.

        Args:
            input: The input.
        """
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xs = []

        x = self.layer1(x)
        xs.append(x)  # 4X
        x = self.layer2(x)
        xs.append(x)  # 8X
        x = self.layer3(x)
        xs.append(x)  # 16X
        # Following STMVOS, we drop stage 5.
        xs.append(x)  # 16X

        return xs

    def _init_weight(self):
        """Helper function to init weight."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze(self, freeze_at):
        """Freeze.

        Args:
            freeze_at: The freeze at.
        """
        if freeze_at >= 1:
            for m in self.stem:
                freeze_params(m)

        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)


def ResNet50(output_stride, BatchNorm, freeze_at=0):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, freeze_at=freeze_at)
    return model


def ResNet101(output_stride, BatchNorm, freeze_at=0):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, freeze_at=freeze_at)
    return model


if __name__ == "__main__":
    import torch

    model = ResNet101(BatchNorm=nn.BatchNorm2d, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
