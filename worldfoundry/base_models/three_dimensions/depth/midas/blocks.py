import torch
import torch.nn as nn
from timm import create_model

from worldfoundry.base_models.three_dimensions.depth.midas.fusion import ResidualConvUnit as ResidualConvUnit_custom
from worldfoundry.base_models.three_dimensions.depth.midas.fusion import make_scratch as _make_scratch

from .backbones.beit import (
    _make_pretrained_beitb16_384,
    _make_pretrained_beitl16_384,
    _make_pretrained_beitl16_512,
)
from .backbones.beit import (
    forward_beit as forward_beit,
)
from .backbones.levit import (
    _make_pretrained_levit_384,
)
from .backbones.levit import (
    forward_levit as forward_levit,
)
from .backbones.swin import (
    _make_pretrained_swinl12_384,
)
from .backbones.swin2 import (
    _make_pretrained_swin2b24_384,
    _make_pretrained_swin2l24_384,
    _make_pretrained_swin2t16_256,
)
from .backbones.swin_common import (
    forward_swin as forward_swin,
)
from .backbones.vit import (
    _make_pretrained_vitb16_384,
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
)
from .backbones.vit import (
    forward_vit as forward_vit,
)


def _make_encoder(
    backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    in_features=[96, 256, 512, 1024],  # noqa: B006 - Keeps the upstream default object and signature.
):
    if backbone == "beitl16_512":
        pretrained = _make_pretrained_beitl16_512(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)  # BEiT_512-L (backbone)
    elif backbone == "beitl16_384":
        pretrained = _make_pretrained_beitl16_384(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)  # BEiT_384-L (backbone)
    elif backbone == "beitb16_384":
        pretrained = _make_pretrained_beitb16_384(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)  # BEiT_384-B (backbone)
    elif backbone == "swin2l24_384":
        pretrained = _make_pretrained_swin2l24_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch(
            [192, 384, 768, 1536], features, groups=groups, expand=expand
        )  # Swin2-L/12to24 (backbone)
    elif backbone == "swin2b24_384":
        pretrained = _make_pretrained_swin2b24_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch(
            [128, 256, 512, 1024], features, groups=groups, expand=expand
        )  # Swin2-B/12to24 (backbone)
    elif backbone == "swin2t16_256":
        pretrained = _make_pretrained_swin2t16_256(use_pretrained, hooks=hooks)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)  # Swin2-T/16 (backbone)
    elif backbone == "swinl12_384":
        pretrained = _make_pretrained_swinl12_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch([192, 384, 768, 1536], features, groups=groups, expand=expand)  # Swin-L/12 (backbone)
    elif backbone == "next_vit_large_6m":
        from .backbones.next_vit import _make_pretrained_next_vit_large_6m

        pretrained = _make_pretrained_next_vit_large_6m(hooks=hooks)
        scratch = _make_scratch(
            in_features, features, groups=groups, expand=expand
        )  # Next-ViT-L on ImageNet-1K-6M (backbone)
    elif backbone == "levit_384":
        pretrained = _make_pretrained_levit_384(use_pretrained, hooks=hooks)
        scratch = _make_scratch([384, 512, 768], features, groups=groups, expand=expand)  # LeViT 384 (backbone)
    elif backbone == "vitl16_384":
        pretrained = _make_pretrained_vitl16_384(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # ViT-L/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb_rn50_384":
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384(use_pretrained, hooks=hooks, use_readout=use_readout)
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)
    elif backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False  # noqa: B011 - Keeps the assertion semantics under -O.

    return pretrained, scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = create_model(
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    # Modern timm folds the stem activation into ``bn1``. Keep an identity
    # slot so official MiDaS checkpoint module indices remain unchanged.
    stem_activation = getattr(effnet, "act1", nn.Identity())
    pretrained.layer1 = nn.Sequential(effnet.conv_stem, effnet.bn1, stem_activation, *effnet.blocks[0:2])
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    from torchvision.models import resnext101_32x8d

    resnet = resnext101_32x8d(weights=None)
    # The MiDaS WSL backbone always initializes from these weights.
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
        progress=True,
    )
    resnet.load_state_dict(state_dict)
    return _make_resnet_backbone(resnet)


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output
