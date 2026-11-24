"""Neural layers unique to SHARP panorama inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dap_model import (
    DepthAnythingV2,
)
from worldfoundry.base_models.three_dimensions.depth.midas.fusion import make_fusion_block as _make_fusion_block
from worldfoundry.base_models.three_dimensions.depth.midas.fusion import make_scratch as _make_scratch

from .types import ImageFeatures


class ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module) -> None:
        super().__init__()
        self.residual = residual
        self.shortcut = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.residual(value)


class FeatureFusionBlock2d(nn.Module):
    """Fuse an input feature map with one skip feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()

        def residual() -> ResidualBlock:
            return ResidualBlock(
                nn.Sequential(
                    nn.ReLU(False),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.ReLU(False),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )
            )

        self.resnet1 = residual()
        self.resnet2 = residual()
        self.deconv = nn.Sequential()
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, value: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            value = self.skip_add.add(value, self.resnet1(skip))
        return self.out_conv(self.deconv(self.resnet2(value)))


class DAPFeatures(NamedTuple):
    depth: torch.Tensor | None
    mask: torch.Tensor | None
    encoder_features: list[torch.Tensor]


class DAPFeatureAdapter(nn.Module):
    """Expose shared DAP features in the form consumed by SHARP."""

    def __init__(
        self,
        depth_model: DepthAnythingV2,
        duplicate_depth_layer: bool = False,
    ) -> None:
        super().__init__()
        self.model = depth_model
        self.duplicate_depth_layer = duplicate_depth_layer
        if duplicate_depth_layer:
            self._duplicate_final_conv(self.model.depth_head)

    @staticmethod
    def _duplicate_final_conv(depth_head: nn.Module) -> None:
        output = depth_head.scratch.output_conv2
        index = next(
            (item for item in reversed(range(len(output))) if isinstance(output[item], nn.Conv2d)),
            None,
        )
        if index is None:
            raise RuntimeError("Depth head does not contain a final Conv2d layer.")
        original = output[index]
        replacement = nn.Conv2d(
            original.in_channels,
            2,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight.expand_as(replacement.weight))
            if original.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(original.bias.expand_as(replacement.bias))
        output[index] = replacement

    def forward(self, image: torch.Tensor, infer_depth: bool = True) -> DAPFeatures:
        encoder_features = self.model.extract_features(image)
        if not infer_depth:
            return DAPFeatures(None, None, encoder_features)

        depth, mask = self.model.decode_features(
            encoder_features,
            image.shape[-2:],
        )
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        return DAPFeatures(depth, mask, encoder_features)


class DPTFeatureHead(nn.Module):
    """Decode shared Depth Anything features into SHARP geometry/texture features."""

    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        input_channels: int = 4,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        use_bn: bool = False,
        use_clstoken: bool = False,
        patch_size: int = 16,
        stride: int = 2,
    ) -> None:
        super().__init__()
        out_channels = list(out_channels)
        self.use_clstoken = use_clstoken
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.stride = stride
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, channels, 1) for channels in out_channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
            ]
        )
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()) for _ in self.projects]
            )

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        half_features = features // 2
        self.output_conv1 = nn.Conv2d(features, half_features, 3, padding=1)
        self.input_encoder = nn.Sequential(
            nn.Conv2d(
                input_channels,
                half_features // 2,
                kernel_size=stride,
                stride=stride,
            ),
            nn.ReLU(),
            nn.Conv2d(half_features // 2, half_features, 3, padding=1),
            nn.ReLU(),
        )
        self.input_fusion = FeatureFusionBlock2d(half_features)

        def output_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(half_features, half_features, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(half_features, feature_dim, 1),
                nn.ReLU(),
            )

        self.texture_head = output_head()
        self.geometry_head = output_head()

    def _process_encoder_features(
        self,
        encoder_features: list[torch.Tensor],
        patch_height: int,
        patch_width: int,
    ) -> list[torch.Tensor]:
        output = []
        for index, feature_map in enumerate(encoder_features):
            if feature_map.ndim == 3:
                feature_map = feature_map.permute(0, 2, 1).reshape(
                    feature_map.shape[0],
                    feature_map.shape[-1],
                    patch_height,
                    patch_width,
                )
            elif feature_map.shape[-2:] != (patch_height, patch_width):
                feature_map = F.interpolate(
                    feature_map,
                    size=(patch_height, patch_width),
                    mode="bilinear",
                    align_corners=True,
                )
            output.append(self.resize_layers[index](self.projects[index](feature_map)))
        return output

    def forward(
        self,
        input_features: torch.Tensor,
        encodings: list[torch.Tensor],
    ) -> ImageFeatures:
        patch_height, patch_width = encodings[0].shape[-2:]
        layer_1, layer_2, layer_3, layer_4 = self._process_encoder_features(encodings, patch_height, patch_width)
        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer2_rn(layer_2)
        layer_3 = self.scratch.layer3_rn(layer_3)
        layer_4 = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        shared = self.output_conv1(self.scratch.refinenet1(path_2, layer_1))
        shared = F.interpolate(
            shared,
            size=(
                input_features.shape[-2] // self.stride,
                input_features.shape[-1] // self.stride,
            ),
            mode="bilinear",
            align_corners=True,
        )
        fused = self.input_fusion(shared, self.input_encoder(input_features))
        return ImageFeatures(
            texture_features=self.texture_head(fused),
            geometry_features=self.geometry_head(fused),
        )


class DirectPredictionHead(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.geometry_prediction_head = nn.Conv2d(feature_dim, 3 * num_layers, 1)
        self.texture_prediction_head = nn.Conv2d(feature_dim, 11 * num_layers, 1)
        nn.init.zeros_(self.geometry_prediction_head.weight)
        nn.init.zeros_(self.geometry_prediction_head.bias)
        nn.init.zeros_(self.texture_prediction_head.weight)
        nn.init.zeros_(self.texture_prediction_head.bias)

    def forward(self, features: ImageFeatures) -> torch.Tensor:
        geometry = self.geometry_prediction_head(features.geometry_features).unflatten(1, (3, self.num_layers))
        texture = self.texture_prediction_head(features.texture_features).unflatten(1, (11, self.num_layers))
        return torch.cat((geometry, texture), dim=1)
