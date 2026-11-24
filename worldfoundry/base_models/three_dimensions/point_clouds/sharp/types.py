"""Small data structures used by SHARP panorama inference."""

from __future__ import annotations

import dataclasses
from typing import Literal, NamedTuple

import torch

ActivationType = Literal["linear", "exp", "sigmoid", "softplus"]
ColorSpace = Literal["sRGB", "linearRGB"]
ColorInitOption = Literal["none", "first_layer", "all_layers"]
DepthInitOption = Literal[
    "surface_min",
    "surface_max",
    "base_depth",
    "linear_disparity",
]


class ImageFeatures(NamedTuple):
    texture_features: torch.Tensor
    geometry_features: torch.Tensor


class Gaussians3D(NamedTuple):
    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

    def to(self, device: torch.device | str) -> Gaussians3D:
        return Gaussians3D(*(value.to(device) for value in self))


@dataclasses.dataclass
class DeltaFactor:
    xy: float = 0.001
    z: float = 0.001
    color: float = 0.1
    opacity: float = 1.0
    scale: float = 1.0
    quaternion: float = 1.0


@dataclasses.dataclass
class InitializerParams:
    scale_factor: float = 1.0
    disparity_factor: float = 1.0
    stride: int = 2
    num_layers: int = 2
    first_layer_depth_option: DepthInitOption = "surface_min"
    rest_layer_depth_option: DepthInitOption = "surface_min"
    color_option: ColorInitOption = "all_layers"
    base_depth: float = 10.0
    feature_input_stop_grad: bool = True
    normalize_depth: bool = True
    scale_mode: Literal["linear", "exp2lin"] = "linear"
    exp_growth_rate: float = 1.3
    transition_depth: float = 100.0


@dataclasses.dataclass
class GaussianDecoderParams:
    dim_out: int = 32
    stride: int = 2


@dataclasses.dataclass
class PredictorParams:
    initializer: InitializerParams = dataclasses.field(default_factory=InitializerParams)
    gaussian_decoder: GaussianDecoderParams = dataclasses.field(default_factory=GaussianDecoderParams)
    delta_factor: DeltaFactor = dataclasses.field(default_factory=DeltaFactor)
    max_scale: float = 10.0
    min_scale: float = 0.0
    color_activation_type: ActivationType = "sigmoid"
    opacity_activation_type: ActivationType = "sigmoid"
    color_space: ColorSpace = "linearRGB"
    base_scale_on_predicted_mean: bool = True
