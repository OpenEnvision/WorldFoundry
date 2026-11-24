"""Panorama-to-Gaussian SHARP inference model."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dap_model import (
    DepthAnythingV2,
    make_dap_model,
)

from . import math_utils
from .color_space import srgb_to_linear
from .layers import (
    DAPFeatureAdapter,
    DAPFeatures,
    DirectPredictionHead,
    DPTFeatureHead,
)
from .types import (
    ColorInitOption,
    DepthInitOption,
    Gaussians3D,
    InitializerParams,
    PredictorParams,
)

LOGGER = logging.getLogger(__name__)


class PanoGaussianBaseValues(NamedTuple):
    dir_x: torch.Tensor
    dir_y: torch.Tensor
    dir_z: torch.Tensor
    inverse_depth: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class PanoInitializerOutput(NamedTuple):
    gaussian_base_values: PanoGaussianBaseValues
    feature_input: torch.Tensor
    global_scale: torch.Tensor | None = None


def _rescale_depth(
    depth: torch.Tensor, depth_min: float = 1.0, depth_max: float = 100.0
) -> tuple[torch.Tensor, torch.Tensor]:
    current_min = depth.flatten(depth.ndim - 3).min(dim=-1).values
    factor = depth_min / (current_min + 1e-6)
    return (depth * factor[..., None, None, None]).clamp(max=depth_max), factor


def _equirectangular_directions(
    depth: torch.Tensor, stride: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = depth.shape
    height //= stride
    width //= stride
    horizontal = torch.linspace(
        0.5 / width,
        1.0 - 0.5 / width,
        width,
        device=depth.device,
        dtype=depth.dtype,
    )
    vertical = torch.linspace(
        0.5 / height,
        1.0 - 0.5 / height,
        height,
        device=depth.device,
        dtype=depth.dtype,
    )
    vertical, horizontal = torch.meshgrid(vertical, horizontal, indexing="ij")
    theta = (horizontal - 0.5) * 2.0 * math.pi
    phi = (0.5 - vertical) * math.pi
    cos_phi = torch.cos(phi)
    shape = (batch_size, 1, num_layers, height, width)
    return (
        (cos_phi * torch.sin(theta))[None, None, None].expand(shape),
        (-torch.sin(phi))[None, None, None].expand(shape),
        (cos_phi * torch.cos(theta))[None, None, None].expand(shape),
    )


def _base_scale(
    disparity: torch.Tensor,
    factor: float,
    mode: str,
    growth_rate: float,
    transition_depth: float,
) -> torch.Tensor:
    height = disparity.shape[-2]
    vertical = torch.linspace(
        0.5 / height,
        1.0 - 0.5 / height,
        height,
        device=disparity.device,
        dtype=disparity.dtype,
    )
    cos_phi = torch.cos((0.5 - vertical) * math.pi)
    depth = torch.ones_like(disparity) / disparity
    if mode == "linear":
        scales = depth
    elif mode == "exp2lin":
        power = depth**growth_rate
        linear = transition_depth**growth_rate + depth - transition_depth
        scales = torch.where(depth <= transition_depth, power, linear)
    else:
        raise ValueError(f"Unknown SHARP scale mode: {mode!r}.")
    return scales * factor * cos_phi[None, None, None, :, None].clamp(min=0.1)


class PanoMultiLayerInitializer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        stride: int,
        base_depth: float,
        scale_factor: float,
        disparity_factor: float,
        color_option: ColorInitOption,
        first_layer_depth_option: DepthInitOption,
        rest_layer_depth_option: DepthInitOption,
        normalize_depth: bool,
        feature_input_stop_grad: bool,
        scale_mode: str,
        exp_growth_rate: float,
        transition_depth: float,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.color_option = color_option
        self.first_layer_depth_option = first_layer_depth_option
        self.rest_layer_depth_option = rest_layer_depth_option
        self.normalize_depth = normalize_depth
        self.feature_input_stop_grad = feature_input_stop_grad
        self.scale_mode = scale_mode
        self.exp_growth_rate = exp_growth_rate
        self.transition_depth = transition_depth

    def prepare_feature_input(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.feature_input_stop_grad:
            image = image.detach()
            depth = depth.detach()
        stable_depth, _ = _rescale_depth(depth + 1.0)
        features = torch.cat((image, self.disparity_factor / stable_depth), dim=1)
        return 2.0 * features - 1.0

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> PanoInitializerOutput:
        image = image.contiguous()
        depth = depth.contiguous()
        batch_size, _, image_height, image_width = depth.shape
        base_height = image_height // self.stride
        base_width = image_width // self.stride
        global_scale = None
        if self.normalize_depth:
            depth, factor = _rescale_depth(depth)
            global_scale = 1.0 / factor

        def fixed_disparity(count: int = 1) -> torch.Tensor:
            values = torch.linspace(
                1.0 / self.base_depth,
                0.0,
                count + 1,
                device=depth.device,
                dtype=depth.dtype,
            )
            return values[None, None, :-1, None, None].repeat(batch_size, 1, 1, base_height, base_width)

        def surface_disparity(value: torch.Tensor, mode: str) -> torch.Tensor:
            disparity = 1.0 / (value + 1e-6)
            if mode == "min":
                disparity = F.max_pool2d(disparity, self.stride, self.stride)
            elif mode == "max":
                disparity = -F.max_pool2d(-disparity, self.stride, self.stride)
            else:
                raise ValueError(f"Unknown depth pooling mode: {mode!r}.")
            return disparity[:, :, None]

        if self.first_layer_depth_option == "surface_min":
            first = surface_disparity(depth[:, :1], "min")
        elif self.first_layer_depth_option == "surface_max":
            first = surface_disparity(depth[:, :1], "max")
        elif self.first_layer_depth_option in {"base_depth", "linear_disparity"}:
            first = fixed_disparity()
        else:
            raise ValueError(f"Unknown first depth mode: {self.first_layer_depth_option!r}.")

        if self.num_layers == 1:
            disparity = first
        else:
            remaining_depth = depth if depth.shape[1] == 1 else depth[:, 1:]
            if self.rest_layer_depth_option == "surface_min":
                remaining = surface_disparity(remaining_depth, "min")
            elif self.rest_layer_depth_option == "surface_max":
                remaining = surface_disparity(remaining_depth, "max")
            elif self.rest_layer_depth_option == "base_depth":
                remaining = torch.cat([fixed_disparity() for _ in range(self.num_layers - 1)], dim=2)
            elif self.rest_layer_depth_option == "linear_disparity":
                remaining = fixed_disparity(self.num_layers - 1)
            else:
                raise ValueError(f"Unknown remaining depth mode: {self.rest_layer_depth_option!r}.")
            disparity = torch.cat((first, remaining), dim=2)

        dir_x, dir_y, dir_z = _equirectangular_directions(depth, self.stride, self.num_layers)
        scales = _base_scale(
            disparity,
            2.0 * self.scale_factor * self.stride / float(image_width),
            self.scale_mode,
            self.exp_growth_rate,
            self.transition_depth,
        )
        quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=depth.device, dtype=depth.dtype)[
            None, :, None, None, None
        ]
        opacities = torch.tensor(
            [min(1.0 / self.num_layers, 0.5)],
            device=depth.device,
            dtype=depth.dtype,
        )
        colors = torch.full(
            (batch_size, 3, self.num_layers, base_height, base_width),
            0.5,
            device=image.device,
            dtype=image.dtype,
        )
        if self.color_option == "first_layer":
            colors[:, :, 0] = F.avg_pool2d(image, self.stride, self.stride)
        elif self.color_option == "all_layers":
            pooled = F.avg_pool2d(image, self.stride, self.stride)
            colors = pooled[:, :, None].repeat(1, 1, self.num_layers, 1, 1)
        elif self.color_option != "none":
            raise ValueError(f"Unknown color initialization: {self.color_option!r}.")

        values = PanoGaussianBaseValues(
            dir_x=dir_x,
            dir_y=dir_y,
            dir_z=dir_z,
            inverse_depth=disparity,
            scales=scales,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        )
        return PanoInitializerOutput(
            gaussian_base_values=values,
            feature_input=self.prepare_feature_input(image, depth),
            global_scale=global_scale,
        )


def create_initializer(params: InitializerParams) -> PanoMultiLayerInitializer:
    return PanoMultiLayerInitializer(
        num_layers=params.num_layers,
        stride=params.stride,
        base_depth=params.base_depth,
        scale_factor=params.scale_factor,
        disparity_factor=params.disparity_factor,
        color_option=params.color_option,
        first_layer_depth_option=params.first_layer_depth_option,
        rest_layer_depth_option=params.rest_layer_depth_option,
        normalize_depth=params.normalize_depth,
        feature_input_stop_grad=params.feature_input_stop_grad,
        scale_mode=params.scale_mode,
        exp_growth_rate=params.exp_growth_rate,
        transition_depth=params.transition_depth,
    )


def _scale_activation_constants(max_scale: float, min_scale: float) -> tuple[float, float]:
    constant_a = (max_scale - min_scale) / (1.0 - min_scale) / (max_scale - 1.0)
    constant_b = math_utils.inverse_sigmoid(torch.tensor((1.0 - min_scale) / (max_scale - min_scale))).item()
    return constant_a, constant_b


class PanoGaussianComposer(nn.Module):
    def __init__(self, params: PredictorParams, scale_factor: int) -> None:
        super().__init__()
        self.delta_factor = params.delta_factor
        self.max_scale = params.max_scale
        self.min_scale = params.min_scale
        self.color_activation_type = params.color_activation_type
        self.opacity_activation_type = params.opacity_activation_type
        self.color_space = params.color_space
        self.scale_factor = scale_factor
        self.base_scale_on_predicted_mean = params.base_scale_on_predicted_mean

    def forward(
        self,
        delta: torch.Tensor,
        base_values: PanoGaussianBaseValues,
        global_scale: torch.Tensor | None = None,
    ) -> Gaussians3D:
        actual_scale = base_values.dir_x.shape[-1] // delta.shape[-1]
        if self.scale_factor != 1 and actual_scale != 1:
            batch, channels, layers, height, width = delta.shape
            delta = F.interpolate(
                delta.view(batch, channels * layers, height, width),
                scale_factor=self.scale_factor,
            ).view(
                batch,
                channels,
                layers,
                height * self.scale_factor,
                width * self.scale_factor,
            )

        inverse_depth = F.softplus(
            math_utils.inverse_softplus(base_values.inverse_depth) + self.delta_factor.z * delta[:, 2:3]
        )
        theta = torch.atan2(base_values.dir_x, base_values.dir_z)
        phi = -torch.asin(base_values.dir_y.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        theta = theta + self.delta_factor.xy * delta[:, 0:1]
        phi = (phi + self.delta_factor.xy * delta[:, 1:2]).clamp(-math.pi / 2.0 + 1e-5, math.pi / 2.0 - 1e-5)
        cos_phi = torch.cos(phi)
        directions = torch.cat(
            (cos_phi * torch.sin(theta), -torch.sin(phi), cos_phi * torch.cos(theta)),
            dim=1,
        )
        means = directions / (inverse_depth + 1e-3)

        if self.base_scale_on_predicted_mean:
            base_scales = base_values.scales * base_values.inverse_depth / (inverse_depth + 1e-3)
        else:
            base_scales = base_values.scales
        constant_a, constant_b = _scale_activation_constants(self.max_scale, self.min_scale)
        scale_multiplier = (self.max_scale - self.min_scale) * torch.sigmoid(
            constant_a * self.delta_factor.scale * delta[:, 3:6] + constant_b
        ) + self.min_scale
        scales = base_scales * scale_multiplier
        quaternions = base_values.quaternions + self.delta_factor.quaternion * delta[:, 6:10]

        color_base = base_values.colors
        if self.color_activation_type == "sigmoid":
            color_base = color_base.clamp(0.01, 0.99)
        elif self.color_activation_type in {"exp", "softplus"}:
            color_base = color_base.clamp_min(0.01)
        color_activation = math_utils.create_activation_pair(self.color_activation_type)
        colors = color_activation.forward(
            color_activation.inverse(color_base) + self.delta_factor.color * delta[:, 10:13]
        )
        if self.color_space == "linearRGB":
            colors = srgb_to_linear(colors)
        opacity_activation = math_utils.create_activation_pair(self.opacity_activation_type)
        opacities = opacity_activation.forward(
            opacity_activation.inverse(base_values.opacities) + self.delta_factor.opacity * delta[:, 13]
        )

        means = means.permute(0, 2, 3, 4, 1).flatten(1, 3)
        scales = scales.permute(0, 2, 3, 4, 1).flatten(1, 3)
        quaternions = quaternions.permute(0, 2, 3, 4, 1).flatten(1, 3)
        colors = colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
        opacities = opacities.flatten(1, 3)
        if global_scale is not None:
            means = global_scale[:, None, None] * means
            scales = global_scale[:, None, None] * scales
        return Gaussians3D(means, scales, quaternions, colors, opacities)


class PanoPredictorOutput(NamedTuple):
    gaussians: Gaussians3D
    delta_values: torch.Tensor
    predicted_depth: torch.Tensor | None = None


class PanoGaussianPredictor(nn.Module):
    def __init__(
        self,
        pano_depth_model: DAPFeatureAdapter,
        init_model: PanoMultiLayerInitializer,
        feature_model: DPTFeatureHead,
        prediction_head: DirectPredictionHead,
        gaussian_composer: PanoGaussianComposer,
    ) -> None:
        super().__init__()
        self.pano_depth_model = pano_depth_model
        self.init_model = init_model
        self.feature_model = feature_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer

    def forward(
        self,
        image: torch.Tensor,
        depth_gt: torch.Tensor | None = None,
        infer_depth: bool = False,
        dap_scale_factor: float = 1.0,
    ) -> PanoPredictorOutput:
        if not infer_depth and depth_gt is None:
            raise ValueError("depth_gt is required when infer_depth is false.")
        depth_output: DAPFeatures = self.pano_depth_model(image, infer_depth)
        predicted_depth = depth_output.depth
        if infer_depth:
            if predicted_depth is None:
                raise RuntimeError("Depth model did not return a depth map.")
            predicted_depth = predicted_depth * dap_scale_factor
            depth = predicted_depth + 1.0
        else:
            depth = depth_gt
        if depth is None:
            raise RuntimeError("SHARP inference requires a depth map.")
        initialized = self.init_model(image, depth)
        features = self.feature_model(
            initialized.feature_input,
            encodings=depth_output.encoder_features,
        )
        deltas = self.prediction_head(features)
        gaussians = self.gaussian_composer(
            delta=deltas,
            base_values=initialized.gaussian_base_values,
            global_scale=initialized.global_scale,
        )
        return PanoPredictorOutput(gaussians, deltas, predicted_depth)


_DAP_MODEL_TYPES = {"vits", "vitb", "vitl"}


def create_depth_model(encoder: str = "vitl", checkpoint_path: str | Path | None = None) -> DepthAnythingV2:
    if encoder not in _DAP_MODEL_TYPES:
        raise ValueError(f"Unsupported DAP model type {encoder!r}; expected one of {sorted(_DAP_MODEL_TYPES)}.")
    model = make_dap_model(
        midas_model_type=encoder,
        max_depth=1.0,
    ).core
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            LOGGER.warning("Depth checkpoint has %d missing keys.", len(missing))
        if unexpected:
            LOGGER.warning("Depth checkpoint has %d unexpected keys.", len(unexpected))
    return model


def create_panorama_predictor(
    params: PredictorParams,
    depth_model: DepthAnythingV2,
) -> PanoGaussianPredictor:
    if params.gaussian_decoder.stride < params.initializer.stride:
        raise ValueError("Gaussian decoder stride must be >= initializer stride.")
    if params.gaussian_decoder.stride % params.initializer.stride:
        raise ValueError("Gaussian decoder and initializer strides must be divisible.")
    scale_factor = params.gaussian_decoder.stride // params.initializer.stride
    adapter = DAPFeatureAdapter(
        depth_model,
        duplicate_depth_layer=params.initializer.num_layers == 2,
    )
    pretrained = depth_model.pretrained
    depth_head = depth_model.depth_head
    output_channels = [int(layer.out_channels) for layer in depth_head.projects]
    refinenet = depth_head.scratch.refinenet1
    dpt_features = int(refinenet.out_conv.in_channels)
    feature_model = DPTFeatureHead(
        in_channels=int(pretrained.embed_dim),
        feature_dim=params.gaussian_decoder.dim_out,
        input_channels=3 + params.initializer.num_layers,
        features=dpt_features,
        out_channels=output_channels,
        use_bn=False,
        use_clstoken=bool(depth_head.use_clstoken),
        patch_size=int(pretrained.patch_size),
        stride=params.gaussian_decoder.stride,
    )
    return PanoGaussianPredictor(
        pano_depth_model=adapter,
        init_model=create_initializer(params.initializer),
        feature_model=feature_model,
        prediction_head=DirectPredictionHead(
            params.gaussian_decoder.dim_out,
            params.initializer.num_layers,
        ),
        gaussian_composer=PanoGaussianComposer(params, scale_factor),
    )


def build_panorama_predictor(
    depth_checkpoint: str | Path | None,
    *,
    num_layers: int = 1,
    device: torch.device | str = "cpu",
) -> PanoGaussianPredictor:
    params = PredictorParams(
        initializer=InitializerParams(
            num_layers=num_layers,
            stride=2,
            scale_factor=1.0,
            disparity_factor=1.0,
            color_option="all_layers",
            first_layer_depth_option="surface_min",
            rest_layer_depth_option="surface_min",
            normalize_depth=False,
            feature_input_stop_grad=True,
        )
    )
    return create_panorama_predictor(params, create_depth_model(checkpoint_path=depth_checkpoint)).to(device)


def load_predictor_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        LOGGER.warning("SHARP checkpoint has %d missing keys.", len(missing))
    if unexpected:
        LOGGER.warning("SHARP checkpoint has %d unexpected keys.", len(unexpected))
