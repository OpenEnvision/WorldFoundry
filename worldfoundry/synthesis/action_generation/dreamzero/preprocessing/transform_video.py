"""Deterministic video preprocessing used by DreamZero inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
import torchvision.transforms.v2 as T
from einops import rearrange
from pydantic import Field, PrivateAttr, field_validator

from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import ModalityTransform

from .schema import DatasetMetadata


class VideoTransform(ModalityTransform):
    backend: str = Field(default="torchvision")
    _transform: Callable | None = PrivateAttr(default=None)
    _original_resolutions: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)
    _INTERPOLATION_MAP: ClassVar[dict[str, Any]] = {
        "nearest": T.InterpolationMode.NEAREST,
        "linear": T.InterpolationMode.BILINEAR,
        "cubic": T.InterpolationMode.BICUBIC,
        "lanczos4": T.InterpolationMode.LANCZOS,
        "nearest_exact": T.InterpolationMode.NEAREST_EXACT,
    }

    @property
    def original_resolutions(self) -> dict[str, tuple[int, int]]:
        return self._original_resolutions

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        if self.backend != "torchvision":
            raise ValueError("DreamZero inference supports the torchvision video backend")
        self._original_resolutions = {}
        for key in self.apply_to:
            split = key.split(".", 1)
            if len(split) != 2 or split[1] not in dataset_metadata.modalities.video:
                raise ValueError(f"Video key {key!r} is absent from checkpoint metadata")
            self._original_resolutions[key] = dataset_metadata.modalities.video[split[1]].resolution
        self._transform = self.get_transform()

    def check_input(self, data: dict[str, Any]) -> None:
        for key in self.apply_to:
            if key not in data:
                raise KeyError(f"Missing DreamZero video input: {key}")
            if not isinstance(data[key], (torch.Tensor, np.ndarray)) or data[key].ndim not in (4, 5):
                raise ValueError(f"Video {key} must be a 4D/5D tensor or array")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        transform = self._transform
        if transform is None:
            return data
        self.check_input(data)
        views = [data[key] for key in self.apply_to]
        num_views = len(views)
        is_batched = views[0].ndim == 5
        batch_size = views[0].shape[0] if is_batched else 1
        concatenate = torch.cat if isinstance(views[0], torch.Tensor) else np.concatenate
        joined = concatenate(views, 0)
        if is_batched:
            joined = rearrange(joined, "(v b) t ... -> (v b t) ...", v=num_views, b=batch_size)
        joined = transform(joined)
        if is_batched:
            split = rearrange(joined, "(v b t) ... -> v b t ...", v=num_views, b=batch_size)
        else:
            split = rearrange(joined, "(v t) ... -> v t ...", v=num_views)
        for key, view in zip(self.apply_to, split, strict=False):
            data[key] = view
        return data

    def get_transform(self) -> Callable | None:
        raise NotImplementedError


class VideoCrop(VideoTransform):
    height: int | None = None
    width: int | None = None
    scale: float
    mode: str = "center"

    def get_transform(self) -> Callable:
        if len(set(self.original_resolutions.values())) != 1:
            raise ValueError("All DreamZero camera inputs must share one resolution")
        if self.height is None:
            self.width, self.height = self.original_resolutions[self.apply_to[0]]
        if self.width is None:
            raise ValueError("VideoCrop width and height must be provided together")
        return T.CenterCrop((int(self.height * self.scale), int(self.width * self.scale)))


class VideoResize(VideoTransform):
    height: int
    width: int
    interpolation: str = "linear"
    antialias: bool = True

    @field_validator("interpolation")
    @classmethod
    def validate_interpolation(cls, value):
        if value not in cls._INTERPOLATION_MAP:
            raise ValueError(f"Unsupported interpolation mode: {value}")
        return value

    def get_transform(self) -> Callable:
        return T.Resize(
            (self.height, self.width),
            interpolation=self._INTERPOLATION_MAP[self.interpolation],
            antialias=self.antialias,
        )


class _InferenceNoOpVideoTransform(VideoTransform):
    def get_transform(self) -> None:
        return None


class VideoRandomErasing(_InferenceNoOpVideoTransform):
    probability: float = 0.0
    scale: tuple[float, float] = (0.02, 0.33)
    ratio: tuple[float, float] = (0.3, 3.3)
    value: str | tuple[float, float, float] = "random"


class VideoRandomRotation(_InferenceNoOpVideoTransform):
    degrees: float | tuple[float, float]
    interpolation: str = "linear"


class VideoHorizontalFlip(_InferenceNoOpVideoTransform):
    p: float


class VideoGrayscale(_InferenceNoOpVideoTransform):
    p: float


class VideoColorJitter(_InferenceNoOpVideoTransform):
    brightness: float | tuple[float, float]
    contrast: float | tuple[float, float]
    saturation: float | tuple[float, float]
    hue: float | tuple[float, float]


class VideoRandomGrayscale(_InferenceNoOpVideoTransform):
    p: float


class VideoRandomPosterize(_InferenceNoOpVideoTransform):
    bits: int
    p: float


class VideoFocusRect(_InferenceNoOpVideoTransform):
    rect_key: str = ""


class VideoToTensor(VideoTransform):
    output_on_cuda: bool = False

    def get_transform(self) -> Callable:
        return self.to_tensor

    def check_input(self, data: dict[str, Any]) -> None:
        for key in self.apply_to:
            value = data[key]
            if not isinstance(value, np.ndarray) or value.ndim not in (4, 5) or value.dtype != np.uint8:
                raise ValueError(f"Video {key} must be a uint8 NumPy array")

    def to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frames)
        if self.output_on_cuda:
            tensor = tensor.cuda()
        return tensor.to(torch.float32).div_(255.0).movedim(-1, -3)


class VideoToNumpy(VideoTransform):
    def get_transform(self) -> Callable:
        return self.to_numpy

    @staticmethod
    def to_numpy(frames: torch.Tensor) -> np.ndarray:
        return frames.movedim(-3, -1).mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy()


class VideoNormalize(VideoTransform):
    mean: list[float]
    std: list[float]

    def get_transform(self) -> Callable:
        return T.Normalize(mean=self.mean, std=self.std)
