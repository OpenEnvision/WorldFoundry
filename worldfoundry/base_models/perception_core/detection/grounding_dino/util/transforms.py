from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torchvision.transforms.functional as F

from worldfoundry.core.geometry.boxes import box_xyxy_to_cxcywh


def _resolve_size(
    image_size: tuple[int, int],
    size: int | Sequence[int],
    max_size: int | None,
) -> tuple[int, int]:
    if isinstance(size, Sequence):
        width, height = size
        return int(height), int(width)

    width, height = image_size
    target = int(size)
    if max_size is not None:
        min_original = float(min(width, height))
        max_original = float(max(width, height))
        if max_original / min_original * target > max_size:
            target = int(round(max_size * min_original / max_original))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.

    if (width <= height and width == target) or (height <= width and height == target):
        return height, width
    if width < height:
        return int(target * height / width), target
    return target, int(target * width / height)


def resize(image, target, size: int | Sequence[int], max_size: int | None = None):
    resolved_size = _resolve_size(image.size, size, max_size)
    resized = F.resize(image, resolved_size)
    if target is None:
        return resized, None

    target = target.copy()
    ratio_width, ratio_height = (float(new) / float(old) for new, old in zip(resized.size, image.size, strict=False))
    if "boxes" in target:
        target["boxes"] = target["boxes"] * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    if "area" in target:
        target["area"] = target["area"] * (ratio_width * ratio_height)
    target["size"] = torch.tensor(resolved_size)
    return resized, target


class RandomResize:
    def __init__(self, sizes: Sequence[int], max_size: int | None = None) -> None:
        self.sizes = tuple(sizes)
        self.max_size = max_size

    def __call__(self, image, target=None):
        return resize(image, target, random.choice(self.sizes), self.max_size)


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None

        target = target.copy()
        if "boxes" in target:
            height, width = image.shape[-2:]
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            target["boxes"] = boxes / torch.tensor(
                [width, height, width, height],
                dtype=torch.float32,
            )
        return image, target


class Compose:
    def __init__(self, transforms) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self) -> str:
        body = "\n".join(f"    {transform}" for transform in self.transforms)
        return f"{self.__class__.__name__}(\n{body}\n)"
