# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Image batching and PyIQA inference shared by WorldScore and 4DWorldBench.

Adapted from the WorldScore and 4DWorldBench image metric bases.
The 512-square resize is part of these benchmarks' protocol, not a default
preprocessing policy for other IQA metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

from worldfoundry.evaluation.tasks.metrics._shared.images import load_rgb_image

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image


class ImageQualityMetric(ABC):
    """Batch images using the WorldScore/4DWorldBench 512-square transform."""

    def __init__(self, *, device: Any, image_loader: Callable[[str], Image.Image] = load_rgb_image) -> None:
        self._device = device
        self._image_loader = image_loader
        self._metric: Any = None

    def _process_image(self, rendered_images: Sequence[str | Image.Image]) -> torch.Tensor:
        import torch
        from torchvision import transforms

        preprocessing = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        images = [
            preprocessing(self._image_loader(image) if isinstance(image, str) else image) for image in rendered_images
        ]
        return torch.stack(images).to(self._device)

    def _process_images(
        self,
        rendered_images: Sequence[str | Image.Image],
        reference_image: str | Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self._process_image([reference_image])
        rendered = self._process_image(rendered_images)
        return rendered, reference.repeat(len(rendered_images), 1, 1, 1)

    def _process_np_to_tensor(
        self, rendered_image: np.ndarray, reference_image: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torchvision.transforms import ToTensor

        return (
            ToTensor()(rendered_image).unsqueeze(0).to(self._device),
            ToTensor()(reference_image).unsqueeze(0).to(self._device),
        )

    @abstractmethod
    def _compute_scores(self, *args: Any) -> Any:
        """Compute the benchmark's metric-specific score."""


class PyIQAMetric(ImageQualityMetric):
    """Load the explicitly selected PyIQA variant without changing its defaults."""

    def __init__(
        self, metric_name: str, *, device: Any, image_loader: Callable[[str], Image.Image] = load_rgb_image
    ) -> None:
        super().__init__(device=device, image_loader=image_loader)
        self._metric = self._create_metric(metric_name).to(self._device)

    def _create_metric(self, metric: str) -> Any:
        return import_module("pyiqa").create_metric(metric)
