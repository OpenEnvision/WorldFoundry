# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""WonderWorld single-image and camera-path normalization, before loading."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base_operator import BaseOperator


class WonderWorldOperator(BaseOperator):
    """Normalize one RGB seed and explicit novel-view camera matrices."""

    def __init__(self) -> None:
        super().__init__(operation_types=["textual_instruction", "visual_instruction", "action_instruction"])

    @staticmethod
    def camera_matrices(camera_path: Any = None, interactions: Any = None) -> np.ndarray:
        """Return [N,4,4] OpenCV world-to-camera transforms in splat units.

        camera_path accepts an array or JSON file of row-major matrices.
        The named camera_view_matrix_sequence input instead accepts upstream
        viewer column-major 16-vectors; its initial diag(-1,-1,1,1) is identity
        in WorldFoundry's OpenCV world frame.
        """
        values = camera_path if camera_path is not None else interactions
        if values is None:
            return np.empty((0, 4, 4), dtype=np.float32)
        if isinstance(values, (str, Path)):
            values = json.loads(Path(values).expanduser().read_text(encoding="utf-8"))
        viewer = False
        if isinstance(values, Mapping):
            if "camera_view_matrix_sequence" in values:
                values = values["camera_view_matrix_sequence"]
                viewer = True
            else:
                values = values["world_to_camera"]
        if isinstance(values, Sequence) and len(values) and isinstance(values[0], Mapping):
            viewer = True
            values = [item["view_matrix"] for item in values]
        matrices = np.asarray(values, dtype=np.float32)
        if not matrices.size:
            return np.empty((0, 4, 4), dtype=np.float32)
        if viewer:
            if matrices.ndim == 1:
                matrices = matrices[None]
            if matrices.ndim != 2 or matrices.shape[1] != 16:
                raise ValueError("WonderWorld viewer cameras must be column-major 16-vectors")
            matrices = matrices.reshape(-1, 4, 4).transpose(0, 2, 1) @ np.diag([-1, -1, 1, 1]).astype(np.float32)
        elif matrices.shape == (4, 4):
            matrices = matrices[None]
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
            raise ValueError("WonderWorld camera_path must contain finite [N,4,4] transforms")
        rotation = matrices[:, :3, :3]
        if (
            not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)
        ):
            raise ValueError("WonderWorld camera_path must contain rigid OpenCV world-to-camera transforms")
        return matrices.copy()

    def prepare(
        self,
        *,
        images: Any = None,
        input_path: str | Path | None = None,
        video: Any = None,
        camera_path: Any = None,
        interactions: Any = None,
    ) -> tuple[Image.Image, np.ndarray]:
        """Validate the complete observation before initializing any checkpoint."""
        if video is not None:
            raise ValueError("WonderWorld accepts one initial image, not video")
        source = images if images is not None else input_path
        if isinstance(source, (list, tuple)):
            if len(source) != 1:
                raise ValueError("WonderWorld requires exactly one initial image")
            source = source[0]
        if source is None:
            raise ValueError("WonderWorld requires an initial RGB image; prompt-only generation is unsupported")
        if isinstance(source, (str, Path)):
            with Image.open(Path(source).expanduser()) as image:
                image = image.convert("RGB")
        elif isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            array = np.asarray(source)
            if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                raise ValueError("WonderWorld array images must be uint8 HWC RGB")
            image = Image.fromarray(array)
        return image.resize((512, 512), Image.Resampling.BICUBIC), self.camera_matrices(camera_path, interactions)
