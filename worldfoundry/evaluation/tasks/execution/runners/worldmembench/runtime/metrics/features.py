# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Masked DINOv2 appearance features shared by human and object protocols."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from .human.target_association import normalized_embedding


def get_dino_model(context: EvaluationContext, checkpoint: str | Path) -> Any:
    """Share a local CLS backbone while keeping each protocol's preprocessing separate."""
    from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import load_dinov2_feature_model

    checkpoint = Path(checkpoint).expanduser().resolve()
    return context.get_resource(
        ("dinov2-cls", checkpoint, context.device),
        lambda: load_dinov2_feature_model(checkpoint, device=context.device),
    )


class MaskedDinoV2Features:
    """Batched masked-crop DINOv2-Large CLS embeddings."""

    def __init__(
        self,
        *,
        model: Any,
        device: str,
        input_size: int = 518,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("DINO batch_size must be positive")
        try:
            import torch
            import torchvision.transforms as transforms
        except ImportError as error:
            raise EvaluationError(
                "Masked DINOv2 features require torch and torchvision; install the WorldMemBench metric requirements"
            ) from error
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.device = device
        self.torch = torch
        self.effective_batch_size = batch_size

    @staticmethod
    def _masked_crop(
        frame_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        if mask.shape != frame_rgb.shape[:2] or not mask.any():
            return None
        y_coordinates, x_coordinates = np.where(mask)
        x1, x2 = int(x_coordinates.min()), int(x_coordinates.max()) + 1
        y1, y2 = int(y_coordinates.min()), int(y_coordinates.max()) + 1
        crop = frame_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0
        return crop

    @staticmethod
    def _is_out_of_memory(error: RuntimeError) -> bool:
        return "out of memory" in str(error).lower()

    def embed_batch(
        self,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray | None]:
        results: list[np.ndarray | None] = [None] * len(samples)
        position = 0
        while position < len(samples):
            chunk_size = min(
                self.effective_batch_size,
                len(samples) - position,
            )
            chunk = samples[position : position + chunk_size]
            tensors = []
            result_indices = []
            for offset, (frame_rgb, mask) in enumerate(chunk):
                crop = self._masked_crop(frame_rgb, mask)
                if crop is None:
                    continue
                tensors.append(self.transform(crop))
                result_indices.append(position + offset)
            if not tensors:
                position += chunk_size
                continue
            try:
                tensor_batch = self.torch.stack(tensors).to(self.device)
                with self.torch.inference_mode():
                    vectors = self.model(tensor_batch)
                vectors = vectors.detach().float().cpu().numpy()
            except RuntimeError as error:
                if not self._is_out_of_memory(error) or self.effective_batch_size == 1:
                    raise
                self.effective_batch_size = max(
                    1,
                    self.effective_batch_size // 2,
                )
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                continue
            for result_index, vector in zip(result_indices, vectors):
                results[result_index] = normalized_embedding(vector)
            position += chunk_size
        return results

    def embed(self, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Extract one masked crop with the same preprocessing as a batch."""
        return self.embed_batch([(frame_rgb, mask)])[0]
