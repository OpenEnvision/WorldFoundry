# SPDX-FileCopyrightText: 2026 OpenEnvision contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared image/video-text scoring interface for T2V-Metrics backends.

Adapted from https://github.com/linzhiqiu/t2v_metrics (Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from worldfoundry.base_models.perception_core.video_text.vqa_score.constants import HF_CACHE_DIR

_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv"})


def _concat_video(video: str, destination: Path, num_frames: int) -> str:
    """Use the backend's floor-step frame sampling and JPEG mosaic layout."""
    import cv2

    from worldfoundry.base_models.perception_core.video_text.vqa_score.models.vqascore_models.mm_utils import (
        concatenate_images_horizontal,
        extract_frames,
    )

    extract_frames(video, num_frames, str(destination))
    frames = [cv2.imread(str(path)) for path in sorted(destination.glob("frame_*.jpg"))]
    output = destination / "concat.jpg"
    cv2.imwrite(str(output), concatenate_images_horizontal(frames, dist_images=10))
    return str(output)


class Score(nn.Module):
    """Return an image/video-by-text matrix using a selected scoring backend."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        cache_dir: str = HF_CACHE_DIR,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model not in self.list_all_models():
            raise ValueError(f"unsupported score model: {model!r}")
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir, **kwargs)
        self.model_name = model

    def prepare_scoremodel(self, model: str, device: str, cache_dir: str, **kwargs: Any) -> Any:
        """Construct the selected backend; implemented by each score family."""
        raise NotImplementedError("Subclasses must implement prepare_scoremodel")

    def list_all_models(self) -> list[str]:
        """List the model identifiers supported by this score family."""
        raise NotImplementedError("Subclasses must implement list_all_models")

    def _prepare_visual(self, visual: str, stack: ExitStack, num_frames: int) -> str:
        """Prepare one visual, retaining temporary mosaics until its scores are computed."""
        if Path(visual).suffix.lower() not in _VIDEO_SUFFIXES:
            return visual
        if self.model.video_mode == "concat":
            temporary = stack.enter_context(TemporaryDirectory(prefix="worldfoundry-score-"))
            return _concat_video(visual, Path(temporary) / "frames", num_frames)
        if self.model.video_mode != "direct":
            raise ValueError(f"unsupported video_mode: {self.model.video_mode!r}")
        return visual

    def forward(
        self,
        images: str | Sequence[str] | None = None,
        texts: str | Sequence[str] | None = None,
        num_frames: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Score every visual-text pair, with temporary mosaics for image-only models."""
        if images is None or texts is None:
            raise ValueError("scoring requires images/videos and texts")
        visuals = [images] if isinstance(images, str) else list(images)
        prompts = [texts] if isinstance(texts, str) else list(texts)
        scores = torch.zeros(len(visuals), len(prompts), device=self.device)
        with ExitStack() as stack:
            for index, visual in enumerate(visuals):
                visual = self._prepare_visual(visual, stack, num_frames)
                scores[index] = self.model.forward([visual] * len(prompts), prompts, **kwargs)
        return scores

    def batch_forward(
        self,
        dataset: Sequence[Mapping[str, Sequence[str]]],
        batch_size: int = 16,
        num_frames: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Score fixed-size visual/text options, returning sample-by-visual-by-text scores."""
        media_type = "videos" if "videos" in dataset[0] else "images"
        num_visuals = len(dataset[0][media_type])
        num_texts = len(dataset[0]["texts"])
        scores = torch.zeros(len(dataset), num_visuals, num_texts, device=self.device)
        offset = 0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            if len(batch[media_type]) != num_visuals or len(batch["texts"]) != num_texts:
                raise ValueError("each sample must have the same number of visual and text options")
            count = len(batch[media_type][0])
            with ExitStack() as stack:
                for visual_index, visuals in enumerate(batch[media_type]):
                    prepared = [self._prepare_visual(visual, stack, num_frames) for visual in visuals]
                    for text_index, prompts in enumerate(batch["texts"]):
                        scores[offset : offset + count, visual_index, text_index] = torch.cat(
                            [
                                self.model.forward([visual], [prompt], **kwargs).reshape(-1)
                                for visual, prompt in zip(prepared, prompts)
                            ]
                        )
            offset += count
        return scores
