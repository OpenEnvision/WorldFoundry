"""GroundingDINO + SAM2 video segmentation composition."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


class GroundedSAM2VideoSegmenter:
    """Detect requested phrases on frame zero and propagate each box with SAM2."""

    def __init__(
        self,
        *,
        grounding_config: str | Path | None = None,
        grounding_checkpoint: str | Path | None = None,
        sam2_checkpoint: str | Path | None = None,
        sam2_model_id: str = "facebook/sam2.1-hiera-base-plus",
        device: str = "auto",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        self.grounding_config = Path(grounding_config) if grounding_config else None
        self.grounding_checkpoint = Path(grounding_checkpoint) if grounding_checkpoint else None
        self.sam2_checkpoint = Path(sam2_checkpoint) if sam2_checkpoint else None
        self.sam2_model_id = sam2_model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._detector: Any = None
        self._tracker: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._detector is not None:
            return self._detector, self._tracker
        import torch

        from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
            checkpoint_path,
            config_path,
        )
        from worldfoundry.base_models.perception_core.detection.grounding_dino.util.inference import Model
        from worldfoundry.base_models.perception_core.segment.sam2.video_tracker import SAM2MaskTracker

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        detector_checkpoint = self.grounding_checkpoint or checkpoint_path()
        if not detector_checkpoint.is_file():
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found: {detector_checkpoint}; set WORLDFOUNDRY_GROUNDING_DINO_CKPT"
            )
        self._detector = Model(
            model_config_path=str(self.grounding_config or config_path()),
            model_checkpoint_path=str(detector_checkpoint),
            device=resolved_device,
        )
        self._tracker = SAM2MaskTracker(
            model_id=self.sam2_model_id,
            checkpoint=self.sam2_checkpoint,
            device=resolved_device,
        )
        return self._detector, self._tracker

    def segment(self, frames: np.ndarray, phrases: Sequence[str]) -> list[dict[str, Any]]:
        from worldfoundry.base_models.perception_core.segment.sam2.video_tracker import stage_video_frames

        frames = np.asarray(frames, dtype=np.uint8)
        classes = [str(phrase).strip() for phrase in phrases if str(phrase).strip()]
        if not classes:
            return []
        detector, tracker = self._load()
        detector_frame = np.ascontiguousarray(frames[0][..., ::-1])
        if len(classes) == 1:
            detections, phrase_labels = detector.predict_with_caption(
                image=detector_frame,
                caption=classes[0],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        else:
            detections = detector.predict_with_classes(
                image=detector_frame,
                classes=classes,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            class_ids = (
                np.asarray(detections.class_id)
                if detections.class_id is not None
                else np.zeros(len(detections.xyxy), dtype=int)
            )
            phrase_labels = [
                classes[int(class_id)] if 0 <= int(class_id) < len(classes) else classes[0] for class_id in class_ids
            ]
        boxes = np.asarray(detections.xyxy, dtype=np.float32)
        if not len(boxes):
            return []
        height, width = frames.shape[1:3]
        with tempfile.TemporaryDirectory(prefix="worldfoundry-grounded-sam2-") as directory:
            frames_dir = Path(directory) / "frames"
            stage_video_frames(frames, frames_dir, size=(height, width))
            prompts = {index + 1: (0, box) for index, box in enumerate(boxes)}
            label_maps = np.stack(
                tracker.track(
                    frames_dir,
                    prompts,
                    frame_count=len(frames),
                    size=(height, width),
                )
            )
        records = []
        for index, phrase in enumerate(phrase_labels):
            records.append({"phrase": str(phrase), "mask": label_maps == (index + 1)})
        return records
