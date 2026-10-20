# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""YOLO-World person anchors and phase-local SAM2 video propagation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from .schemas import BoundingBox


class PersonDetector(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> list[tuple[BoundingBox, float]]: ...


class MaskPropagator(Protocol):
    def track(
        self,
        frames_rgb: Sequence[np.ndarray],
        prompts: dict[int, list[tuple[int, BoundingBox]]],
        *,
        propagate: bool = True,
    ) -> dict[int, dict[int, np.ndarray]]: ...


class YoloWorldPersonDetector:
    """Ultralytics YOLO-World detector restricted to the ``person`` class."""

    def __init__(
        self,
        *,
        checkpoint: str,
        clip_checkpoint: str,
        device: str,
        confidence: float = 0.05,
    ) -> None:
        try:
            from ultralytics import YOLOWorld
            from ultralytics.nn.text_model import CLIP
        except ImportError as error:
            raise EvaluationError(
                "Human Appearance requires ultralytics YOLO-World; install the WorldMemBench metric requirements"
            ) from error
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise EvaluationError(f"YOLO-World checkpoint not found: {checkpoint_path}")
        clip_path = Path(clip_checkpoint).expanduser().resolve()
        if not clip_path.is_file():
            raise EvaluationError(f"YOLO-World CLIP checkpoint not found: {clip_path}")
        self.model = YOLOWorld(str(checkpoint_path))
        # The upstream CLIP wrapper forwards `size` to clip.load, which also
        # accepts a local file. Seed its cache so set_classes never downloads.
        self.model.model.clip_model = CLIP(
            str(clip_path),
            device=next(self.model.model.parameters()).device,
        )
        self.model.set_classes(["person"])
        self.device = device
        self.confidence = confidence

    def detect(self, frame_rgb: np.ndarray) -> list[tuple[BoundingBox, float]]:
        results = self.model.predict(
            source=frame_rgb,
            classes=[0],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        detections: list[tuple[BoundingBox, float]] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections
        coordinates = boxes.xyxy.detach().float().cpu().numpy()
        confidences = boxes.conf.detach().float().cpu().numpy()
        for raw_box, confidence in zip(coordinates, confidences):
            try:
                box = BoundingBox.from_value(raw_box.tolist(), field_name="YOLO person")
            except ValueError:
                continue
            detections.append((box, float(confidence)))
        return detections


class TransformersSam2MaskPropagator:
    """SAM2 video tracking in both temporal directions within one phase."""

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        min_mask_area_ratio: float = 0.0,
    ) -> None:
        try:
            from transformers import Sam2VideoModel, Sam2VideoProcessor
        except ImportError as error:
            raise EvaluationError(
                "Human Appearance requires transformers with Sam2VideoModel; "
                "install the WorldMemBench metric requirements"
            ) from error
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise EvaluationError(f"SAM2 checkpoint not found: {checkpoint_path}")
        self.processor = Sam2VideoProcessor.from_pretrained(
            checkpoint_path,
            local_files_only=True,
        )
        self.model = (
            Sam2VideoModel.from_pretrained(
                checkpoint_path,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.min_mask_area_ratio = min_mask_area_ratio

    def track(
        self,
        frames_rgb: Sequence[np.ndarray],
        prompts: dict[int, list[tuple[int, BoundingBox]]],
        *,
        propagate: bool = True,
    ) -> dict[int, dict[int, np.ndarray]]:
        import torch
        from PIL import Image

        if not frames_rgb or not prompts:
            return {}
        height, width = frames_rgb[0].shape[:2]
        pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
        session = self.processor.init_video_session(
            video=pil_frames,
            inference_device=self.device,
            # Evicted image features are recomputed on demand. Temporal memory,
            # conditioning masks, frame order and numeric precision are unchanged.
            max_vision_features_cache_size=1,
        )
        known_ids = sorted(prompts)
        tracks: dict[int, dict[int, np.ndarray]] = {target_id: {} for target_id in known_ids}

        def store_output(
            output: Any,
            *,
            prompted_target_id: int | None = None,
        ) -> None:
            masks = getattr(output, "pred_masks", None)
            if masks is None:
                return
            processed = self.processor.post_process_masks(
                [masks],
                original_sizes=[[height, width]],
                binarize=True,
            )[0]
            binary = processed.detach().cpu().numpy()
            if binary.ndim == 4 and binary.shape[1] == 1:
                binary = binary[:, 0]
            output_ids = list(session.obj_ids)
            for index, target_id in enumerate(output_ids):
                target_id = int(target_id)
                if (
                    index >= binary.shape[0]
                    or target_id not in tracks
                    or (prompted_target_id is not None and target_id != prompted_target_id)
                ):
                    continue
                mask = np.asarray(binary[index], dtype=bool)
                if mask.shape != (height, width):
                    continue
                if float(mask.mean()) <= self.min_mask_area_ratio:
                    continue
                frame_index = int(output.frame_idx)
                if prompted_target_id is not None:
                    # A mask inferred directly from this frame's person box is
                    # more reliable than any propagated mask at the same frame.
                    tracks[target_id][frame_index] = mask
                else:
                    tracks[target_id].setdefault(frame_index, mask)

        first_prompt = len(frames_rgb)
        last_prompt = 0
        with torch.inference_mode():
            for target_id in known_ids:
                for frame_index, box in prompts[target_id]:
                    first_prompt = min(first_prompt, frame_index)
                    last_prompt = max(last_prompt, frame_index)
                    self.processor.add_inputs_to_inference_session(
                        session,
                        frame_idx=frame_index,
                        obj_ids=target_id,
                        input_boxes=[[[float(value) for value in box.as_list()]]],
                        clear_old_inputs=True,
                    )
                    # Hugging Face SAM2 marks new inputs at object scope rather
                    # than frame scope. Consume each box immediately so every
                    # prompted frame becomes a conditioning frame before the
                    # next prompt overwrites that object-level marker.
                    output = self.model(session, frame_idx=frame_index)
                    store_output(output, prompted_target_id=target_id)

        if not propagate:
            return tracks

        directions = ((first_prompt, False), (last_prompt, True))
        for start_frame, reverse in directions:
            with torch.inference_mode():
                iterator = self.model.propagate_in_video_iterator(
                    session,
                    start_frame_idx=start_frame,
                    reverse=reverse,
                )
                for output in iterator:
                    store_output(output)
        return tracks
