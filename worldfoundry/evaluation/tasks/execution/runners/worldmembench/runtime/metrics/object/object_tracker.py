# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""First-frame-mask SAM2 tracking with independent object-presence evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from ..revisit_protocol import MAX_SHORT_MISSED_FRAMES
from .schemas import ObjectRedetectionPrompt, ObjectTrackObservation

PRESENCE_LOGIT_THRESHOLD = 0.0
MIN_MASK_AREA_RATIO = 0.0005
MAX_MASK_AREA_RATIO = 0.70
MIN_MASK_AREA_RELATIVE_TO_SEED = 0.10
MIN_REDETECTED_TRACKLET_FRAMES = 3


def _anchor_connected_presence_indices(
    observations: dict[int, ObjectTrackObservation],
    *,
    total_frames: int,
    anchor_index: int,
    max_short_missed_frames: int = MAX_SHORT_MISSED_FRAMES,
) -> tuple[int, ...]:
    presence = [bool(observations.get(index) and observations[index].present) for index in range(total_frames)]
    index = 0
    while index < total_frames:
        if presence[index]:
            index += 1
            continue
        start = index
        while index < total_frames and not presence[index]:
            index += 1
        if start > 0 and index < total_frames and index - start <= max_short_missed_frames:
            presence[start:index] = [True] * (index - start)
    if not presence[anchor_index]:
        return ()
    start = anchor_index
    while start > 0 and presence[start - 1]:
        start -= 1
    end = anchor_index
    while end + 1 < total_frames and presence[end + 1]:
        end += 1
    return tuple(range(start, end + 1))


class ObjectTrackerBackend(Protocol):
    def track(
        self,
        frames_rgb: Sequence[np.ndarray],
        first_frame_mask: np.ndarray,
        redetection_prompts: Sequence[ObjectRedetectionPrompt] = (),
    ) -> dict[int, ObjectTrackObservation]: ...


class TransformersSam2ObjectTracker:
    """Track one seeded object and preserve SAM2's occlusion/presence logit."""

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str,
        presence_logit_threshold: float = PRESENCE_LOGIT_THRESHOLD,
        min_mask_area_ratio: float = MIN_MASK_AREA_RATIO,
        max_mask_area_ratio: float = MAX_MASK_AREA_RATIO,
        min_mask_area_relative_to_seed: float = MIN_MASK_AREA_RELATIVE_TO_SEED,
        dtype: str = "bfloat16",
        vision_cache_size: int | None = 1,
    ) -> None:
        try:
            import torch
            from transformers import Sam2VideoModel, Sam2VideoProcessor
        except ImportError as error:
            raise EvaluationError(
                "Object Appearance requires transformers with Sam2VideoModel; "
                "install the WorldMemBench metric requirements"
            ) from error
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise EvaluationError(f"SAM2 checkpoint not found: {checkpoint_path}")
        if vision_cache_size is not None and (
            isinstance(vision_cache_size, bool) or not isinstance(vision_cache_size, int) or vision_cache_size < 1
        ):
            raise ValueError("sam2_vision_cache_size must be a positive integer or null (all frames)")
        self.vision_cache_size = vision_cache_size
        self._dtype = getattr(torch, dtype)
        self.processor = Sam2VideoProcessor.from_pretrained(
            checkpoint_path,
            local_files_only=True,
        )
        self.model = (
            Sam2VideoModel.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                torch_dtype=self._dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.presence_logit_threshold = presence_logit_threshold
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_mask_area_ratio = max_mask_area_ratio
        self.min_mask_area_relative_to_seed = min_mask_area_relative_to_seed

    def _observation(
        self,
        output: Any,
        *,
        height: int,
        width: int,
        minimum_mask_area_ratio: float,
    ) -> ObjectTrackObservation:
        masks, logits = output.pred_masks, output.object_score_logits
        if masks is None or logits is None:
            raise EvaluationError("SAM2 must return masks and object_score_logits; check the installed version")
        frame_index = int(output.frame_idx)

        processed = self.processor.post_process_masks(
            [masks],
            original_sizes=[[height, width]],
            binarize=True,
        )[0]
        binary = processed.detach().cpu().numpy()
        while binary.ndim > 2:
            binary = binary[0]
        mask = np.asarray(binary, dtype=bool)
        if mask.shape != (height, width):
            raise EvaluationError(f"SAM2 returned mask shape {mask.shape}, expected {(height, width)}")

        presence_logit = float(logits.detach().float().cpu().numpy().reshape(-1)[0])
        if not math.isfinite(presence_logit):
            raise EvaluationError("SAM2 returned a non-finite object score")
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, presence_logit))))
        area_ratio = float(mask.mean())
        mask_valid = bool(mask.any()) and minimum_mask_area_ratio <= area_ratio <= self.max_mask_area_ratio
        present = presence_logit > self.presence_logit_threshold and mask_valid
        return ObjectTrackObservation(
            frame_index=frame_index,
            mask=mask,
            presence_logit=presence_logit,
            presence_probability=probability,
            mask_area_ratio=area_ratio,
            mask_valid=mask_valid,
            present=present,
        )

    def track(
        self,
        frames_rgb: Sequence[np.ndarray],
        first_frame_mask: np.ndarray,
        redetection_prompts: Sequence[ObjectRedetectionPrompt] = (),
    ) -> dict[int, ObjectTrackObservation]:
        import torch
        from PIL import Image

        if not frames_rgb:
            return {}
        height, width = frames_rgb[0].shape[:2]
        if first_frame_mask.shape != (height, width) or not first_frame_mask.any():
            raise EvaluationError("the aligned first-frame object mask is empty or has the wrong size")
        ordered_prompts = tuple(sorted(redetection_prompts, key=lambda prompt: prompt.frame_index))
        for prompt in ordered_prompts:
            if prompt.frame_index <= 0 or prompt.frame_index >= len(frames_rgb):
                raise EvaluationError("SAM2 redetection prompt frame is out of range")
            x1, y1, x2, y2 = prompt.box_xyxy
            if not (x2 > x1 and y2 > y1):
                raise EvaluationError("SAM2 redetection prompt box is invalid")

        minimum_mask_area_ratio = max(
            self.min_mask_area_ratio,
            float(first_frame_mask.mean()) * self.min_mask_area_relative_to_seed,
        )
        pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
        # None keeps all frames for large GPUs; the default recomputes evicted
        # image features. Neither mode drops temporal tracking state or frames.
        session = self.processor.init_video_session(
            video=pil_frames,
            inference_device=self.device,
            max_vision_features_cache_size=self.vision_cache_size or len(pil_frames),
            dtype=self._dtype,
        )

        def store(
            destination: dict[int, ObjectTrackObservation],
            observation: ObjectTrackObservation,
        ) -> None:
            previous = destination.get(observation.frame_index)
            if (
                previous is None
                or (observation.present and not previous.present)
                or (
                    observation.present
                    and previous.present
                    and observation.presence_probability > previous.presence_probability
                )
            ):
                destination[observation.frame_index] = observation

        def run_tracklet(
            *,
            anchor_index: int,
            input_mask: np.ndarray | None = None,
            input_box: tuple[float, float, float, float] | None = None,
            propagate_reverse: bool,
        ) -> dict[int, ObjectTrackObservation]:
            if input_mask is not None:
                self.processor.add_inputs_to_inference_session(
                    session,
                    frame_idx=anchor_index,
                    obj_ids=0,
                    input_masks=np.asarray(input_mask, dtype=bool),
                    clear_old_inputs=True,
                )
            elif input_box is not None:
                x1, y1, x2, y2 = input_box
                center_point = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
                self.processor.add_inputs_to_inference_session(
                    session,
                    frame_idx=anchor_index,
                    obj_ids=0,
                    input_points=[[[center_point]]],
                    input_labels=[[[1]]],
                    input_boxes=[[[float(value) for value in input_box]]],
                    clear_old_inputs=True,
                )
            else:
                raise EvaluationError("SAM2 tracklet requires a mask or box prompt")

            tracklet: dict[int, ObjectTrackObservation] = {}
            with torch.inference_mode():
                conditioned = self.model(session, frame_idx=anchor_index)
                store(
                    tracklet,
                    self._observation(
                        conditioned,
                        height=height,
                        width=width,
                        minimum_mask_area_ratio=minimum_mask_area_ratio,
                    ),
                )
                directions = [False]
                if propagate_reverse:
                    directions.insert(0, True)
                for reverse in directions:
                    iterator = self.model.propagate_in_video_iterator(
                        session,
                        start_frame_idx=anchor_index,
                        reverse=reverse,
                    )
                    for output in iterator:
                        store(
                            tracklet,
                            self._observation(
                                output,
                                height=height,
                                width=width,
                                minimum_mask_area_ratio=minimum_mask_area_ratio,
                            ),
                        )

            connected_indices = _anchor_connected_presence_indices(
                tracklet,
                total_frames=len(frames_rgb),
                anchor_index=anchor_index,
            )
            return {
                index: tracklet[index] for index in connected_indices if index in tracklet and tracklet[index].present
            }

        observations = run_tracklet(
            anchor_index=0,
            input_mask=first_frame_mask,
            propagate_reverse=False,
        )
        accepted_indices = set(observations)
        for prompt in ordered_prompts:
            existing = observations.get(prompt.frame_index)
            if existing is not None and existing.present:
                continue
            recovered = run_tracklet(
                anchor_index=prompt.frame_index,
                input_box=prompt.box_xyxy,
                propagate_reverse=True,
            )
            if len(recovered) < MIN_REDETECTED_TRACKLET_FRAMES:
                continue
            if accepted_indices.intersection(recovered):
                continue
            for observation in recovered.values():
                store(observations, observation)
            accepted_indices.update(recovered)

        for frame_index in range(len(frames_rgb)):
            observations.setdefault(
                frame_index,
                ObjectTrackObservation(
                    frame_index=frame_index,
                    mask=None,
                    presence_logit=None,
                    presence_probability=0.0,
                    mask_area_ratio=0.0,
                    mask_valid=False,
                    present=False,
                ),
            )
        return observations
