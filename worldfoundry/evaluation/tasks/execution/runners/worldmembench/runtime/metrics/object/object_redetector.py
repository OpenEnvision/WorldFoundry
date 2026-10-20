# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Text-prompted object re-detection independent from appearance scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from .schemas import ObjectRedetectionPrompt

REDETECTION_SEARCH_STRIDE = 4
REDETECTION_BATCH_SIZE = 8
REDETECTION_BOX_THRESHOLD = 0.30
REDETECTION_TEXT_THRESHOLD = 0.25
REDETECTION_NMS_THRESHOLD = 0.50
REDETECTION_CANDIDATE_LIMIT = 64
REID_BATCH_SIZE = 64
REID_SIMILARITY_THRESHOLD = 0.75
REID_HIGH_CONFIDENCE_THRESHOLD = 0.90
REID_MIN_SIMILARITY_MARGIN = 0.04
REID_CROP_CONTEXT_FRACTION = 0.10
REDETECTION_MIN_CONFIRMATIONS = 2
REDETECTION_MAX_SAMPLE_GAP = 2
REDETECTION_MIN_BOX_AREA_RATIO = 0.0005
REDETECTION_MAX_BOX_AREA_RATIO = 0.70
REDETECTION_MAX_CENTER_DISTANCE_RATIO = 0.35
REDETECTION_MAX_AREA_CHANGE = 10.0
QUERY_CONTEXT_FRACTION = 0.10


@dataclass(frozen=True)
class _DetectionCandidate:
    detector_score: float
    box_xyxy: tuple[float, float, float, float]


def reid_match_is_accepted(
    top_similarity: float,
    runner_up_similarity: float | None,
    *,
    similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
    high_confidence_threshold: float = REID_HIGH_CONFIDENCE_THRESHOLD,
    min_similarity_margin: float = REID_MIN_SIMILARITY_MARGIN,
) -> bool:
    """Accept an absolute-high match or a lower match with usable evidence.

    A lone candidate has no relative-margin evidence, but still has absolute
    similarity evidence and must pass the downstream temporal confirmation.
    """

    if top_similarity < similarity_threshold:
        return False
    if top_similarity >= high_confidence_threshold:
        return True
    return runner_up_similarity is None or top_similarity - runner_up_similarity >= min_similarity_margin


class ObjectRedetectorBackend(Protocol):
    def detect(
        self,
        text_prompt: str,
        query_rgb: np.ndarray,
        frames_rgb: Sequence[np.ndarray],
    ) -> tuple[ObjectRedetectionPrompt, ...]: ...


def object_query_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    context_fraction: float = QUERY_CONTEXT_FRACTION,
) -> np.ndarray:
    """Crop the prompted object while retaining a small amount of natural context."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("object query image must be RGB")
    if mask.shape != image_rgb.shape[:2] or not mask.any():
        raise ValueError("object query mask must be non-empty and match the image")
    if context_fraction < 0.0:
        raise ValueError("context_fraction must be non-negative")

    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = max(2, int(round(max(x2 - x1, y2 - y1) * context_fraction)))
    height, width = mask.shape
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return np.ascontiguousarray(image_rgb[y1:y2, x1:x2])


def _box_center_and_area(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1) * (y2 - y1)


def _boxes_share_object_hypothesis(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    intersection_over_smaller_threshold: float = 0.50,
) -> bool:
    """Return whether two boxes are alternate crops of one spatial object."""

    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(
        0.0,
        min(left_y2, right_y2) - max(left_y1, right_y1),
    )
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0.0:
        return False
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(
        0.0,
        right_y2 - right_y1,
    )
    smaller_area = min(left_area, right_area)
    return smaller_area > 0.0 and intersection_area / smaller_area >= intersection_over_smaller_threshold


def _temporally_compatible(
    previous: ObjectRedetectionPrompt,
    current: ObjectRedetectionPrompt,
    *,
    frame_height: int,
    frame_width: int,
    max_center_distance_ratio: float,
    max_area_change: float,
) -> bool:
    previous_x, previous_y, previous_area = _box_center_and_area(previous.box_xyxy)
    current_x, current_y, current_area = _box_center_and_area(current.box_xyxy)
    diagonal = max(1.0, float(np.hypot(frame_width, frame_height)))
    center_distance = float(np.hypot(current_x - previous_x, current_y - previous_y) / diagonal)
    if center_distance > max_center_distance_ratio:
        return False
    smaller_area = min(previous_area, current_area)
    larger_area = max(previous_area, current_area)
    return smaller_area > 0.0 and larger_area / smaller_area <= max_area_change


def confirm_redetection_prompts(
    prompts: Sequence[ObjectRedetectionPrompt],
    *,
    frame_height: int,
    frame_width: int,
    search_stride: int = REDETECTION_SEARCH_STRIDE,
    min_confirmations: int = REDETECTION_MIN_CONFIRMATIONS,
    max_sample_gap: int = REDETECTION_MAX_SAMPLE_GAP,
    max_center_distance_ratio: float = REDETECTION_MAX_CENTER_DISTANCE_RATIO,
    max_area_change: float = REDETECTION_MAX_AREA_CHANGE,
) -> tuple[ObjectRedetectionPrompt, ...]:
    """Keep only temporally persistent ReID matches, without using score-time DINO."""

    if search_stride <= 0:
        raise ValueError("search_stride must be positive")
    if min_confirmations <= 0:
        raise ValueError("min_confirmations must be positive")
    if max_sample_gap <= 0:
        raise ValueError("max_sample_gap must be positive")
    ordered = sorted(prompts, key=lambda prompt: prompt.frame_index)
    if not ordered:
        return ()

    runs: list[list[ObjectRedetectionPrompt]] = []
    current_run = [ordered[0]]
    maximum_gap = search_stride * max_sample_gap
    for prompt in ordered[1:]:
        previous = current_run[-1]
        compatible = 0 < prompt.frame_index - previous.frame_index <= maximum_gap and _temporally_compatible(
            previous,
            prompt,
            frame_height=frame_height,
            frame_width=frame_width,
            max_center_distance_ratio=max_center_distance_ratio,
            max_area_change=max_area_change,
        )
        if compatible:
            current_run.append(prompt)
        else:
            runs.append(current_run)
            current_run = [prompt]
    runs.append(current_run)
    return tuple(prompt for run in runs if len(run) >= min_confirmations for prompt in run)


class GroundingDinoClipObjectRedetector:
    """Generate candidates with Grounding DINO text-prompt detection and confirm with independent CLIP ReID."""

    def __init__(
        self,
        *,
        grounding_dino_checkpoint: str,
        reid_checkpoint: str,
        device: str,
        search_stride: int = REDETECTION_SEARCH_STRIDE,
        batch_size: int = REDETECTION_BATCH_SIZE,
        box_threshold: float = REDETECTION_BOX_THRESHOLD,
        text_threshold: float = REDETECTION_TEXT_THRESHOLD,
        nms_threshold: float = REDETECTION_NMS_THRESHOLD,
        candidate_limit: int = REDETECTION_CANDIDATE_LIMIT,
        reid_batch_size: int = REID_BATCH_SIZE,
        reid_similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        reid_high_confidence_threshold: float = (REID_HIGH_CONFIDENCE_THRESHOLD),
        reid_min_similarity_margin: float = REID_MIN_SIMILARITY_MARGIN,
        reid_crop_context_fraction: float = REID_CROP_CONTEXT_FRACTION,
        min_confirmations: int = REDETECTION_MIN_CONFIRMATIONS,
        max_sample_gap: int = REDETECTION_MAX_SAMPLE_GAP,
        min_box_area_ratio: float = REDETECTION_MIN_BOX_AREA_RATIO,
        max_box_area_ratio: float = REDETECTION_MAX_BOX_AREA_RATIO,
    ) -> None:
        try:
            import torch
            from torchvision.ops import roi_align
            from transformers import (
                AutoProcessor,
                GroundingDinoForObjectDetection,
            )

            from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
        except ImportError as error:
            raise EvaluationError(
                "Object Appearance re-detection requires Grounding DINO and "
                "openai-clip; install the WorldMemBench metric requirements"
            ) from error

        grounding_dino_checkpoint_path = Path(grounding_dino_checkpoint).expanduser().resolve()
        reid_checkpoint_path = Path(reid_checkpoint).expanduser().resolve()
        if not grounding_dino_checkpoint_path.exists():
            raise EvaluationError(f"Grounding DINO checkpoint not found: {grounding_dino_checkpoint_path}")
        if not reid_checkpoint_path.is_file():
            raise EvaluationError(f"CLIP ReID checkpoint not found: {reid_checkpoint_path}")
        self.processor = AutoProcessor.from_pretrained(
            grounding_dino_checkpoint_path,
            local_files_only=True,
        )
        self.model = (
            GroundingDinoForObjectDetection.from_pretrained(
                grounding_dino_checkpoint_path,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.reid_model, self.reid_preprocess = clip.load(
            str(reid_checkpoint_path),
            device=device,
            jit=False,
        )
        self.reid_model.eval()
        self.roi_align = roi_align
        self.reid_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device,
        ).view(1, 3, 1, 1)
        self.reid_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device,
        ).view(1, 3, 1, 1)

        self.device = device
        self.search_stride = search_stride
        self.batch_size = batch_size
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.candidate_limit = candidate_limit
        self.reid_batch_size = reid_batch_size
        self.reid_similarity_threshold = reid_similarity_threshold
        self.reid_high_confidence_threshold = reid_high_confidence_threshold
        self.reid_min_similarity_margin = reid_min_similarity_margin
        self.reid_crop_context_fraction = reid_crop_context_fraction
        self.min_confirmations = min_confirmations
        self.max_sample_gap = max_sample_gap
        self.min_box_area_ratio = min_box_area_ratio
        self.max_box_area_ratio = max_box_area_ratio

    def _candidates(
        self,
        result: dict[str, object],
        *,
        frame_height: int,
        frame_width: int,
    ) -> tuple[_DetectionCandidate, ...]:
        boxes_tensor = result["boxes"]
        scores_tensor = result["scores"]
        boxes = boxes_tensor.detach().float().cpu().numpy()
        scores = scores_tensor.detach().float().cpu().numpy()
        frame_area = float(frame_height * frame_width)
        candidates: list[_DetectionCandidate] = []
        for raw_box, raw_score in zip(boxes, scores, strict=True):
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            x1 = min(max(x1, 0.0), float(frame_width))
            x2 = min(max(x2, 0.0), float(frame_width))
            y1 = min(max(y1, 0.0), float(frame_height))
            y2 = min(max(y2, 0.0), float(frame_height))
            area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
            if np.isfinite(raw_score) and self.min_box_area_ratio <= area_ratio <= self.max_box_area_ratio:
                candidates.append(
                    _DetectionCandidate(
                        detector_score=float(raw_score),
                        box_xyxy=(x1, y1, x2, y2),
                    )
                )
        if len(candidates) > 1:
            import torch
            from torchvision.ops import nms

            boxes_tensor = torch.tensor(
                [candidate.box_xyxy for candidate in candidates],
                dtype=torch.float32,
            )
            scores_tensor = torch.tensor(
                [candidate.detector_score for candidate in candidates],
                dtype=torch.float32,
            )
            keep = nms(boxes_tensor, scores_tensor, self.nms_threshold)
            candidates = [candidates[int(index)] for index in keep.tolist()]
        candidates.sort(
            key=lambda candidate: candidate.detector_score,
            reverse=True,
        )
        return tuple(candidates[: self.candidate_limit])

    def _reidentify(
        self,
        *,
        query_embedding: object,
        batch_indices: Sequence[int],
        frames_rgb: Sequence[np.ndarray],
        candidates_by_frame: Sequence[Sequence[_DetectionCandidate]],
    ) -> list[ObjectRedetectionPrompt]:
        import torch

        owners: list[tuple[int, _DetectionCandidate]] = []
        rois: list[list[float]] = []
        for owner, (frame_index, candidates) in enumerate(zip(batch_indices, candidates_by_frame, strict=True)):
            height, width = frames_rgb[frame_index].shape[:2]
            for candidate in candidates:
                x1, y1, x2, y2 = candidate.box_xyxy
                padding = max(x2 - x1, y2 - y1) * self.reid_crop_context_fraction
                rois.append(
                    [
                        float(owner),
                        max(0.0, x1 - padding),
                        max(0.0, y1 - padding),
                        min(float(width), x2 + padding),
                        min(float(height), y2 + padding),
                    ]
                )
                owners.append((owner, candidate))
        if not rois:
            return []

        frame_tensor = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(frames_rgb[index])).permute(2, 0, 1) for index in batch_indices]
        ).to(device=self.device, dtype=torch.float32)
        frame_tensor = frame_tensor / 255.0
        crop_tensor = self.roi_align(
            frame_tensor,
            torch.tensor(rois, device=self.device, dtype=torch.float32),
            output_size=(224, 224),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        crop_tensor = (crop_tensor - self.reid_mean) / self.reid_std

        similarities: list[float] = []
        with torch.inference_mode():
            for offset in range(0, len(crop_tensor), self.reid_batch_size):
                features = self.reid_model.encode_image(crop_tensor[offset : offset + self.reid_batch_size])
                features = features / features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                batch_similarities = (features @ query_embedding.T).squeeze(1)
                similarities.extend(batch_similarities.float().cpu().tolist())

        ranked_by_owner: list[list[tuple[float, _DetectionCandidate]]] = [[] for _ in batch_indices]
        for (owner, candidate), similarity in zip(
            owners,
            similarities,
            strict=True,
        ):
            ranked_by_owner[owner].append((similarity, candidate))

        prompts: list[ObjectRedetectionPrompt] = []
        for owner, ranked in enumerate(ranked_by_owner):
            ranked.sort(key=lambda entry: entry[0], reverse=True)
            frame_index = batch_indices[owner]
            if not ranked:
                continue
            top_similarity, top_candidate = ranked[0]
            runner_up_similarity = next(
                (
                    similarity
                    for similarity, candidate in ranked[1:]
                    if not _boxes_share_object_hypothesis(
                        top_candidate.box_xyxy,
                        candidate.box_xyxy,
                    )
                ),
                None,
            )
            if not reid_match_is_accepted(
                top_similarity,
                runner_up_similarity,
                similarity_threshold=self.reid_similarity_threshold,
                high_confidence_threshold=self.reid_high_confidence_threshold,
                min_similarity_margin=self.reid_min_similarity_margin,
            ):
                continue
            prompts.append(
                ObjectRedetectionPrompt(
                    frame_index=frame_index,
                    box_xyxy=top_candidate.box_xyxy,
                    detector_score=top_candidate.detector_score,
                    reid_similarity=top_similarity,
                    runner_up_reid_similarity=runner_up_similarity,
                    candidate_count=len(ranked),
                )
            )
        return prompts

    def detect(
        self,
        text_prompt: str,
        query_rgb: np.ndarray,
        frames_rgb: Sequence[np.ndarray],
    ) -> tuple[ObjectRedetectionPrompt, ...]:
        import torch
        from PIL import Image

        if not frames_rgb:
            return ()
        frame_height, frame_width = frames_rgb[0].shape[:2]
        if any(frame.shape[:2] != (frame_height, frame_width) for frame in frames_rgb):
            raise EvaluationError("Object Appearance requires constant-size video frames")

        text = text_prompt.strip()
        if not text:
            raise EvaluationError("Grounding DINO requires a non-empty text prompt")
        sampled_indices = list(range(self.search_stride, len(frames_rgb), self.search_stride))
        raw_prompts: list[ObjectRedetectionPrompt] = []
        query_image = Image.fromarray(query_rgb)
        with torch.inference_mode():
            query_embedding = self.reid_model.encode_image(
                self.reid_preprocess(query_image).unsqueeze(0).to(self.device)
            )
            query_embedding = query_embedding / query_embedding.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

        for offset in range(0, len(sampled_indices), self.batch_size):
            batch_indices = sampled_indices[offset : offset + self.batch_size]
            target_images = [Image.fromarray(frames_rgb[index]) for index in batch_indices]
            # transformers >= 5 treats a plain str or list[str] as one sample
            # ("one image, many phrases"); list[list[str]] is the only form
            # that yields one text batch entry per image.
            inputs = self.processor(
                images=target_images,
                text=[[text]] * len(batch_indices),
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            target_sizes = torch.tensor(
                [[frame_height, frame_width]] * len(batch_indices),
                device=self.device,
            )
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
            candidates_by_frame = [
                self._candidates(
                    result,
                    frame_height=frame_height,
                    frame_width=frame_width,
                )
                for result in results
            ]
            raw_prompts.extend(
                self._reidentify(
                    query_embedding=query_embedding,
                    batch_indices=batch_indices,
                    frames_rgb=frames_rgb,
                    candidates_by_frame=candidates_by_frame,
                )
            )

        return confirm_redetection_prompts(
            raw_prompts,
            frame_height=frame_height,
            frame_width=frame_width,
            search_stride=self.search_stride,
            min_confirmations=self.min_confirmations,
            max_sample_gap=self.max_sample_gap,
        )


__all__ = [
    "ObjectRedetectorBackend",
    "GroundingDinoClipObjectRedetector",
    "confirm_redetection_prompts",
    "object_query_crop",
    "reid_match_is_accepted",
]
