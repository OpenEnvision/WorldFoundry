"""Reusable SAM3 video segmentation for inference pipelines.

The model implementation and this high-level inference wrapper both live in
``base_models`` so downstream integrations share one SAM3 stack.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.perception_core.segment.sam3.model_builder import (
    build_sam3_video_predictor,
)
from worldfoundry.core.media.codecs.video import load_video_frames

_NULL_PROMPT_PHRASES = (
    "nothing",
    "none",
    "no moving",
    "no motion",
    "no movement",
    "no moving object",
    "no moving objects",
    "no dynamic",
    "only background",
    "background only",
    "static scene",
    "static background",
)


@dataclass
class Sam3VideoSegmenter:
    gpus_to_use: list[int] | None = None
    propagation_direction: str = "both"
    score_threshold_detection: float | None = None
    new_det_thresh: float | None = None
    checkpoint_path: str | None = None
    multi_prompt: bool = False
    resize_input_to: int | None = None

    def __post_init__(self):
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=self.gpus_to_use,
            checkpoint_path=self.checkpoint_path,
        )
        self._apply_model_overrides()

    def _apply_model_overrides(self) -> None:
        model = getattr(self.predictor, "model", None)
        if model is None:
            return
        if self.score_threshold_detection is not None and hasattr(model, "score_threshold_detection"):
            model.score_threshold_detection = float(self.score_threshold_detection)
        if self.new_det_thresh is not None and hasattr(model, "new_det_thresh"):
            model.new_det_thresh = float(self.new_det_thresh)

    @staticmethod
    def normalize_prompts(prompts: Iterable[str]) -> list[str]:
        normalized = []
        seen = set()
        for prompt in prompts:
            if prompt is None:
                continue
            text = str(prompt).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in {"nothing", "none"} or any(p in lowered for p in _NULL_PROMPT_PHRASES):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(text)
        return normalized

    def _segment_single_prompt(
        self,
        video_resource: str | list[Image.Image],
        prompt: str,
        frame_index: int,
        start_frame_index: int | None,
        max_frame_num_to_track: int | None,
        expected_frames: int | None,
    ) -> np.ndarray:
        response = self.predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_resource,
            }
        )
        session_id = response["session_id"]

        masks_by_frame: dict[int, np.ndarray] = {}
        try:
            self.predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "text": prompt,
                    "obj_id": 1,
                }
            )

            for response in self.predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": self.propagation_direction,
                    "start_frame_index": start_frame_index,
                    "max_frame_num_to_track": max_frame_num_to_track,
                }
            ):
                response["prompt"] = prompt
                frame_idx = int(response["frame_index"])
                outputs = response["outputs"]
                masks = outputs["out_binary_masks"]
                masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
                merged = None if masks.size == 0 else np.any(masks, axis=0)

                if merged is None:
                    if frame_idx not in masks_by_frame:
                        masks_by_frame[frame_idx] = None
                else:
                    if frame_idx in masks_by_frame and masks_by_frame[frame_idx] is not None:
                        masks_by_frame[frame_idx] = masks_by_frame[frame_idx] | merged
                    else:
                        masks_by_frame[frame_idx] = merged
        finally:
            self.predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )

        result = self._finalize_masks(masks_by_frame, expected_frames)
        masks_by_frame.clear()
        return result

    def _segment_multi_prompt(
        self,
        video_resource: str | list[Image.Image],
        prompts: Iterable[str],
        frame_index: int,
        start_frame_index: int | None,
        max_frame_num_to_track: int | None,
        expected_frames: int | None,
    ) -> np.ndarray:
        response = self.predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_resource,
            }
        )
        session_id = response["session_id"]

        masks_by_frame: dict[int, np.ndarray] = {}
        try:
            prompt_list = list(prompts)
            for obj_id, prompt in enumerate(prompt_list, start=1):
                self.predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "text": prompt,
                        "obj_id": obj_id,
                    }
                )

            for response in self.predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": self.propagation_direction,
                    "start_frame_index": start_frame_index,
                    "max_frame_num_to_track": max_frame_num_to_track,
                }
            ):
                response["prompts"] = prompt_list
                frame_idx = int(response["frame_index"])
                outputs = response["outputs"]
                masks = outputs["out_binary_masks"]
                masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
                merged = None if masks.size == 0 else np.any(masks, axis=0)

                if merged is None:
                    if frame_idx not in masks_by_frame:
                        masks_by_frame[frame_idx] = None
                else:
                    if frame_idx in masks_by_frame and masks_by_frame[frame_idx] is not None:
                        masks_by_frame[frame_idx] = masks_by_frame[frame_idx] | merged
                    else:
                        masks_by_frame[frame_idx] = merged
        finally:
            self.predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )

        result = self._finalize_masks(masks_by_frame, expected_frames)
        masks_by_frame.clear()
        return result

    @staticmethod
    def _finalize_masks(
        masks_by_frame: dict[int, np.ndarray],
        expected_frames: int | None,
    ) -> np.ndarray:
        if not masks_by_frame:
            return np.zeros((0, 0, 0), dtype=bool)

        max_idx = max(masks_by_frame.keys())
        num_frames = expected_frames if expected_frames is not None else max_idx + 1

        sample_mask = next((m for m in masks_by_frame.values() if m is not None), None)
        if sample_mask is None:
            return np.zeros((0, 0, 0), dtype=bool)
        H, W = sample_mask.shape

        masks_out = np.zeros((num_frames, H, W), dtype=bool)
        for idx, mask in masks_by_frame.items():
            if mask is None:
                continue
            if idx < num_frames:
                masks_out[idx] = mask

        if not masks_out.any():
            return np.zeros((0, 0, 0), dtype=bool)

        return masks_out

    @staticmethod
    def _resize_masks_to_hw(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        H, W = target_hw
        resized = []
        for mask in masks:
            if mask.shape != (H, W):
                mask_u8 = mask.astype(np.uint8) * 255
                mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = mask_u8 > 0
            resized.append(mask)
        return np.stack(resized, axis=0)

    @staticmethod
    def _letterbox_frames(
        frames: list[np.ndarray],
        target_size: int,
    ) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int], tuple[int, int]]:
        if not frames:
            return [], (0, 0), (0, 0), (0, 0)
        orig_h, orig_w = frames[0].shape[:2]
        scale = target_size / float(max(orig_h, orig_w))
        new_w = max(1, int(round(orig_w * scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        new_h = max(1, int(round(orig_h * scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        pad_x = max(0, (target_size - new_w) // 2)
        pad_y = max(0, (target_size - new_h) // 2)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        padded_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
            canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            padded_frames.append(canvas)
        return padded_frames, (orig_h, orig_w), (new_h, new_w), (pad_y, pad_x)

    def _restore_masks_from_letterbox(
        self,
        masks: np.ndarray,
        orig_hw: tuple[int, int],
        resized_hw: tuple[int, int],
        pad_xy: tuple[int, int],
    ) -> np.ndarray:
        pad_y, pad_x = pad_xy
        resized_h, resized_w = resized_hw
        if masks.size == 0:
            return masks
        cropped = masks[:, pad_y : pad_y + resized_h, pad_x : pad_x + resized_w]
        return self._resize_masks_to_hw(cropped, orig_hw)

    def _release_gpu_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def segment_per_category(
        self,
        video_path: str | list[Image.Image],
        prompts: Iterable[str],
        frame_index: int = 0,
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        expected_frames: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return a separate temporal mask array for every normalized prompt."""
        normalized = self.normalize_prompts(prompts)
        if not normalized:
            return {}

        video_resource: str | list[Image.Image] = video_path
        resize_meta = None
        owned_frames: list[Image.Image] | None = None
        if self.resize_input_to and isinstance(video_path, (str, Path)):
            frames = load_video_frames(video_path)
            padded, original_hw, resized_hw, padding = self._letterbox_frames(frames, self.resize_input_to)
            owned_frames = [Image.fromarray(frame) for frame in padded]
            video_resource = owned_frames
            expected_frames = len(owned_frames)
            resize_meta = (original_hw, resized_hw, padding)

        masks_by_category: dict[str, np.ndarray] = {}
        try:
            for prompt in normalized:
                masks = self._segment_single_prompt(
                    video_resource=video_resource,
                    prompt=prompt,
                    frame_index=frame_index,
                    start_frame_index=start_frame_index,
                    max_frame_num_to_track=max_frame_num_to_track,
                    expected_frames=expected_frames,
                )
                if masks.size == 0:
                    continue
                if resize_meta is not None:
                    original_hw, resized_hw, padding = resize_meta
                    masks = self._restore_masks_from_letterbox(masks, original_hw, resized_hw, padding)
                masks_by_category[prompt] = masks
        finally:
            if owned_frames is not None:
                for frame in owned_frames:
                    frame.close()
            self._release_gpu_memory()

        return masks_by_category

    def segment_instances(
        self,
        video_path: str,
        prompt: str,
        frame_index: int = 0,
        expected_frames: int | None = None,
    ) -> list[np.ndarray]:
        """Segment a single prompt and return per-instance masks (no merging).

        Unlike segment() which merges all instances into a union mask, this
        method preserves individual instance masks for instance-level dedup.

        Args:
            video_path: Path to image/video file, or list of PIL Images.
            prompt: Single text prompt for the entity class.
            frame_index: Frame index to detect on.
            expected_frames: Expected number of frames (use 1 for single image).

        Returns:
            List of (H, W) bool masks, one per detected instance.
            Empty list if nothing detected.
        """
        prompt = prompt.strip()
        if not prompt:
            return []

        video_resource: str | list[Image.Image] = video_path
        resize_meta = None
        pil_frames: list[Image.Image] | None = None
        if self.resize_input_to:
            frames = load_video_frames(video_path)
            padded_frames, orig_hw, resized_hw, pad_xy = self._letterbox_frames(frames, self.resize_input_to)
            pil_frames = [Image.fromarray(frame) for frame in padded_frames]
            video_resource = pil_frames
            _expected_frames = len(pil_frames)
            resize_meta = (orig_hw, resized_hw, pad_xy)
            del frames, padded_frames

        instance_masks: list[np.ndarray] = []
        try:
            response = self.predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": video_resource,
                }
            )
            session_id = response["session_id"]

            try:
                self.predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "text": prompt,
                        "obj_id": 1,
                    }
                )

                for response in self.predictor.handle_stream_request(
                    request={
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": self.propagation_direction,
                        "start_frame_index": None,
                        "max_frame_num_to_track": None,
                    }
                ):
                    outputs = response["outputs"]
                    masks = outputs["out_binary_masks"]
                    masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
                    if masks.size == 0:
                        continue
                    # Keep each instance mask separately (masks shape: N_instances, H, W).
                    for i in range(masks.shape[0]):
                        inst_mask = masks[i].astype(bool)
                        if inst_mask.any():
                            instance_masks.append(inst_mask)
            finally:
                self.predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
        finally:
            if pil_frames is not None:
                for img in pil_frames:
                    img.close()
                pil_frames.clear()
                del pil_frames
            self._release_gpu_memory()

        # Restore from letterbox if resize was applied.
        if resize_meta is not None and instance_masks:
            orig_hw, resized_hw, pad_xy = resize_meta
            stacked = np.stack(instance_masks, axis=0)
            restored = self._restore_masks_from_letterbox(stacked, orig_hw, resized_hw, pad_xy)
            instance_masks = [restored[i] for i in range(restored.shape[0])]

        return instance_masks

    def segment_instances_video(
        self,
        video_path: str | list[Image.Image],
        prompts: Iterable[str],
        frame_index: int = 0,
        expected_frames: int | None = None,
    ) -> dict[int, np.ndarray]:
        """Per-instance per-frame segmentation with temporal tracking.

        Each prompt is processed in its own session. SAM3's out_obj_ids are
        used to track instance identity across frames.

        Args:
            video_path: Path to video or list of PIL Images.
            prompts: Entity class names to segment.
            frame_index: Frame to detect on.
            expected_frames: Total number of frames in the video.

        Returns:
            {obj_id: (T, H, W) bool} — per-instance masks across all frames.
        """
        prompts = self.normalize_prompts(prompts)
        if not prompts:
            return {}

        video_resource: str | list[Image.Image] = video_path
        resize_meta = None
        pil_frames: list[Image.Image] | None = None

        if isinstance(video_path, list) and video_path and isinstance(video_path[0], Image.Image):
            pil_frames_input = video_path
            if self.resize_input_to:
                frames_np = [np.asarray(img) for img in pil_frames_input]
                padded, orig_hw, resized_hw, pad_xy = self._letterbox_frames(frames_np, self.resize_input_to)
                resize_meta = (orig_hw, resized_hw, pad_xy)
                pil_frames = [Image.fromarray(f) for f in padded]
                video_resource = pil_frames
                expected_frames = len(pil_frames)
            else:
                video_resource = pil_frames_input
                if expected_frames is None:
                    expected_frames = len(pil_frames_input)
        elif self.resize_input_to:
            frames_np = load_video_frames(video_path)
            padded, orig_hw, resized_hw, pad_xy = self._letterbox_frames(frames_np, self.resize_input_to)
            resize_meta = (orig_hw, resized_hw, pad_xy)
            pil_frames = [Image.fromarray(f) for f in padded]
            video_resource = pil_frames
            expected_frames = len(pil_frames)

        masks_by_obj: dict[int, dict[int, np.ndarray]] = {}
        next_global_obj_id = 1

        try:
            for prompt in prompts:
                response = self.predictor.handle_request(
                    request={
                        "type": "start_session",
                        "resource_path": video_resource,
                    }
                )
                session_id = response["session_id"]

                try:
                    self.predictor.handle_request(
                        request={
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": frame_index,
                            "text": prompt,
                        }
                    )

                    local_to_global: dict[int, int] = {}

                    for resp in self.predictor.handle_stream_request(
                        request={
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "propagation_direction": self.propagation_direction,
                            "start_frame_index": None,
                            "max_frame_num_to_track": None,
                        }
                    ):
                        fidx = int(resp["frame_index"])
                        outputs = resp["outputs"]
                        obj_ids = outputs.get("out_obj_ids")
                        masks = outputs.get("out_binary_masks")
                        if obj_ids is None or masks is None:
                            continue

                        obj_ids = obj_ids.detach().cpu().numpy() if hasattr(obj_ids, "detach") else np.asarray(obj_ids)
                        masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)

                        if masks.size == 0 or obj_ids.size == 0:
                            continue
                        if masks.ndim == 4 and masks.shape[1] == 1:
                            masks = masks[:, 0]

                        if resize_meta is not None:
                            orig_hw, resized_hw, pad_xy = resize_meta
                            masks = self._restore_masks_from_letterbox(masks, orig_hw, resized_hw, pad_xy)

                        for i, local_id in enumerate(obj_ids.tolist()):
                            mask = masks[i]
                            if mask is None or not mask.any():
                                continue
                            if local_id not in local_to_global:
                                local_to_global[local_id] = next_global_obj_id
                                next_global_obj_id += 1
                            gid = local_to_global[local_id]
                            per_obj = masks_by_obj.setdefault(gid, {})
                            if fidx in per_obj:
                                per_obj[fidx] = per_obj[fidx] | mask
                            else:
                                per_obj[fidx] = mask

                finally:
                    self.predictor.handle_request(
                        request={
                            "type": "close_session",
                            "session_id": session_id,
                        }
                    )

        finally:
            if pil_frames is not None:
                for img in pil_frames:
                    img.close()
                pil_frames.clear()
                del pil_frames
            self._release_gpu_memory()

        if not masks_by_obj:
            return {}

        if expected_frames is None:
            max_idx = max(
                (max(fdict.keys()) for fdict in masks_by_obj.values() if fdict),
                default=-1,
            )
            expected_frames = max_idx + 1

        result: dict[int, np.ndarray] = {}
        for obj_id, frame_dict in masks_by_obj.items():
            if not frame_dict:
                continue
            sample_mask = next(iter(frame_dict.values()))
            H, W = sample_mask.shape
            masks_out = np.zeros((expected_frames, H, W), dtype=bool)
            for fidx, mask in frame_dict.items():
                if fidx < expected_frames:
                    masks_out[fidx] = masks_out[fidx] | mask
            if masks_out.any():
                result[obj_id] = masks_out

        return result

    def segment(
        self,
        video_path: str,
        prompts: Iterable[str],
        frame_index: int = 0,
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        expected_frames: int | None = None,
        output_dir: str | Path | None = None,
    ) -> np.ndarray:
        prompts = self.normalize_prompts(prompts)
        if not prompts:
            return np.zeros((0, 0, 0), dtype=bool)

        video_resource: str | list[Image.Image] = video_path
        resize_meta = None
        pil_frames: list[Image.Image] | None = None
        if self.resize_input_to:
            frames = load_video_frames(video_path)
            padded_frames, orig_hw, resized_hw, pad_xy = self._letterbox_frames(frames, self.resize_input_to)
            pil_frames = [Image.fromarray(frame) for frame in padded_frames]
            video_resource = pil_frames
            expected_frames = len(pil_frames)
            resize_meta = (orig_hw, resized_hw, pad_xy)
            del frames, padded_frames

        try:
            if self.multi_prompt and len(prompts) > 1:
                merged_masks = self._segment_multi_prompt(
                    video_resource=video_resource,
                    prompts=prompts,
                    frame_index=frame_index,
                    start_frame_index=start_frame_index,
                    max_frame_num_to_track=max_frame_num_to_track,
                    expected_frames=expected_frames,
                )
            else:
                merged_masks = None
                for prompt in prompts:
                    masks = self._segment_single_prompt(
                        video_resource=video_resource,
                        prompt=prompt,
                        frame_index=frame_index,
                        start_frame_index=start_frame_index,
                        max_frame_num_to_track=max_frame_num_to_track,
                        expected_frames=expected_frames,
                    )
                    if masks.size == 0:
                        continue
                    merged_masks = masks if merged_masks is None else merged_masks | masks
        finally:
            if pil_frames is not None:
                for img in pil_frames:
                    img.close()
                pil_frames.clear()
                del pil_frames
            self._release_gpu_memory()

        if merged_masks is None or not merged_masks.any():
            return np.zeros((0, 0, 0), dtype=bool)

        if resize_meta is not None:
            orig_hw, resized_hw, pad_xy = resize_meta
            merged_masks = self._restore_masks_from_letterbox(merged_masks, orig_hw, resized_hw, pad_xy)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(merged_masks.shape[0]):
                mask_u8 = (merged_masks[idx].astype(np.uint8)) * 255
                Image.fromarray(mask_u8).save(output_dir / f"{idx:05d}.png")

        return merged_masks
