"""Detector agent for entity detection in the pipeline.

DetectorAgent detects dynamic entities from frames using Qwen3-VL,
then segments them with SAM3 and produces cropped images.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager

import cv2
import numpy as np
import torch

from worldfoundry.base_models.perception_core.general_perception.qwen3_vl_entity import (
    Qwen3VLEntityExtractor,
)
from worldfoundry.base_models.perception_core.segment.sam3.video_segmenter import (
    Sam3VideoSegmenter,
)

from .event_types import EntityDetectionResult


def _unwrap_torch_module(obj):
    """Extract the underlying torch module if wrapped."""
    if obj is None:
        return None
    if hasattr(obj, "model"):
        return obj.model
    return obj


def _move_to_device(obj, device: str) -> None:
    """Move a torch module to a device if possible."""
    module = _unwrap_torch_module(obj)
    if module is None:
        return
    if hasattr(module, "to"):
        module.to(device)


@contextmanager
def _offload_scope(obj, enable: bool, device: str):
    """Context manager for CPU offload behavior."""
    if not enable:
        yield
        return
    _move_to_device(obj, device)
    try:
        yield
    finally:
        _move_to_device(obj, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _crop_entity_on_black_square(
    frame: np.ndarray,
    mask: np.ndarray,
    padding: int = 10,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """Crop entity from frame using mask, place on black square background.

    Args:
        frame: Original frame (H, W, 3), uint8.
        mask: Binary mask for the entity (H, W), bool or uint8.
        padding: Padding around bounding box.

    Returns:
        Tuple of (cropped_image, bbox, cropped_mask).
        cropped_image: Entity on black square background (size, size, 3), uint8.
        bbox: Bounding box [x1, y1, x2, y2] in original frame.
        cropped_mask: Cropped SAM mask aligned with cropped_image, uint8 {0,1}.
    """
    # Find bounding box from mask.
    mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask

    coords = cv2.findNonZero(mask_uint8)
    if coords is None:
        # Empty mask, return empty result.
        return (
            np.zeros((64, 64, 3), dtype=np.uint8),
            (0, 0, 0, 0),
            np.zeros((64, 64), dtype=np.uint8),
        )

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding.
    H, W = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    # Crop region.
    cropped_frame = frame[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]

    # Create black background square.
    crop_h, crop_w = cropped_frame.shape[:2]
    size = max(crop_h, crop_w)
    black_square = np.zeros((size, size, 3), dtype=np.uint8)
    black_square_mask = np.zeros((size, size), dtype=np.uint8)

    # Center the cropped entity on black square.
    offset_y = (size - crop_h) // 2
    offset_x = (size - crop_w) // 2

    # Apply SAM mask: only copy foreground pixels, leave rest black.
    cropped_mask_bool = cropped_mask > 0
    region = black_square[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    region[cropped_mask_bool] = cropped_frame[cropped_mask_bool]
    region_mask = black_square_mask[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    region_mask[cropped_mask_bool] = 1

    return black_square, (x1, y1, x2, y2), black_square_mask


_ENTITY_DETECT_PROMPT = """\
List all LIVING BEINGS and VEHICLES visible in this image.

INCLUDE (even if stationary, sitting, resting, or parked):
- People (any human, including hands/arms/feet)
- Animals (dog, cat, bird, horse, etc.)
- Vehicles (car, truck, bus, motorcycle, bicycle, scooter, etc.)

EXCLUDE: furniture, appliances, fixtures, lamps, plants, decorations, \
buildings, structures, signs, sky, water, food, drinks, bags, boxes, \
and all other non-living/non-vehicle objects.

RULES:
1. Output CATEGORIES, not individual instances
2. Use AT MOST 4 categories total
3. For any human, ALWAYS write exactly: person
4. Include them even if they are NOT moving — what matters is whether \
they are a living creature or a vehicle
5. Keep items short (1-3 words)

OUTPUT FORMAT:
Nothing
OR numbered list:
1) person
2) dog"""


class DetectorAgent:
    """Detector for dynamic entity detection and segmentation.

    This agent performs:
    1. Entity detection using Qwen3-VL
    2. Static entity filtering (keep only dynamic/movable entities)
    3. Segmentation using SAM3
    4. Cropping entities onto black square backgrounds
    """

    def __init__(
        self,
        qwen_model: Qwen3VLEntityExtractor,
        sam3_model: Sam3VideoSegmenter,
        device: str,
        detect_prompt: str,
        cpu_offload_qwen: bool = False,
        cpu_offload_sam3: bool = False,
    ) -> None:
        if qwen_model is None:
            raise ValueError("qwen_model must be provided")
        if sam3_model is None:
            raise ValueError("sam3_model must be provided")

        self.qwen_model = qwen_model
        self.sam3_model = sam3_model
        self.device = device
        if not detect_prompt:
            raise ValueError("detect_prompt must be provided (set in system_config.yaml)")
        self.detect_prompt = detect_prompt
        self.cpu_offload_qwen = cpu_offload_qwen
        self.cpu_offload_sam3 = cpu_offload_sam3

    def detect(
        self,
        frame_path: str,
        frame: np.ndarray,
        frame_index: int = 0,
        preset_entity_names: list[str] | None = None,
    ) -> list[EntityDetectionResult]:
        """Detect and segment entities from a frame.

        Args:
            frame_path: Path to the frame image or video (for Qwen/SAM3 input).
                        If empty, frame numpy array will be saved to a temp file.
            frame: Frame as numpy array (H, W, 3), uint8 RGB.
            frame_index: Frame index if frame_path is a video.
            preset_entity_names: If provided, skip Qwen detection and use these
                entity names directly for SAM3 segmentation.

        Returns:
            List of EntityDetectionResult, one per detected entity.
        """
        # If frame_path is empty, save frame to a temporary file
        temp_file = None
        if not frame_path:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
            temp_path = temp_file.name
            temp_file.close()
            # Convert RGB to BGR for cv2.imwrite
            cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_path = temp_path
            frame_index = 0  # Single image, use index 0

        if preset_entity_names is not None:
            # Skip Qwen: use preset entities directly.
            entity_names = list(preset_entity_names)
        else:
            # Detect living beings / vehicles using a single Qwen call.
            with _offload_scope(self.qwen_model, self.cpu_offload_qwen, self.device):
                entity_names, _raw = self.qwen_model.extract(frame_path, prompt=_ENTITY_DETECT_PROMPT)

        if not entity_names:
            return []

        # 2. Segment each entity using SAM3.
        results = []
        with _offload_scope(self.sam3_model, self.cpu_offload_sam3, self.device):
            for entity_name in entity_names:
                # SAM3 segment — per-instance masks for instance-level dedup.
                instance_masks = self.sam3_model.segment_instances(
                    video_path=frame_path,
                    prompt=entity_name,
                    frame_index=frame_index,
                    expected_frames=1,
                )

                for inst_mask in instance_masks:
                    inst_mask = np.asarray(inst_mask)
                    if inst_mask.ndim != 2:
                        raise RuntimeError(
                            f"SAM instance mask for '{entity_name}' must be 2D, got shape={inst_mask.shape}"
                        )
                    h_img, w_img = frame.shape[:2]
                    if inst_mask.shape != (h_img, w_img):
                        # Align SAM output to the exact frame used for crop.
                        inst_mask = cv2.resize(
                            inst_mask.astype(np.uint8),
                            (w_img, h_img),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    inst_mask = (inst_mask > 0).astype(np.uint8)
                    cropped_image, bbox, cropped_mask = _crop_entity_on_black_square(
                        frame,
                        inst_mask,
                    )
                    results.append(
                        EntityDetectionResult(
                            name=entity_name.strip().lower(),
                            mask=inst_mask,
                            cropped_image=cropped_image,
                            bbox=bbox,
                            cropped_mask=cropped_mask,
                        )
                    )

        return results
        # finally:
        #     # Clean up temporary file
        #     if temp_file is not None:
        #         import os
        #         try:
        #             os.unlink(temp_path)
        #         except OSError:
        #             pass
