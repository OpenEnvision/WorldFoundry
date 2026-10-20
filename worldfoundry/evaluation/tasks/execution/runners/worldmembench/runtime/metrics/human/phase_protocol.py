# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Human input contract and shared revisit allocation."""

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.revisit_protocol import (
    validate_expected_revisits,
)

from .schemas import BoundingBox, HumanEntityProtocol


def load_human_entity_protocol(
    item: dict,
    *,
    total_frames: int,
    frame_width: int,
    frame_height: int,
) -> HumanEntityProtocol:
    annotations = item.get("annotations", {})
    try:
        raw_boxes = annotations["target_face_bboxes_first_frame"]
        raw_descriptions = annotations["target_descriptions"]
    except KeyError as error:
        raise DataInvalidError(f"missing annotations.{error.args[0]}") from error

    expected_revisits = validate_expected_revisits(annotations.get("expected_revisits"))
    minimum_frames = (expected_revisits or 1) + 1
    if total_frames < minimum_frames:
        raise DataInvalidError(
            f"generated video is too short for the requested revisit split; requires at least {minimum_frames} frames"
        )
    if not isinstance(raw_boxes, (list, tuple)) or not raw_boxes:
        raise DataInvalidError("annotations.target_face_bboxes_first_frame must contain at least one bbox")
    try:
        boxes = tuple(
            BoundingBox.from_value(
                value,
                field_name="annotations.target_face_bboxes_first_frame",
            )
            for value in raw_boxes
        )
    except ValueError as error:
        raise DataInvalidError(str(error)) from error
    for box in boxes:
        clipped = box.clipped(width=frame_width, height=frame_height)
        if clipped != box:
            raise DataInvalidError("target_face_bboxes_first_frame must lie inside the first frame")

    if not isinstance(raw_descriptions, (list, tuple)) or len(raw_descriptions) != len(boxes):
        raise DataInvalidError("annotations.target_descriptions must contain one description per first-frame target")
    descriptions = tuple(str(value).strip() for value in raw_descriptions)
    if any(not value for value in descriptions):
        raise DataInvalidError("annotations.target_descriptions cannot contain blanks")
    return HumanEntityProtocol(
        total_frames=total_frames,
        target_face_boxes=boxes,
        target_descriptions=descriptions,
        expected_revisits=expected_revisits,
    )
