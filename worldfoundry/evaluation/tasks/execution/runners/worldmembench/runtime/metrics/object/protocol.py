# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Single-object input contract and fixed/automatic revisit selection."""

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

from ..revisit_protocol import validate_expected_revisits
from .schemas import ObjectEntityProtocol


def load_object_entity_protocol(
    item: dict, *, total_frames: int, condition_image_path: str, target_mask_path: str
) -> ObjectEntityProtocol:
    annotations = item.get("annotations") or {}
    target = annotations.get("target")
    if not isinstance(target, str) or not target.strip():
        raise DataInvalidError("annotations.target must be a non-empty object description")
    revisits = validate_expected_revisits(annotations.get("expected_revisits"))
    if total_frames < (revisits or 1) + 1:
        raise DataInvalidError("generated video is too short for the requested object revisit split")
    return ObjectEntityProtocol(
        total_frames=total_frames,
        condition_image_path=condition_image_path,
        target_mask_path=target_mask_path,
        target_description=target.strip(),
        expected_revisits=revisits,
    )
