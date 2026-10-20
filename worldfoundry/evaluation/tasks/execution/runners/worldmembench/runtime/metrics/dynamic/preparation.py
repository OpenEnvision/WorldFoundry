# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Embed per-case checklist and planned LR-hold phases in the run manifest."""

import json
from dataclasses import asdict
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)

from .checklist import parse_checklist
from .protocol import LR_HOLD_TRAJECTORY_ID, LRHoldProtocol


def prepare_for_evaluation(samples: list[dict]) -> None:
    for sample in samples:
        if sample.get("category") not in (None, "dynamic_memory"):
            continue
        sample.pop("dynamic_preparation_error", None)
        try:
            directory = Path(sample["video"]).parent
            annotations = dict(sample.get("annotations") or {})
            if "dynamic_checklist" not in annotations:
                annotations["dynamic_checklist"] = json.loads(
                    (directory / "checklist.json").read_text(encoding="utf-8")
                )
            case = parse_checklist(annotations["dynamic_checklist"])
            if sample.get("sample_id") not in (None, case.case_id):
                raise DataInvalidError("checklist case_id does not match the video case")
            annotations["dynamic_checklist"] = {
                "case_id": case.case_id,
                "questions": [asdict(question) for question in case.questions],
            }
            if "lr_hold_protocol" not in annotations:
                trajectory = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
                annotations["lr_hold_protocol"] = {
                    "trajectory_id": LR_HOLD_TRAJECTORY_ID,
                    "native_trajectory_id": sample.get("condition_id") or LR_HOLD_TRAJECTORY_ID,
                    **trajectory,
                }
            protocol = annotations["lr_hold_protocol"]
            # The declared length belongs to the planned trajectory, not to the decoded video.
            protocol = {
                "trajectory_id": protocol["trajectory_id"],
                "native_trajectory_id": protocol["native_trajectory_id"],
                "actions": protocol["actions"],
                "phase_transitions": protocol["phase_transitions"],
                "tail_hold_transitions": protocol.get("tail_hold_transitions", 0),
                "declared_output_frames": protocol.get(
                    "declared_output_frames",
                    sum(protocol["phase_transitions"]) + protocol.get("tail_hold_transitions", 0) + 1,
                ),
            }
            parsed = LRHoldProtocol.from_metadata(protocol, actual_output_frames=protocol["declared_output_frames"])
            if sample.get("condition_id") not in (None, parsed.native_trajectory_id):
                raise DataInvalidError("condition_id does not match LR-hold native trajectory")
            annotations["lr_hold_protocol"] = protocol
            sample["annotations"] = annotations
        except (DataInvalidError, EvaluationError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            sample["dynamic_preparation_error"] = {"status": "data_invalid", "message": str(error)}
