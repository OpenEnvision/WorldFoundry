# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Read target camera inputs and prepare required predicted poses before scoring."""

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

from .metric import load_inputs, required_estimators


def prepare_for_evaluation(samples, *, required=False):
    for sample in samples:
        sample.pop("camera_preparation_error", None)
        try:
            directory = Path(sample["video"]).parent
            annotations = dict(sample.get("annotations") or {})
            artifacts = dict(sample.get("artifacts") or {})
            if "camera_motion_type" not in annotations:
                source = directory / "trajectory.json"
                if source.is_file():
                    value = json.loads(source.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        raise DataInvalidError("trajectory.json must be an object")
                    if "camera_motion_type" in value:
                        annotations["camera_motion_type"] = value["camera_motion_type"]
            if "camera_motion_type" not in annotations:
                if required or artifacts.get("target_poses") or (directory / "target_poses.npz").exists():
                    raise DataInvalidError("Camera evaluation requires camera_motion_type in trajectory.json")
                continue
            if annotations["camera_motion_type"] not in ("rotation", "translation", "mixed"):
                raise DataInvalidError("camera_motion_type must be rotation, translation or mixed")
            artifacts.setdefault("target_poses", str(directory / "target_poses.npz"))
            artifacts.setdefault("da3", str(directory / "da3/results.npz"))
            if annotations["camera_motion_type"] != "rotation":
                artifacts.setdefault("vipe", str(directory / "vipe/results.npz"))
            sample.update(annotations=annotations, artifacts=artifacts)
        except (DataInvalidError, OSError, ValueError, KeyError, TypeError) as error:
            sample["camera_preparation_error"] = {"status": "data_invalid", "message": str(error)}


def prepare_artifacts(samples, context, metric_names):
    eligible = [
        sample
        for sample in samples
        if not sample.get("camera_preparation_error")
        and (
            (sample.get("annotations") or {}).get("camera_motion_type") == "mixed"
            or "camera_" + str((sample.get("annotations") or {}).get("camera_motion_type")) in metric_names
        )
    ]
    if not eligible:
        return
    from ...preparation.da3 import ensure_artifacts as da3
    from ...preparation.vipe import ensure_artifacts as vipe

    reader = VideoReader()
    print(f"[{context.device}][Camera poses] 0/{len(eligible)}", flush=True)
    for index, sample in enumerate(eligible, 1):
        try:
            load_inputs(sample, reader)
        except Exception as error:
            sample["camera_preparation_error"] = {
                "status": "data_invalid" if isinstance(error, DataInvalidError) else "model_failure",
                "message": str(error),
            }
        else:
            motion = sample["annotations"]["camera_motion_type"]
            sample["camera_artifact_errors"] = {}
            for name in required_estimators(motion):
                if Path(sample["artifacts"][name]).is_file():
                    continue
                try:
                    if name == "da3":
                        da3(
                            sample,
                            context,
                            weight_metric="camera_rotation" if motion == "rotation" else "camera_translation",
                            require_scene=False,
                        )
                    else:
                        vipe(sample, context)
                except Exception as error:
                    sample["camera_artifact_errors"][name] = f"{type(error).__name__}: {error}"
        failed = sample.get("camera_preparation_error") or sample.get("camera_artifact_errors")
        print(
            f"[{context.device}][Camera poses] {index}/{len(eligible)} "
            f"{'failed' if failed else 'ready'}: {sample['item_id']}",
            flush=True,
        )
