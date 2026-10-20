# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Separate rotation and translation RPE, preserving the reference estimator policy."""

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.aggregation import (
    aggregate_video_summary,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import (
    ItemResult,
    ResultStatus,
    UnitResult,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader


def required_estimators(motion_type):
    return ("da3",) if motion_type == "rotation" else ("da3", "vipe")


def load_inputs(sample, reader):
    from .pose_io import load_trajectory
    from .rpe import rotation_path_length_deg
    from .trajectory import canonicalize_target

    motion = sample["annotations"]["camera_motion_type"]
    try:
        target = load_trajectory(sample["artifacts"]["target_poses"], "target")
        target, path_length = canonicalize_target(target, motion)
        rotation_path = rotation_path_length_deg(target) if motion in ("rotation", "mixed") else None
        if rotation_path is not None and rotation_path <= 1e-12:
            raise ValueError("rotation target must have a non-zero rotation path")
    except (KeyError, ValueError) as error:
        raise DataInvalidError(str(error)) from error
    info = reader.probe(sample["video"])
    if info.total_frames != target.num_poses:
        raise EvaluationError(f"video/target frame count mismatch: {info.total_frames} != {target.num_poses}")
    return target, path_length, rotation_path


class _CameraMetric(Metric):
    motion = ""
    score_name = ""

    def __init__(self, *, reader=None):
        self.reader = reader or VideoReader()

    def evaluate(self, sample, context):
        from .pose_io import load_trajectory
        from .rpe import compute_rpe
        from .schemas import PoseTrajectory
        from .trajectory import umeyama_align

        if error := sample.get("camera_preparation_error"):
            return [UnitResult("video", status=ResultStatus(error["status"]), error=error["message"])]
        motion = (sample.get("annotations") or {}).get("camera_motion_type")
        if motion not in (self.motion, "mixed"):
            return [UnitResult("video", status=ResultStatus.NOT_APPLICABLE)]
        try:
            target, path_length, rotation_path = load_inputs(sample, self.reader)
        except DataInvalidError:
            raise
        except Exception as error:
            return [UnitResult("video", status=ResultStatus.MODEL_FAILURE, error=str(error))]

        def analyze():
            results = {}
            for name in required_estimators(motion):
                preparation_error = (sample.get("camera_artifact_errors") or {}).get(name)
                try:
                    if preparation_error:
                        raise EvaluationError(preparation_error)
                    trajectory = load_trajectory(sample["artifacts"][name], name)
                    if trajectory.num_poses != target.num_poses:
                        raise EvaluationError(
                            f"{name} pose/video count mismatch: {trajectory.num_poses} != {target.num_poses}"
                        )
                    scale = None
                    if motion != "rotation":
                        alignment = umeyama_align(trajectory, target)
                        scale = alignment.scale
                        trajectory = PoseTrajectory(alignment.poses, source=name)
                    statistics = compute_rpe(target, trajectory, motion)
                    results[name] = {
                        "status": "ok",
                        "num_poses": trajectory.num_poses,
                        "alignment_scale": scale,
                        "rotation_rpe_mean_deg": statistics.rotation_mean_deg,
                        "rotation_rpe_path": statistics.rotation_path(rotation_path)
                        if rotation_path is not None
                        else None,
                        "translation_rpe_path": statistics.translation_path,
                    }
                except Exception as error:
                    results[name] = {"status": "eval_failed", "error": f"{type(error).__name__}: {error}"}
            return results

        results = context.get_cached(("camera:rpe", sample["item_id"]), analyze)
        metadata = {
            "camera_motion_type": motion,
            "num_frames": target.num_poses,
            "num_rpe_steps": target.num_poses - 1,
            "target_path_length": path_length,
            "target_rotation_path_deg": rotation_path,
            "estimators": results,
            "direction": "lower_is_better",
        }
        failures = [value["error"] for value in results.values() if value["status"] != "ok"]
        if failures:
            return [UnitResult("video", status=ResultStatus.EVAL_FAILED, error="; ".join(failures), metadata=metadata)]
        value = (
            results["da3"]["rotation_rpe_path"]
            if self.motion == "rotation"
            else (results["da3"]["translation_rpe_path"] + results["vipe"]["translation_rpe_path"]) / 2.0
        )
        return [UnitResult("video", unit_type="video", scores={self.score_name: value}, metadata=metadata)]

    def aggregate_item(self, sample, units):
        unit = units[0]
        return ItemResult(
            sample["item_id"],
            scores=dict(unit.scores),
            num_units=1,
            status=unit.status,
            error=unit.error,
            metadata=dict(unit.metadata),
        )

    def aggregate_summary(self, items):
        summary = aggregate_video_summary(items)
        summary.metadata["direction"] = "lower_is_better"
        return summary


class CameraRotationMetric(_CameraMetric):
    name = "camera_rotation"
    motion = "rotation"
    score_name = "camera_rotation_rpe_path"
    description = "DA3 adjacent rotation RPE normalized by target rotation path; lower is better."


class CameraTranslationMetric(_CameraMetric):
    name = "camera_translation"
    motion = "translation"
    score_name = "camera_translation_rpe"
    description = "Mean DA3/ViPE path-normalized translation RPE after Sim(3) alignment; lower is better."
