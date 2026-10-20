# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Resolve the Environment case layout without inferring missing trajectory data."""

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

from .protocol import protocol_fields


def prepare_for_evaluation(samples: list[dict]) -> None:
    """Read trajectory.json and register DA3 paths; inference is a separate step."""
    for sample in samples:
        if sample.get("category") not in (None, "environment"):
            continue
        sample.pop("environment_preparation_error", None)
        try:
            case_dir = Path(sample["video"]).parent
            annotations = dict(sample.get("annotations") or {})
            if "environment_protocol" not in annotations:
                annotations["environment_protocol"] = json.loads(
                    (case_dir / "trajectory.json").read_text(encoding="utf-8")
                )
            annotations["environment_protocol"] = protocol_fields(annotations["environment_protocol"])
            artifacts = dict(sample.get("artifacts") or {})
            for key, filename in (("da3", "results.npz"), ("da3_scene", "scene_geometry.npz")):
                artifacts.setdefault(key, str((case_dir / "da3" / filename).resolve()))
            sample["annotations"] = annotations
            sample["artifacts"] = artifacts
        except (DataInvalidError, OSError, ValueError, KeyError, TypeError) as error:
            sample["environment_preparation_error"] = {
                "status": "data_invalid",
                "message": f"invalid Environment trajectory: {error}",
            }


def prepare_artifacts(samples: list[dict], context, metric_names) -> None:
    """Run before a shard loads scoring models, then release the DA3 subprocess."""
    from ...preparation.da3 import ensure_artifacts

    require_scene = bool({"environment_depth_fidelity", "environment_scene_reconstruction"}.intersection(metric_names))
    weight_metric = next(name for name in metric_names if name.startswith("environment_"))
    eligible = [
        sample
        for sample in samples
        if sample.get("category") in (None, "environment") and not sample.get("environment_preparation_error")
    ]
    if eligible:
        print(f"[{context.device}][DA3] 0/{len(eligible)}", flush=True)
    for index, sample in enumerate(eligible, 1):
        try:
            ensure_artifacts(sample, context, weight_metric=weight_metric, require_scene=require_scene)
        except Exception as error:
            sample["environment_preparation_error"] = {
                "status": "eval_failed",
                "message": f"DA3 preparation failed: {type(error).__name__}: {error}",
            }
            print(
                f"[{context.device}][DA3] {index}/{len(eligible)} failed: {sample['item_id']}: "
                f"{sample['environment_preparation_error']['message']}",
                flush=True,
            )
        else:
            print(f"[{context.device}][DA3] {index}/{len(eligible)} ready: {sample['item_id']}", flush=True)
