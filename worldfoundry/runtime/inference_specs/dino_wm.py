# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``dino_wm`` model family."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.io.paths import checkpoint_root_candidates

from ._common import _RUNTIME_CONFIGS_ROOT, _artifact, _field, _first_existing_path

DINO_WM_CHECKPOINT_BASE = str(
    Path(
        os.environ.get("WORLDFOUNDRY_DINO_WM_CKPT_BASE") or _first_existing_path(*checkpoint_root_candidates("dino_wm"))
    )
    .expanduser()
    .resolve()
)


DINO_WM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="dino-wm",
    display_name="DINO-WM",
    default_variant_id="wall-official",
    default_task_id="wall-planning",
    aliases=("dino_wm", "dino-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id="wall-official",
            label="Wall Planning",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="ckpt_base_path",
                    uri=DINO_WM_CHECKPOINT_BASE,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_id": "dino-wm",
                "config": str(_RUNTIME_CONFIGS_ROOT / "dino_wm" / "conf" / "plan_wall.yaml"),
                "ckpt_base_path": DINO_WM_CHECKPOINT_BASE,
                "model_name": "wall_single",
                "model_epoch": "latest",
            },
            aliases=("default", "wall", "official-demo"),
            notes=(
                "The released wall checkpoint lives at outputs/wall_single/checkpoints/model_latest.pth under ckpt_base_path.",
                "The wall_single integration passed real GPU inference with official trajectories; provide a local checkpoint and DATASET_DIR for execution.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="wall-planning",
            label="Wall Planning",
            description="Run or preflight the official DINO-WM wall planning/evaluation demo.",
            aliases=("default", "official-demo", "plan-wall"),
            inputs=(
                _field(
                    "config",
                    "Config",
                    kind="path",
                    target="load_kwargs",
                    default=str(_RUNTIME_CONFIGS_ROOT / "dino_wm" / "conf" / "plan_wall.yaml"),
                ),
                _field(
                    "ckpt_base_path",
                    "Checkpoint Base Path",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=DINO_WM_CHECKPOINT_BASE,
                ),
                _field(
                    "model_name",
                    "Model Name",
                    target="load_kwargs",
                    required=True,
                    default="wall_single",
                    choices=("wall_single", "pusht", "point_maze"),
                ),
                _field("model_epoch", "Model Epoch", target="load_kwargs", default="latest"),
                _field("seed", "Seed", kind="integer", target="load_kwargs", default=99),
                _field("n_evals", "Eval Count", kind="integer", target="load_kwargs", default=1),
                _field("goal_source", "Goal Source", target="load_kwargs", default="dset"),
                _field("goal_H", "Goal Horizon", kind="integer", target="load_kwargs", default=1),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("result", "json"),
            ),
            default_call_kwargs={"plan_only": True, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "Stage the official checkpoint under ckpt_base_path/outputs/wall_single and set DATASET_DIR to the extracted official wall trajectories.",
        "Set WORLDFOUNDRY_DINO_WM_CKPT_BASE before importing the catalog to select a nonstandard checkpoint root; a published smoke run used a smaller planner budget than plan_wall.yaml defaults.",
    ),
)
