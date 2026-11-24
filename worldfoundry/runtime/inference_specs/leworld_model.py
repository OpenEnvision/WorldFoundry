# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``leworld_model`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _artifact, _field

LEWORLD_MODEL_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="leworldmodel",
    display_name="LeWorldModel",
    default_variant_id="pusht-random",
    default_task_id="pusht-eval",
    aliases=("le-wm", "lewm", "leworldmodel"),
    variants=(
        InferenceVariantSpec(
            variant_id="pusht-random",
            label="PushT Random Policy",
            status="local_dataset_smoke_ready",
            checkpoints=(),
            load_kwargs={
                "model_id": "leworldmodel",
                "config_dir": "${WORLDFOUNDRY_DATA_ROOT}/models/runtime/configs/le_wm/config/eval",
                "config_name": "pusht",
                "policy": "random",
                "cache_dir": "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel",
                "dataset_name": "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5",
                "num_eval": 1,
                "eval_budget": 25,
                "goal_offset_steps": 5,
                "seed": 42,
            },
            aliases=("default", "pusht", "random"),
            notes=(
                "Uses the official PushT eval flow with a deterministic locally collected trajectory and random policy; no policy checkpoint is required.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="pusht-eval",
            label="PushT Eval",
            description="Run or preflight the official LeWorldModel PushT evaluation route.",
            aliases=("default", "official-demo", "pusht"),
            inputs=(
                _field(
                    "config_dir",
                    "Config Dir",
                    kind="path",
                    target="load_kwargs",
                    default="${WORLDFOUNDRY_DATA_ROOT}/models/runtime/configs/le_wm/config/eval",
                ),
                _field(
                    "config_name",
                    "Config Name",
                    target="load_kwargs",
                    default="pusht",
                    choices=("pusht", "cube", "tworoom", "reacher"),
                ),
                _field("policy", "Policy", target="load_kwargs", default="random"),
                _field(
                    "cache_dir",
                    "Cache Dir",
                    kind="path",
                    target="load_kwargs",
                    default="${WORLDFOUNDRY_CKPT_DIR}/leworldmodel",
                ),
                _field(
                    "dataset_name",
                    "Dataset Name",
                    target="load_kwargs",
                    default="${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5",
                ),
                _field("num_eval", "Eval Count", kind="integer", target="load_kwargs", default=1),
                _field("eval_budget", "Eval Budget", kind="integer", target="load_kwargs", default=25),
                _field("goal_offset_steps", "Goal Offset Steps", kind="integer", target="load_kwargs", default=5),
                _field("img_size", "Image Size", kind="integer", target="load_kwargs"),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("result", "json"),
                _artifact("video", "video"),
            ),
            default_call_kwargs={"plan_only": False, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "The PushT random-policy demo uses a deterministic trajectory prepared under the checkpoint root. Non-random policies still require compatible policy checkpoints.",
    ),
)
