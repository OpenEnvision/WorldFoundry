# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``starwm`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _RUNTIME_CONFIGS_ROOT, _WORKSPACE_ROOT, _artifact, _field

STARWM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="starwm",
    display_name="StarWM",
    default_variant_id="offline-client",
    default_task_id="starcraft-offline-client",
    aliases=("star-wm", "starcraft-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id="offline-client",
            label="Offline Client",
            status="requires_endpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="model_dir",
                    uri=str(_WORKSPACE_ROOT / "ckpt" / "StarWM"),
                    status="local_confirmed",
                ),
            ),
            load_kwargs={
                "model_id": "starwm",
                "input_file": str(_RUNTIME_CONFIGS_ROOT / "starwm" / "data" / "wm_test_horizon5_1traj.json"),
                "mode": "nothink",
                "api_base": "http://localhost:12000",
                "api_key": "sk-11223344",
                "served_model_id": "StarWM-demo",
                "max_workers": 8,
            },
            aliases=("default", "official-demo", "nothink"),
            notes=("Uses the official offline inference client against an OpenAI-compatible StarWM endpoint.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="starcraft-offline-client",
            label="StarCraft Offline Client",
            description="Run or preflight the official StarWM offline client prompt fixture.",
            aliases=("default", "official-demo", "starcraft"),
            inputs=(
                _field(
                    "input_file",
                    "Input File",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=str(_RUNTIME_CONFIGS_ROOT / "starwm" / "data" / "wm_test_horizon5_1traj.json"),
                ),
                _field("mode", "Mode", target="load_kwargs", default="nothink", choices=("nothink", "think")),
                _field("api_base", "API Base", target="load_kwargs", default="http://localhost:12000"),
                _field("api_key", "API Key", target="load_kwargs", default="sk-11223344"),
                _field(
                    "served_model_id", "Served Model ID", target="load_kwargs", required=True, default="StarWM-demo"
                ),
                _field("max_workers", "Max Workers", kind="integer", target="load_kwargs", default=8),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("predictions", "jsonl"),
            ),
            default_call_kwargs={"plan_only": True, "timeout_seconds": 21600},
        ),
    ),
    notes=("Set call_kwargs.plan_only=false only after a StarWM/vLLM OpenAI-compatible endpoint is running.",),
)
