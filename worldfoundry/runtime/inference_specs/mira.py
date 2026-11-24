# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``mira`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec
from worldfoundry.core.io.paths import checkpoint_root_path, local_data_root_path

from ._common import _artifact, _field

MIRA_DEFAULT_CHECKPOINT = str(checkpoint_root_path("mira"))


MIRA_DEFAULT_DATASET = str(local_data_root_path() / "rocket-science" / "test")


MIRA_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="mira",
    display_name="MIRA",
    default_variant_id="local-checkpoint",
    default_task_id="rocket-science-rollout",
    aliases=("mira-wm", "multiplayer-interactive-world-model"),
    variants=(
        InferenceVariantSpec(
            variant_id="local-checkpoint",
            label="Local MIRA Checkpoint",
            status="requires_local_checkpoint_and_dataset",
            load_kwargs={"model_id": "mira"},
            aliases=("default", "offline", "rocket-science"),
            notes=("Uses a local MIRA run directory and local Rocket Science test split.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="rocket-science-rollout",
            label="Rocket Science Rollout",
            description="Generate one offline, action-conditioned MIRA rollout from a Rocket Science clip.",
            aliases=("default", "offline-rollout", "game-rollout"),
            inputs=(
                _field(
                    "checkpoint_path",
                    "Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=MIRA_DEFAULT_CHECKPOINT,
                    description="MIRA checkpoint.pth file or run directory.",
                ),
                _field(
                    "dataset_path",
                    "Rocket Science Split",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=MIRA_DEFAULT_DATASET,
                    description="Local split directory containing index.json, or the index.json path.",
                ),
                _field(
                    "actions",
                    "Action Timeline",
                    kind="json",
                    target="load_kwargs",
                    default=(),
                    description=('JSON segments such as [{"player":0,"keys":["W","D"],"start":0,"frames":20}].'),
                ),
                _field("seed", "Seed", kind="integer", target="load_kwargs", default=42),
                _field("clip_index", "Dataset Clip", kind="integer", target="load_kwargs", default=0),
                _field("n_context_frames", "Context Frames", kind="integer", target="load_kwargs", default=38),
                _field(
                    "num_unrolled_frames",
                    "Generated Latent Frames",
                    kind="integer",
                    target="load_kwargs",
                    default=20,
                ),
                _field("n_diffusion_steps", "Diffusion Steps", kind="integer", target="load_kwargs", default=10),
                _field(
                    "schedule_type",
                    "Schedule",
                    target="load_kwargs",
                    default="linear",
                    choices=("linear", "linear_quadratic"),
                ),
                _field("noise_level", "Cache Noise", kind="number", target="load_kwargs", default=0.0),
                _field("compile", "Compile Model", kind="boolean", target="load_kwargs", default=False),
                _field("overlay_actions", "Overlay Actions", kind="boolean", target="load_kwargs", default=True),
                _field("generated_only", "Generated Frames Only", kind="boolean", target="load_kwargs", default=False),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("runtime_plan", "manifest", required=True),
                _artifact("result", "manifest"),
            ),
            default_call_kwargs={"plan_only": False, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "Inference-only integration; no MIRA training or metric entrypoints are exposed.",
        "The isolated runtime mirrors upstream's torch 2.8, CPU TorchCodec 0.7, and FFmpeg 7 pins.",
    ),
)
