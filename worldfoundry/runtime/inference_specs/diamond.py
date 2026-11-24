# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``diamond`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _WORKSPACE_ROOT, GENERIC_IMAGE_FIXTURE, _artifact, _field

DIAMOND_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="diamond",
    display_name="DIAMOND",
    default_variant_id="atari-pong-pretrained",
    default_task_id="runtime-plan",
    aliases=("diamond-atari", "diamond-wm"),
    variants=(
        InferenceVariantSpec(
            variant_id="atari-pong-pretrained",
            label="Atari Pong Pretrained",
            status="local_checkpoint_confirmed",
            checkpoints=(
                InferenceCheckpointRef(
                    role="pretrained_dir",
                    uri=str(_WORKSPACE_ROOT / "ckpt" / "diamond"),
                    status="local_confirmed",
                ),
            ),
            load_kwargs={
                "model_id": "diamond",
                "pretrained": True,
                "pretrained_game": "Pong",
                "pretrained_dir": str(_WORKSPACE_ROOT / "ckpt" / "diamond"),
                "fps": 15,
                "size": 640,
                "num_steps_initial_collect": 1000,
            },
            aliases=("default", "pong", "pretrained"),
            notes=(
                "Matches the official DIAMOND Atari pretrained play route; Studio defaults to plan-only execution.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="runtime-plan",
            label="Runtime Plan",
            description="Prepare the official DIAMOND play.py Atari route and report runtime requirements.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt"),
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=False,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field(
                    "interactions", "Interactions", kind="interaction_tokens", target="params", default=("forward",)
                ),
                _field("frames", "Frames", kind="integer", target="params", default=16),
                _field("fps", "FPS", kind="integer", target="params", default=15),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field("pretrained", "Pretrained", kind="boolean", target="load_kwargs", default=True),
                _field(
                    "pretrained_game",
                    "Pretrained Game",
                    target="load_kwargs",
                    default="Pong",
                    choices=("Pong", "Breakout", "Seaquest", "Qbert", "MsPacman", "Freeway"),
                ),
                _field(
                    "pretrained_dir",
                    "Pretrained Dir",
                    kind="path",
                    target="load_kwargs",
                    default=str(_WORKSPACE_ROOT / "ckpt" / "diamond"),
                ),
                _field(
                    "num_steps_initial_collect",
                    "Initial Collect Steps",
                    kind="integer",
                    target="load_kwargs",
                    default=1000,
                ),
                _field("size", "Window Size", kind="integer", target="load_kwargs", default=640),
                _field("record", "Record", kind="boolean", target="load_kwargs", default=False),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("generated_world", "generated_world"),
            ),
            default_call_kwargs={"plan_only": True, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "Set call_kwargs.plan_only=false for real Studio/API execution after confirming the Atari runtime and GPU budget.",
    ),
)
