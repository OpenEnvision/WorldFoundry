# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``cameractrl`` model family."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec
from worldfoundry.core.io.paths import official_runtime_repo_path

from ._common import _artifact, _field, _first_existing_path


def _official_repo_path(repo_name: str, *parts: str) -> str:
    return str(official_runtime_repo_path(repo_name).joinpath(*parts))


CAMERACTRL_OFFICIAL_PROMPT = "A serene mountain lake at sunrise, with mist hovering over the water."


CAMERACTRL_OFFICIAL_TRAJECTORY = _first_existing_path(
    Path(__file__).resolve().parents[2] / "data" / "test_cases" / "cameractrl" / "pose_files" / "0f47577ab3441480.txt",
    _official_repo_path("CameraCtrl", "assets", "pose_files", "0f47577ab3441480.txt"),
)


CAMERACTRL_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cameractrl",
    display_name="CameraCtrl",
    default_variant_id="default",
    default_task_id="trajectory-video",
    aliases=("camera-ctrl", "hehao13/CameraCtrl"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            aliases=("official", "animatediff"),
            notes=("Uses the staged CameraCtrl, RealEstate10K LoRA, and SD1.5 paths from the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="trajectory-video",
            label="Trajectory Video",
            description="Run the official CameraCtrl text-to-video trajectory demo.",
            aliases=("default", "interactive-video", "official-demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=CAMERACTRL_OFFICIAL_PROMPT),
                _field(
                    "trajectory_file",
                    "Trajectory File",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=CAMERACTRL_OFFICIAL_TRAJECTORY,
                ),
                _field("frames", "Frames", kind="integer", target="params", default=16),
                _field("fps", "FPS", kind="integer", target="params", default=8),
                _field("height", "Height", kind="integer", target="params", default=256),
                _field("width", "Width", kind="integer", target="params", default=384),
                _field("steps", "Steps", kind="integer", target="params", default=25),
                _field("guidance_scale", "Guidance", kind="number", target="params", default=14.0),
                _field("seed", "Seed", kind="integer", target="params", default=42),
                _field(
                    "original_pose_width", "Original Pose Width", kind="integer", target="call_kwargs", default=1280
                ),
                _field(
                    "original_pose_height", "Original Pose Height", kind="integer", target="call_kwargs", default=720
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "num_frames": 16,
                "fps": 8,
                "height": 256,
                "width": 384,
                "num_inference_steps": 25,
                "guidance_scale": 14.0,
                "seed": 42,
                "trajectory_file": CAMERACTRL_OFFICIAL_TRAJECTORY,
                "original_pose_width": 1280,
                "original_pose_height": 720,
            },
        ),
    ),
    notes=("The upstream reference videos are not runtime inputs for this wrapper; use the paired pose file.",),
)
