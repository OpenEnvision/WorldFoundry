# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``neoverse`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.io.paths import hfd_root_path

from ._common import _TEST_CASES_ROOT, _artifact, _field, _first_existing_path

NEOVERSE_DATA_ROOT = _TEST_CASES_ROOT / "neoverse"


NEOVERSE_CHECKPOINT = str(hfd_root_path("Yuppie1204--NeoVerse"))


NEOVERSE_OFFICIAL_FIXTURE = _first_existing_path(
    NEOVERSE_DATA_ROOT / "videos" / "robot.mp4",
)


NEOVERSE_OFFICIAL_PROMPT = "A two-arm robot assembles parts in front of a table."


NEOVERSE_PREDEFINED_CHOICES = (
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "move_left",
    "move_right",
    "push_in",
    "pull_out",
    "boom_up",
    "boom_down",
    "orbit_left",
    "orbit_right",
    "static",
)


NEOVERSE_OFFICIAL_LOAD_KWARGS = {
    "height": 336,
    "width": 560,
    "num_inference_steps": 4,
    "cfg_scale": 1.0,
    "disable_lora": False,
    # The 14B Wan backbone and WorldMirror reconstructor do not fit together
    # with the official 81-frame reconstruction activations on an 80GB GPU.
    # Keep inactive components CPU-resident; this does not change sampling.
    "enable_vram_management": True,
}


NEOVERSE_OFFICIAL_CALL_KWARGS = {
    "predefined_trajectory": "tilt_up",
    "num_frames": 81,
    "trajectory_mode": "relative",
    "angle": 15,
    "distance": 0,
    "orbit_radius": 0,
    "zoom_ratio": 1.0,
    "alpha_threshold": 1.0,
    "seed": 42,
    "use_first_frame": True,
    "static_scene": False,
    "num_inference_steps": 4,
    "cfg_scale": 1.0,
}


NEOVERSE_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="neoverse",
    display_name="NeoVerse",
    default_variant_id="default",
    default_task_id="official-tilt-up-video",
    aliases=("Yuppie1204/NeoVerse", "neoverse-4d"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="requires_local_checkpoint",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=NEOVERSE_CHECKPOINT,
                    status="local_required",
                ),
            ),
            load_kwargs={
                "model_path": NEOVERSE_CHECKPOINT,
                **NEOVERSE_OFFICIAL_LOAD_KWARGS,
            },
            call_kwargs=NEOVERSE_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "tilt-up"),
            notes=("Matches the NeoVerse README tilt_up CLI example with fast LoRA defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-tilt-up-video",
            label="Official Tilt Up Video",
            description="Run the NeoVerse README tilt_up demo using the packaged robot video.",
            aliases=("default", "official-demo", "interactive-video", "video-to-world"),
            inputs=(
                _field(
                    "video",
                    "Input Video",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=NEOVERSE_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=NEOVERSE_OFFICIAL_PROMPT),
                _field(
                    "predefined_trajectory",
                    "Trajectory",
                    target="call_kwargs",
                    default="tilt_up",
                    choices=NEOVERSE_PREDEFINED_CHOICES,
                ),
                _field("interactions", "Actions", kind="interaction_tokens", target="params"),
                _field("trajectory_file", "Trajectory File", kind="path", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=81),
                _field("height", "Height", kind="integer", target="load_kwargs", default=336),
                _field("width", "Width", kind="integer", target="load_kwargs", default=560),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=4),
                _field("cfg_scale", "CFG Scale", kind="number", target="call_kwargs", default=1.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field(
                    "trajectory_mode",
                    "Trajectory Mode",
                    target="call_kwargs",
                    default="relative",
                    choices=("relative", "global"),
                ),
                _field("angle", "Angle", kind="number", target="call_kwargs", default=15),
                _field("distance", "Distance", kind="number", target="call_kwargs", default=0),
                _field("orbit_radius", "Orbit Radius", kind="number", target="call_kwargs", default=0),
                _field("zoom_ratio", "Zoom Ratio", kind="number", target="call_kwargs", default=1.0),
                _field("alpha_threshold", "Alpha Threshold", kind="number", target="call_kwargs", default=1.0),
                _field("use_first_frame", "Use First Frame", kind="boolean", target="call_kwargs", default=True),
                _field("static_scene", "Static Scene", kind="boolean", target="call_kwargs", default=False),
                _field("disable_lora", "Disable LoRA", kind="boolean", target="load_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=NEOVERSE_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=(
        "The upstream checkout in this workspace does not include examples/videos, so the default video falls back to WorldFoundry packaged test_cases/neoverse.",
    ),
)
