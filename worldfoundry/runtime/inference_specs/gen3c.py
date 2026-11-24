# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``gen3c`` model family."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

GEN3C_OFFICIAL_FIXTURE = str(Path(__file__).resolve().parents[2] / "data" / "test_cases" / "gen3c" / "image.png")


# Catalog copies; canonical strings live in pipelines.gen3c.constants.
GEN3C_OFFICIAL_PROMPT = (
    "A cinematic view of a futuristic science-fiction city with coherent structures, realistic lighting, "
    "and smooth natural camera motion."
)


GEN3C_OFFICIAL_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


GEN3C_OFFICIAL_CALL_KWARGS = {
    "trajectory": "left",
    "camera_rotation": "center_facing",
    "movement_distance": 0.3,
    "negative_prompt": GEN3C_OFFICIAL_NEGATIVE_PROMPT,
    "guidance": 1.0,
    "num_steps": 35,
    "num_video_frames": 121,
    "fps": 24,
    "height": 704,
    "width": 1280,
    "seed": 1,
}


GEN3C_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="gen3c",
    display_name="GEN3C",
    default_variant_id="default",
    default_task_id="official-single-image",
    aliases=("gen-3c", "gen3c-cosmos-7b", "nvidia/GEN3C-Cosmos-7B"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            checkpoints=(InferenceCheckpointRef(role="primary", uri="gen3c", status="configured"),),
            call_kwargs=GEN3C_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "cosmos-7b"),
            notes=("Matches the upstream single-image demo command in GEN3C README.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-single-image",
            label="Official Single Image",
            description="Run the GEN3C README single-image demo with the packaged official fixture.",
            aliases=("default", "interactive-video", "official-demo", "single-image"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GEN3C_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=GEN3C_OFFICIAL_PROMPT),
                _field(
                    "negative_prompt",
                    "Negative Prompt",
                    target="negative_prompt",
                    default=GEN3C_OFFICIAL_NEGATIVE_PROMPT,
                ),
                _field(
                    "trajectory",
                    "Trajectory",
                    target="call_kwargs",
                    choices=(
                        "left",
                        "right",
                        "up",
                        "down",
                        "zoom_in",
                        "zoom_out",
                        "clockwise",
                        "counterclockwise",
                    ),
                ),
                _field(
                    "camera_rotation",
                    "Camera Rotation",
                    target="call_kwargs",
                    choices=("center_facing", "no_rotation", "trajectory_aligned"),
                ),
                _field("movement_distance", "Movement Distance", kind="number", target="call_kwargs"),
                _field("num_video_frames", "Frames", kind="integer", target="call_kwargs"),
                _field("fps", "FPS", kind="integer", target="call_kwargs"),
                _field("height", "Height", kind="integer", target="call_kwargs"),
                _field("width", "Width", kind="integer", target="call_kwargs"),
                _field("num_steps", "Steps", kind="integer", target="call_kwargs"),
                _field("guidance", "Guidance", kind="number", target="call_kwargs"),
                _field("seed", "Seed", kind="integer", target="call_kwargs"),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=GEN3C_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default input is a byte-identical copy of upstream assets/diffusion/000000.png.",),
)
