# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``uni3c`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _TEST_CASES_ROOT, _artifact, _field

UNI3C_DEMO_ROOT = _TEST_CASES_ROOT / "uni3c" / "synthetic-forward-9f-768x480"


UNI3C_DEMO_REFERENCE_IMAGE = str(UNI3C_DEMO_ROOT / "reference.jpg")


UNI3C_DEMO_RENDER_BUNDLE = str(UNI3C_DEMO_ROOT)


UNI3C_DEMO_PROMPT = (
    "A close-up of a wet white cat wearing oversized amber sunglasses in a sunlit swimming pool, "
    "while the camera smoothly pushes forward toward the cat."
)


UNI3C_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="uni3c",
    display_name="Uni3C",
    default_variant_id="camera",
    default_task_id="image-camera-control-video",
    aliases=("pcdcontroller", "alibaba-damo-academy/Uni3C"),
    variants=(
        InferenceVariantSpec(
            variant_id="camera",
            label="PCDController Camera",
            status="input-assets-required",
            checkpoints=(
                InferenceCheckpointRef(role="controller", uri="ewrfcas/Uni3C", status="official"),
                InferenceCheckpointRef(
                    role="base",
                    uri="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                    status="official",
                ),
            ),
            call_kwargs={
                "mode": "camera",
                "render_path": UNI3C_DEMO_RENDER_BUNDLE,
                "num_frames": 9,
                "max_area": 114688,
                "fps": 16,
                "seed": 0,
                "num_gpus": 1,
            },
            aliases=("default", "pcd", "pcd-controller"),
            notes=("Requires the official stage-one pre-rendered camera-control bundle.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="image-camera-control-video",
            label="Image + Camera Control Video",
            description=(
                "Run the official Uni3C stage-two PCDController path from a reference image and "
                "a pre-rendered camera-control bundle."
            ),
            aliases=("default", "camera", "image-to-video"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=UNI3C_DEMO_PROMPT,
                ),
                _field(
                    "input_path",
                    "Reference Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=UNI3C_DEMO_REFERENCE_IMAGE,
                    description="Reference image used to produce the stage-one render bundle.",
                ),
                _field(
                    "render_path",
                    "Stage-One Render Bundle",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=UNI3C_DEMO_RENDER_BUNDLE,
                    description="Directory containing render.mp4, render_mask.mp4, and cam_info.json.",
                ),
                _field("mode", "Mode", target="call_kwargs", default="camera", choices=("camera",)),
                _field("num_frames", "Frames", kind="integer", target="params", default=9),
                _field("fps", "FPS", kind="integer", target="params", default=16),
                _field("max_area", "Maximum Area", kind="integer", target="call_kwargs", default=114688),
                _field("seed", "Seed", kind="integer", target="params", default=0),
                _field("num_gpus", "GPUs", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "mode": "camera",
                "render_path": UNI3C_DEMO_RENDER_BUNDLE,
                "num_frames": 9,
                "max_area": 114688,
                "fps": 16,
                "seed": 0,
                "num_gpus": 1,
            },
        ),
    ),
    notes=(
        "Uni3C stage two does not synthesize its own camera controls; a matching stage-one render bundle is required.",
    ),
)
