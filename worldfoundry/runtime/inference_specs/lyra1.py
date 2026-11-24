# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``lyra1`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

LYRA1_STATIC_OFFICIAL_FIXTURE = "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/lyra/Lyra-1/00172.png"


LYRA1_STATIC_OFFICIAL_CALL_KWARGS = {
    "mode": "static",
    "trajectory": "zoom_in",
    "num_video_frames": 121,
    "fps": 24,
    "height": 704,
    "width": 1280,
    "seed": 1,
    "num_steps": 35,
    "guidance": 1.0,
    "num_gpus": 1,
    "movement_distance": 0.3,
    "camera_rotation": "center_facing",
    "multi_trajectory": True,
    "total_movement_distance_factor": 1.0,
    "foreground_masking": True,
    "filter_points_threshold": 0.05,
    "disable_prompt_encoder": True,
    "offload_diffusion_transformer": True,
    "offload_tokenizer": True,
    "offload_text_encoder_model": True,
    "offload_prompt_upsampler": True,
    "offload_guardrail_models": True,
    "disable_guardrail": True,
    "execute": True,
    "show_progress": True,
}


LYRA1_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="lyra-1",
    display_name="Lyra-1",
    default_variant_id="default",
    default_task_id="official-static-sdg",
    aliases=("lyra1", "nvidia/lyra-1"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="checkpoint_bundle_root",
                    uri="${WORLDFOUNDRY_CKPT_DIR}",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="gen3c_transformer",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/nvidia--GEN3C-Cosmos-7B",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="cosmos_tokenizer",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="text_encoder",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b",
                    status="required",
                ),
            ),
            load_kwargs={"model_id": "lyra-1"},
            call_kwargs=LYRA1_STATIC_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "static-sdg"),
            notes=("Matches the upstream Lyra-1 scripts/bash/static_sdg.sh demo input and SDG defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-static-sdg",
            label="Official Static SDG",
            description="Run the Lyra-1 static single-image SDG demo from the official repository.",
            aliases=("default", "interactive-video", "official-demo", "static"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=LYRA1_STATIC_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=""),
                _field("mode", "Mode", target="call_kwargs", default="static", choices=("static",)),
                _field(
                    "trajectory",
                    "Trajectory",
                    target="call_kwargs",
                    default="zoom_in",
                    choices=("left", "right", "up", "down", "zoom_in", "zoom_out"),
                ),
                _field(
                    "camera_rotation",
                    "Camera Rotation",
                    target="call_kwargs",
                    default="center_facing",
                    choices=("center_facing", "no_rotation", "trajectory_aligned"),
                ),
                _field("movement_distance", "Movement Distance", kind="number", target="call_kwargs", default=0.3),
                _field("multi_trajectory", "Multi Trajectory", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "total_movement_distance_factor",
                    "Movement Factor",
                    kind="number",
                    target="call_kwargs",
                    default=1.0,
                ),
                _field("num_video_frames", "Frames", kind="integer", target="call_kwargs", default=121),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
                _field("num_steps", "Steps", kind="integer", target="call_kwargs", default=35),
                _field("guidance", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=1),
                _field("num_gpus", "GPUs", kind="integer", target="call_kwargs", default=1),
                _field("foreground_masking", "Foreground Masking", kind="boolean", target="call_kwargs", default=True),
                _field("filter_points_threshold", "Point Filter", kind="number", target="call_kwargs", default=0.05),
                _field(
                    "disable_prompt_encoder",
                    "Disable Prompt Encoder",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field("disable_guardrail", "Disable Guardrail", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "offload_diffusion_transformer",
                    "Offload Transformer",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field("offload_tokenizer", "Offload Tokenizer", kind="boolean", target="call_kwargs", default=True),
                _field(
                    "offload_text_encoder_model",
                    "Offload Text Encoder",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field(
                    "offload_prompt_upsampler",
                    "Offload Prompt Upsampler",
                    kind="boolean",
                    target="call_kwargs",
                    default=True,
                ),
                _field(
                    "offload_guardrail_models", "Offload Guardrail", kind="boolean", target="call_kwargs", default=True
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LYRA1_STATIC_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default image path is the upstream assets/demo/static/diffusion_input/images/00172.png fixture.",),
)
