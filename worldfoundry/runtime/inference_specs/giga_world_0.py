# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``giga_world_0`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

GIGA_WORLD_0_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="giga-world-0",
    display_name="GigaWorld-0",
    default_variant_id="gr1-2b",
    default_task_id="robot-image-to-video",
    aliases=("giga_world_0", "gigaworld-0", "gigaworld0"),
    variants=(
        InferenceVariantSpec(
            variant_id="gr1-2b",
            label="GR1 Video 2B",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="giga_world_0_gr1_transformer",
                    uri=("${WORLDFOUNDRY_CKPT_DIR}/open-gigaai--GigaWorld-0-Video-GR1-2b/transformer"),
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="t5_11b_encoder_safetensors",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b-encoder",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="wan_diffusers_vae",
                    uri=("${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B-Diffusers/vae"),
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "giga-world-0",
                "transformer_model_dir": ("${WORLDFOUNDRY_CKPT_DIR}/open-gigaai--GigaWorld-0-Video-GR1-2b/transformer"),
                "text_encoder_model_dir": ("${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b-encoder"),
                "vae_model_dir": ("${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B-Diffusers/vae"),
            },
            aliases=("default", "official", "gr1"),
            notes=(
                "Uses the released GR1 image-conditioned transformer with a safe local T5 encoder and Wan Diffusers VAE.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-image-to-video",
            label="Robot Image To Video",
            description="Generate a short robot-world video from one initial observation and task prompt.",
            aliases=("default", "gr1", "single-image-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/droid/exterior_image_1_left.png"
                    ),
                ),
                _field(
                    "prompt",
                    "Robot Task",
                    target="prompt",
                    required=True,
                    default="A robot arm moves toward the objects on the workbench.",
                ),
                _field(
                    "transformer_model_dir",
                    "GR1 Transformer",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=("${WORLDFOUNDRY_CKPT_DIR}/open-gigaai--GigaWorld-0-Video-GR1-2b/transformer"),
                ),
                _field(
                    "text_encoder_model_dir",
                    "T5-11B Encoder",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/google-t5--t5-11b-encoder",
                ),
                _field(
                    "vae_model_dir",
                    "Wan Diffusers VAE",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=("${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B-Diffusers/vae"),
                ),
                _field(
                    "attention_backend",
                    "Attention Backend",
                    target="call_kwargs",
                    default="natten",
                    choices=("natten", "torch"),
                    description=(
                        "NATTEN is the official Hopper path. Torch SDPA is a low-resolution "
                        "checkpoint-smoke compatibility backend, not quality parity."
                    ),
                ),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=640),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=61),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=30),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=6666),
                _field("max_text_length", "Text Tokens", kind="integer", target="call_kwargs", default=128),
                _field(
                    "mixed_precision",
                    "Precision",
                    target="call_kwargs",
                    default="bf16",
                    choices=("bf16", "fp16", "fp32"),
                ),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "attention_backend": "natten",
                "height": 480,
                "width": 640,
                "num_frames": 61,
                "num_inference_steps": 30,
                "guidance_scale": 1.0,
                "fps": 16,
                "seed": 6666,
                "max_text_length": 128,
                "mixed_precision": "bf16",
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The default uses released 480x640, 61-frame, 30-step NATTEN sampling settings.",
        "Use a checkpoint-matched GR1 observation for dataset parity.",
    ),
)
