# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``tesseract`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

TESSERACT_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="tesseract",
    display_name="TesserAct",
    default_variant_id="rgbdn-sft",
    default_task_id="robot-action-video",
    aliases=("tesser-act", "tesseract-rgbdn"),
    variants=(
        InferenceVariantSpec(
            variant_id="rgbdn-sft",
            label="RGB Depth Normal SFT",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="rgbdn_sft_transformer",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="cogvideox_i2v_base",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "tesseract",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
            },
            aliases=("default", "official", "rgbdn"),
            notes=(
                "Uses the official TesserAct RGB-depth-normal SFT transformer with the local CogVideoX-5b-I2V components.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-action-video",
            label="Robot Action Video",
            description="Generate a robot rollout from RGB, depth, normal, and a task prompt.",
            aliases=("default", "rgbdn", "synthetic-geometry-smoke"),
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
                    default="The robot moves its gripper toward the object on the table.",
                ),
                _field(
                    "checkpoint_dir",
                    "TesserAct RGBDN Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/anyeZHY--tesseract/tesseract_v01e_rgbdn_sft",
                ),
                _field(
                    "base_model_dir",
                    "CogVideoX I2V Base",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/THUDM--CogVideoX-5b-I2V",
                ),
                _field(
                    "geometry_mode",
                    "Geometry Conditioning",
                    target="call_kwargs",
                    default="synthetic-gradient",
                    choices=("synthetic-gradient", "files"),
                    description=(
                        "Synthetic-gradient is a deterministic smoke aid, not estimated geometry; use files for official inputs."
                    ),
                ),
                _field("depth_path", "Official Depth NPY", kind="path", target="call_kwargs"),
                _field("normal_path", "Official Normal Image", kind="path", target="call_kwargs"),
                _field("height", "Height", kind="integer", target="call_kwargs", default=256),
                _field("width", "Per-Modality Width", kind="integer", target="call_kwargs", default=320),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=49),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=4),
                _field("guidance_scale", "Text Guidance", kind="number", target="call_kwargs", default=7.5),
                _field("image_guidance_scale", "Image Guidance", kind="number", target="call_kwargs", default=1.5),
                _field("use_dynamic_cfg", "Dynamic CFG", kind="boolean", target="call_kwargs", default=False),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=8),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=23),
                _field(
                    "mixed_precision",
                    "Precision",
                    target="call_kwargs",
                    default="bf16",
                    choices=("bf16", "fp16", "fp32"),
                ),
                _field("memory_efficient", "CPU Offload", kind="boolean", target="call_kwargs", default=True),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "geometry_mode": "synthetic-gradient",
                "height": 256,
                "width": 320,
                "num_frames": 49,
                "num_inference_steps": 4,
                "guidance_scale": 7.5,
                "image_guidance_scale": 1.5,
                "use_dynamic_cfg": False,
                "fps": 8,
                "seed": 23,
                "mixed_precision": "bf16",
                "memory_efficient": True,
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The generated video concatenates RGB, depth, and normal predictions horizontally, matching the official pipeline.",
        "Synthetic-gradient conditioning validates checkpoint execution only and must not be interpreted as official input quality.",
    ),
)
