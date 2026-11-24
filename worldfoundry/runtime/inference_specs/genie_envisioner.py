# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``genie_envisioner`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

GENIE_ENVISIONER_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="genie-envisioner",
    display_name="Genie Envisioner",
    default_variant_id="ge-base-fast-v0.1",
    default_task_id="robot-multiview-video",
    aliases=("genie_envisioner", "genieenvisioner"),
    variants=(
        InferenceVariantSpec(
            variant_id="ge-base-fast-v0.1",
            label="GE Base Fast v0.1",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="genie_envisioner_transformer",
                    uri=("${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/GE_base_fast_v0.1.safetensors"),
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="ltx_video_base_components",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "genie-envisioner",
                "checkpoint_path": (
                    "${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/GE_base_fast_v0.1.safetensors"
                ),
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
            },
            aliases=("default", "official", "fast", "ge-base-fast"),
            notes=(
                "Uses the released GE-base-fast multiview transformer with local LTX 0.9 tokenizer, T5, and VAE components.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-multiview-video",
            label="Robot Multiview Video",
            description="Generate a three-view robot video from four-frame camera histories; bundled still images are smoke inputs.",
            aliases=("default", "robot-action-video", "synthetic-three-view-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Smoke Preview Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/aloha/observation_images_cam_high.png"
                    ),
                    description="Bundled ALOHA still image used by the default smoke configuration.",
                ),
                _field(
                    "prompt",
                    "Robot Task",
                    target="prompt",
                    required=True,
                    default="Move the robot gripper toward the objects on the workbench.",
                ),
                _field(
                    "checkpoint_path",
                    "Genie Envisioner Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default=("${WORLDFOUNDRY_CKPT_DIR}/agibot-world--Genie-Envisioner/GE_base_fast_v0.1.safetensors"),
                ),
                _field(
                    "base_model_dir",
                    "LTX-Video Base",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/Lightricks--LTX-Video",
                ),
                _field(
                    "input_views",
                    "Camera History Directories or Smoke Images",
                    kind="json",
                    target="call_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                    ),
                    description=(
                        "For official GE-base history, pass three directories in head, left-wrist, "
                        "right-wrist order; each needs 0.png through 3.png. The default three "
                        "ALOHA still images are repeated across four history slots for smoke use."
                    ),
                ),
                _field(
                    "input_mode",
                    "Input Mode",
                    target="call_kwargs",
                    default="explicit-three-view",
                    choices=("explicit-three-view", "synthetic-three-view"),
                    description=(
                        "Explicit mode accepts three four-frame history directories or three still images; "
                        "the bundled still images and synthetic mode are smoke configurations."
                    ),
                ),
                _field("height", "Height", kind="integer", target="call_kwargs", default=192),
                _field("width", "Per-View Width", kind="integer", target="call_kwargs", default=256),
                _field("n_previous", "History Frames", kind="integer", target="call_kwargs", default=4),
                _field("latent_chunk", "Generated Latent Frames", kind="integer", target="call_kwargs", default=2),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=5),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("decode_timestep", "Decode Timestep", kind="number", target="call_kwargs", default=0.03),
                _field("decode_noise_scale", "Decode Noise", kind="number", target="call_kwargs", default=0.025),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=8),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
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
                "input_mode": "explicit-three-view",
                "input_views": [
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                    "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                ],
                "height": 192,
                "width": 256,
                "n_previous": 4,
                "latent_chunk": 2,
                "num_inference_steps": 5,
                "guidance_scale": 1.0,
                "decode_timestep": 0.03,
                "decode_noise_scale": 0.025,
                "fps": 8,
                "seed": 42,
                "mixed_precision": "bf16",
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The default consumes synchronized ALOHA high/left-wrist/right-wrist observations and repeats four history frames.",
        "Use a checkpoint-matched AgiBot or LeRobot trajectory for action-quality parity.",
    ),
)
