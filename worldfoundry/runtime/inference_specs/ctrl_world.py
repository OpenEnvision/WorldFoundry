# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``ctrl_world`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

CTRL_WORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="ctrl-world",
    display_name="Ctrl-World",
    default_variant_id="droid-checkpoint-10000",
    default_task_id="robot-action-video",
    aliases=("ctrl_world", "ctrlworld"),
    variants=(
        InferenceVariantSpec(
            variant_id="droid-checkpoint-10000",
            label="DROID Checkpoint 10000",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="ctrl_world_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="clip_text_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "ctrl-world",
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                "clip_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
            },
            aliases=("default", "official", "droid"),
            notes=("Uses the released DROID action-conditioned checkpoint with local SVD and CLIP components.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-action-video",
            label="Robot Action Video",
            description="Generate a three-camera action-conditioned robot rollout.",
            aliases=("default", "droid", "synthetic-three-view-smoke"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/"
                        "test_vla_case1/aloha/observation_images_cam_high.png"
                    ),
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
                    "Ctrl-World Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/yjguo--Ctrl-World/checkpoint-10000.pt",
                ),
                _field(
                    "base_model_dir",
                    "Stable Video Diffusion Base",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                ),
                _field(
                    "clip_model_dir",
                    "CLIP Text Model",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/openai--clip-vit-base-patch32",
                ),
                _field(
                    "input_views",
                    "Synchronized Camera Views",
                    kind="json",
                    target="call_kwargs",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_high.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_left_wrist.png",
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/data/test_cases/test_vla_case1/aloha/observation_images_cam_right_wrist.png",
                    ),
                    description="Three synchronized high, left-wrist, and right-wrist camera images.",
                ),
                _field(
                    "input_mode",
                    "Three-View Input",
                    target="call_kwargs",
                    default="explicit-three-view",
                    choices=("explicit-three-view", "synthetic-three-view"),
                    description="Consumes three synchronized camera images by default; synthetic mode is retained for smoke tests.",
                ),
                _field(
                    "action_mode",
                    "Action Conditioning",
                    target="call_kwargs",
                    default="absolute-pose",
                    choices=("absolute-pose", "synthetic-zero-smoke"),
                    description="Physical DROID Cartesian pose conditioning is required for action-quality evaluation; zero-pose conditioning is smoke only.",
                ),
                _field(
                    "initial_pose",
                    "Initial Cartesian Pose",
                    kind="json",
                    target="call_kwargs",
                    description="Required in absolute-pose mode: seven physical DROID values [x, y, z, rx, ry, rz, gripper] corresponding to the three camera images.",
                ),
                _field(
                    "action_direction",
                    "Action Direction",
                    target="call_kwargs",
                    default="right",
                    choices=("stationary", "forward", "backward", "left", "right", "up", "down", "open", "close"),
                ),
                _field("action_distance", "XYZ Distance (metres)", kind="number", target="call_kwargs", default=0.08),
                _field(
                    "action_scale",
                    "Smoke Normalized Delta",
                    kind="number",
                    target="call_kwargs",
                    description="Only used with synthetic-zero-smoke conditioning.",
                ),
                _field("height", "Per-View Height", kind="integer", target="call_kwargs", default=192),
                _field("width", "Per-View Width", kind="integer", target="call_kwargs", default=320),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=5),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=25),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=1.0),
                _field("motion_bucket_id", "Motion Bucket", kind="integer", target="call_kwargs", default=127),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=4),
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
                "action_direction": "right",
                "action_scale": 0.2,
                "height": 192,
                "width": 320,
                "num_frames": 5,
                "num_inference_steps": 25,
                "guidance_scale": 1.0,
                "motion_bucket_id": 127,
                "fps": 4,
                "seed": 42,
                "mixed_precision": "bf16",
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The output concatenates three predicted camera views horizontally.",
        "The default uses synchronized ALOHA views; checkpoint-matched DROID actions and views remain necessary for action-quality parity.",
    ),
)
