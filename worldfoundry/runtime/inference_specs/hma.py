# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``hma`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

HMA_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hma",
    display_name="HMA",
    default_variant_id="base-cont-langtable",
    default_task_id="robot-action-video",
    aliases=("hma-magvit-362m", "hma-continuous"),
    variants=(
        InferenceVariantSpec(
            variant_id="base-cont-langtable",
            label="Base Continuous LangTable",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="hma_continuous_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_temporal_vae",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "hma",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
            aliases=("default", "official", "base-cont", "hma-mar-1b"),
            notes=("Runs the vendored STMAR continuous sampler.",),
        ),
        InferenceVariantSpec(
            variant_id="hma-magvit-362m",
            label="Base Discrete LangTable",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="hma_discrete_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-disc",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "hma",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-disc",
                "model_path": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-disc",
            },
            aliases=("base-disc",),
            notes=(
                "Runs the vendored STMaskGIT discrete sampler and requires the MAGVIT2 tokenizer at "
                "${WORLDFOUNDRY_CKPT_DIR}/1x-technologies--worldmodel/magvit2.ckpt or the "
                "hma_magvit_checkpoint override.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="robot-action-video",
            label="Robot Action Video",
            description="Generate a LangTable rollout; the default repeats one image and action for smoke testing.",
            aliases=("default", "official-demo", "langtable"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Observation",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=(
                        "${WORLDFOUNDRY_REPO_ROOT}/worldfoundry/synthesis/visual_generation/hma/hma_runtime/"
                        "assets/langtable_prompt/frame_00.png"
                    ),
                ),
                _field(
                    "checkpoint_dir",
                    "HMA Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                ),
                _field(
                    "base_model_dir",
                    "SVD Temporal VAE",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                ),
                _field("generated_frames", "Generated Frames", kind="integer", target="call_kwargs", default=6),
                _field("prompt_horizon", "Prompt Horizon", kind="integer", target="call_kwargs", default=3),
                _field("maskgit_steps", "MaskGIT Steps", kind="integer", target="call_kwargs", default=2),
                _field(
                    "trajectory_path",
                    "Paired LangTable Trajectory JSON",
                    kind="path",
                    target="call_kwargs",
                    description=(
                        "Optional real-history input: action_format=raw_language_table_delta_yx, "
                        "prompt_frames has prompt_horizon image paths, and actions has "
                        "prompt_horizon-1+generated_frames raw 2D increments. "
                        "The checkpoint normalizes these actions internally."
                    ),
                ),
                _field(
                    "direction",
                    "Action Direction",
                    target="call_kwargs",
                    default="right",
                    choices=("right", "left", "down", "up"),
                ),
                _field(
                    "action_scale",
                    "Smoke Action Scale",
                    kind="number",
                    target="call_kwargs",
                    default=0.05,
                    description="Repeated raw LangTable 2D delta in the selected direction; ignored for trajectory_path.",
                ),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=2),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "generated_frames": 6,
                "prompt_horizon": 3,
                "maskgit_steps": 2,
                "direction": "right",
                "action_scale": 0.05,
                "fps": 2,
                "seed": 42,
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The compatibility wrapper uses the official continuous checkpoint, two-dimensional actions, and the "
        "SVD temporal VAE without importing TensorFlow/OpenX dataset tooling.",
    ),
)
