# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``adaworld`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

ADAWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="adaworld",
    display_name="AdaWorld",
    default_variant_id="official",
    default_task_id="latent-action-world-model",
    variants=(
        InferenceVariantSpec(
            variant_id="official",
            label="Official AdaWorld FDM",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="adaworld_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/Little-Podi--AdaWorld/adaworld.safetensors",
                    status="required",
                ),
            ),
            aliases=("default",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="latent-action-world-model",
            label="Latent Action World Model",
            description="Generate an AdaWorld video from a readable source video using the official FDM checkpoint.",
            aliases=("default",),
            inputs=(
                _field(
                    "video",
                    "Source Video",
                    kind="path",
                    target="input_path",
                    required=True,
                    description="Readable RGB video used to extract AdaWorld's latent action conditioning.",
                ),
                _field(
                    "checkpoint_path",
                    "AdaWorld Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    description="Optional explicit adaworld.safetensors path; the default checkpoint root is discovered automatically.",
                ),
                _field("start_index", "Start Frame", kind="integer", default=0),
                _field("video_len", "Video Frames", kind="integer", default=20),
                _field("resolution", "Resolution", kind="integer", default=256),
                _field("context_frame", "Context Frames", kind="integer", default=6),
                _field("num_steps", "Sampling Steps", kind="integer", default=5),
                _field("cfg_scale", "Guidance Scale", kind="number", default=1.1),
                _field("aug_level", "Augmentation Level", kind="number", default=0.1),
                _field("fps", "FPS", kind="integer", default=5),
                _field("seed", "Seed", kind="integer", default=32),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
        ),
    ),
    notes=(
        "The public path uses the released AdaWorld checkpoint and requires a readable source video.",
        "Only short synthetic-video inference has been verified; task quality and long-horizon behavior are unmeasured.",
    ),
)
