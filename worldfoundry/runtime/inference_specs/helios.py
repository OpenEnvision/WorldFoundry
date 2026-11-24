# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``helios`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field, _local_hfd_checkpoint_path

HELIOS_BASE_CHECKPOINT = _local_hfd_checkpoint_path("BestWishYsh--Helios-Base")


HELIOS_MID_CHECKPOINT = _local_hfd_checkpoint_path("BestWishYsh--Helios-Mid")


HELIOS_DISTILLED_CHECKPOINT = _local_hfd_checkpoint_path("BestWishYsh--Helios-Distilled")


# Product inference must not depend on private or repository test prompts. The
# required field remains empty until the user supplies their own description.
HELIOS_DEMO_PROMPT = (
    "A cat and a dog baking a cake together in a cozy sunlit kitchen. The cat carefully measures flour "
    "while the dog stirs the batter with a wooden spoon, cinematic lighting, detailed natural motion."
)


HELIOS_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="helios",
    display_name="Helios",
    default_variant_id="helios-distilled",
    default_task_id="text-to-video",
    aliases=("bestwishysh/helios",),
    variants=(
        InferenceVariantSpec(
            variant_id="helios-distilled",
            label="Distilled",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=HELIOS_DISTILLED_CHECKPOINT,
                    status="local_required",
                ),
            ),
            call_kwargs={"variant": "distilled", "pyramid_num_inference_steps_list": [2, 2, 2]},
            aliases=("distilled", "helios_distilled", "BestWishYsh/Helios-Distilled"),
            notes=("Official distilled checkpoint and two-step pyramid recipe.",),
        ),
        InferenceVariantSpec(
            variant_id="helios-base",
            label="Base",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=HELIOS_BASE_CHECKPOINT,
                    status="local_required",
                ),
            ),
            call_kwargs={"variant": "base", "pyramid_num_inference_steps_list": [20, 20, 20]},
            aliases=("base", "helios_base", "BestWishYsh/Helios-Base"),
            notes=("Official base checkpoint.",),
        ),
        InferenceVariantSpec(
            variant_id="helios-mid",
            label="Mid",
            status="local_required",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=HELIOS_MID_CHECKPOINT,
                    status="local_required",
                ),
            ),
            call_kwargs={"variant": "mid", "pyramid_num_inference_steps_list": [20, 20, 20]},
            aliases=("mid", "helios_mid", "BestWishYsh/Helios-Mid"),
            notes=("Official mid checkpoint.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="text-to-video",
            label="Text to Video",
            description="Generate a video with an official Helios Base, Mid, or Distilled checkpoint.",
            aliases=("t2v", "video", "default"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=HELIOS_DEMO_PROMPT),
                _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=240),
                _field("height", "Height", kind="integer", target="call_kwargs", default=384),
                _field("width", "Width", kind="integer", target="call_kwargs", default=640),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=24),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field("guidance_scale", "Guidance", target="call_kwargs", default="auto"),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("enable_low_vram_mode", "Low VRAM Mode", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "sample_type": "t2v",
                "num_frames": 240,
                "height": 384,
                "width": 640,
                "fps": 24,
                "num_inference_steps": 50,
                "guidance_scale": "auto",
                "seed": 42,
            },
        ),
    ),
    notes=(
        "Each Workspace variant maps to exactly one official local checkpoint; no synthetic default variant is exposed.",
    ),
)
