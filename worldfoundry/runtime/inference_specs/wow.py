# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``wow`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _TEST_CASES_ROOT, _WORKSPACE_ROOT, _artifact, _field

_WOW_OFFICIAL_LOCAL_CHECKPOINT = _WORKSPACE_ROOT / "ckpt" / "WoW-1-Wan-14B-600k"


WOW_LOCAL_CHECKPOINT = (
    str(_WOW_OFFICIAL_LOCAL_CHECKPOINT) if _WOW_OFFICIAL_LOCAL_CHECKPOINT.exists() else "X-Humanoid/WoW-1-Wan-14B-600k"
)


WOW_OFFICIAL_IMAGE_FIXTURE = str(_TEST_CASES_ROOT / "test_vla_case1" / "droid" / "exterior_image_1_left.png")


WOW_OFFICIAL_PROMPT = "The Franka robot grasps the red bottle on the table."


WOW_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="wow",
    display_name="WoW",
    default_variant_id="wan-14b-600k",
    default_task_id="official-image-to-world",
    aliases=("wow-world-model", "wow-world-model/wow-1-wan-14b-600k"),
    variants=(
        InferenceVariantSpec(
            variant_id="wan-14b-600k",
            label="Wan 14B 600k",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri=WOW_LOCAL_CHECKPOINT,
                    status="configured",
                ),
            ),
            call_kwargs={
                "steps": 50,
                "seed": 42,
                "num_frames": 41,
                "no_tiled": False,
                "enable_vram_management": True,
                "persistent_param_gb": 70,
            },
            aliases=("default", "official", "14b", "wan-14b", "600k"),
            notes=("Matches the official WoW Wan demo checkpoint X-Humanoid/WoW-1-Wan-14B-600k.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-image-to-world",
            label="Image To World",
            description="Run the WoW image-conditioned world-video generation path with explicit prompt and sampling controls.",
            aliases=("default", "official-demo", "interactive-video", "image-to-video"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=WOW_OFFICIAL_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", required=True, default=WOW_OFFICIAL_PROMPT),
                _field("steps", "Steps", kind="integer", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs"),
                _field("seed", "Seed", kind="integer", target="call_kwargs"),
                _field("no_tiled", "Disable Tiling", kind="boolean", target="call_kwargs", default=False),
                _field("enable_vram_management", "VRAM Management", kind="boolean", target="call_kwargs", default=True),
                _field("persistent_param_gb", "Persistent Param GB", kind="integer", target="call_kwargs", default=70),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "no_tiled": False,
                "enable_vram_management": True,
                "persistent_param_gb": 70,
            },
        ),
    ),
    notes=("The upstream WoW script uses image conditioning, prompt text, seed 42, and generated video output.",),
)
