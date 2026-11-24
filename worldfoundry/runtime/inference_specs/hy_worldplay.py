# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``hy_worldplay`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field

HY_WORLDPLAY_OFFICIAL_PROMPT = (
    "A paved pathway leads towards a stone arch bridge spanning a calm body of water.  "
    "Lush green trees and foliage line the path and the far bank of the water. "
    "A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. "
    "The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, "
    "natural light, creating a tranquil and serene atmosphere. The pathway is composed of "
    "large, rectangular stones, and the bridge is constructed of light gray stone.  The overall "
    "composition emphasizes the peaceful and harmonious nature of the landscape."
)


HY_WORLDPLAY_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "hunyuan_worldplay" / "test.png")


# Workspace schedules catalog jobs on an explicitly selected ``cuda:N`` by
# default.  Keep the full upstream generation settings, but use one process so
# the static task default cannot silently turn a single-GPU request into an
# eight-rank torchrun job. Users can still select any supported 2/3/4/6/8-rank
# topology explicitly.
HY_WORLDPLAY_WORKSPACE_NPROC = 1


HY_WORLDPLAY_OFFICIAL_CALL_KWARGS = {
    "num_frames": 125,
    "aspect_ratio": "16:9",
    "num_inference_steps": 4,
    "seed": 1,
    "fps": 24,
    "output_type": "pt",
    "prompt_rewrite": False,
    "enable_sr": False,
    "return_pre_sr_video": False,
    "few_step": True,
    "chunk_latent_frames": 4,
    "model_type": "ar",
    "transformer_resident_ar_rollout": True,
    "user_width": 832,
    "user_height": 480,
}


HY_WORLDPLAY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuan-worldplay",
    display_name="HY-WorldPlay",
    default_variant_id="default",
    default_task_id="official-image-pose-video",
    aliases=("hy-worldplay", "hyworldplay", "tencent/HY-WorldPlay"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=HY_WORLDPLAY_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "480p-i2v"),
            notes=("Matches the upstream HY-WorldPlay run.sh defaults for the 480p image-to-video demo.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-image-pose-video",
            label="Official Image Pose Video",
            description="Run the HY-WorldPlay official image-to-video demo with the default pose string.",
            aliases=("default", "official-demo", "interactive-video", "image-to-video"),
            inputs=(
                _field(
                    "image",
                    "Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=HY_WORLDPLAY_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=HY_WORLDPLAY_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Pose",
                    kind="interaction_tokens",
                    target="params",
                    required=True,
                    default=("w-31",),
                ),
                _field("num_frames", "Frames", kind="integer", target="params", default=125),
                _field("fps", "FPS", kind="integer", target="params", default=24),
                _field("height", "Height", kind="integer", target="params", default=480),
                _field("width", "Width", kind="integer", target="params", default=832),
                _field("num_inference_steps", "Steps", kind="integer", target="params", default=4),
                _field("seed", "Seed", kind="integer", target="params", default=1),
                _field("aspect_ratio", "Aspect Ratio", target="call_kwargs", default="16:9", choices=("16:9", "9:16")),
                _field("few_step", "Few Step", kind="boolean", target="call_kwargs", default=True),
                _field("enable_sr", "Super Resolution", kind="boolean", target="call_kwargs", default=False),
                _field("prompt_rewrite", "Prompt Rewrite", kind="boolean", target="call_kwargs", default=False),
                _field(
                    "torchrun_nproc_per_node",
                    "GPUs",
                    kind="integer",
                    target="load_kwargs",
                    default=HY_WORLDPLAY_WORKSPACE_NPROC,
                    description=(
                        "Torchrun workers for sequence-parallel inference. Workspace defaults to one worker "
                        "for cuda:N jobs; select 2, 3, 4, 6, or 8 explicitly for a coordinated multi-GPU run."
                    ),
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HY_WORLDPLAY_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=(
        "The action checkpoint is resolved from the Studio catalog model_ref; the base video model uses the runtime config.",
    ),
)
