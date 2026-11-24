# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``longcat_video`` model family."""

from __future__ import annotations

import sys

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)
from worldfoundry.core.io.paths import hfd_root_path

from ._common import _WORKSPACE_ROOT, _artifact, _field, _first_existing_path

LONGCAT_VIDEO_OFFICIAL_PROMPT = (
    "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, "
    "wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla "
    "and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice "
    "cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background "
    "features a green lawn and several tall trees, creating a warm and loving scene."
)


LONGCAT_VIDEO_OFFICIAL_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
    "picture, messy background, three legs, many people in the background, walking backwards"
)


LONGCAT_VIDEO_LOCAL_CHECKPOINT = _first_existing_path(
    hfd_root_path("meituan-longcat--LongCat-Video"),
    _WORKSPACE_ROOT / "ckpt" / "LongCat-Video",
    "meituan-longcat/LongCat-Video",
)


LONGCAT_VIDEO_OFFICIAL_LOAD_KWARGS = {
    "python_executable": sys.executable,
}


LONGCAT_VIDEO_OFFICIAL_CALL_KWARGS = {
    "task_type": "t2v",
    "height": 480,
    "width": 832,
    "num_frames": 93,
    "num_inference_steps": 50,
    "distill_num_inference_steps": 16,
    "guidance_scale": 2.0,
    "seed": 0,
    "fps": 15,
    "context_parallel_size": 1,
    "execute": True,
    "timeout_seconds": 7200,
    "return_dict": True,
}


LONGCAT_VIDEO_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="longcat-video",
    display_name="LongCat-Video",
    default_variant_id="default",
    default_task_id="official-t2v",
    aliases=("longcat", "meituan-longcat/LongCat-Video"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            checkpoints=(
                InferenceCheckpointRef(role="primary", uri=LONGCAT_VIDEO_LOCAL_CHECKPOINT, status="configured"),
            ),
            load_kwargs=LONGCAT_VIDEO_OFFICIAL_LOAD_KWARGS,
            call_kwargs=LONGCAT_VIDEO_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "t2v"),
            notes=("Matches the DiffSynth LongCat-Video text-to-video official example defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-t2v",
            label="Official T2V",
            description="Run the LongCat-Video official text-to-video demo prompt.",
            aliases=("default", "interactive-video", "official-demo", "text-to-video"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=LONGCAT_VIDEO_OFFICIAL_PROMPT),
                _field(
                    "negative_prompt",
                    "Negative Prompt",
                    target="call_kwargs",
                    default=LONGCAT_VIDEO_OFFICIAL_NEGATIVE_PROMPT,
                ),
                _field("task_type", "Task Type", target="call_kwargs", default="t2v", choices=("t2v",)),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=93),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=15),
                _field("height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("width", "Width", kind="integer", target="call_kwargs", default=832),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field(
                    "distill_num_inference_steps", "Distill Steps", kind="integer", target="call_kwargs", default=16
                ),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=2.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
                _field("context_parallel_size", "Context Parallel", kind="integer", target="call_kwargs", default=1),
                _field("execute", "Execute", kind="boolean", target="call_kwargs", default=True),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=7200),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=LONGCAT_VIDEO_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The T2V official demo does not require an input image path.",),
)
