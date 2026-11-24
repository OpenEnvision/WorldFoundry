# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``stable_video_infinity`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field

STABLE_VIDEO_INFINITY_DEMO_ROOT = _TEST_CASES_ROOT / "stable-video-infinity" / "svi-2.0"


STABLE_VIDEO_INFINITY_OFFICIAL_PROMPT = (
    "The water shimmers with dancing caustics as gentle bubbles rise; the camera dolly-ins through a blue haze."
)


STABLE_VIDEO_INFINITY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="stable-video-infinity",
    display_name="Stable Video Infinity 2.0",
    default_variant_id="svi-2.0-wan2.1-i2v-14b",
    default_task_id="prompt-stream-i2v",
    aliases=("svi", "svi-2.0", "stable_video_infinity", "stable-video-infinity-2.0"),
    variants=(
        InferenceVariantSpec(
            variant_id="svi-2.0-wan2.1-i2v-14b",
            label="SVI 2.0 · Wan2.1 I2V 14B",
            status="configured",
            load_kwargs={"lazy": True},
            aliases=("default", "svi-2.0", "wan2.1-i2v-14b"),
            notes=("Loads the locally staged Wan2.1-I2V-14B-480P base and official SVI 2.0 LoRA.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="prompt-stream-i2v",
            label="Prompt Stream Image to Long Video",
            description=("Generate linked SVI 2.0 clips from one reference image and an optional JSON prompt stream."),
            aliases=("default", "i2v", "image-to-video", "long-video", "prompt-stream"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=STABLE_VIDEO_INFINITY_OFFICIAL_PROMPT,
                ),
                _field(
                    "prompt_stream",
                    "Prompt Stream (JSON)",
                    kind="json",
                    target="call_kwargs",
                    description=(
                        'Optional JSON array such as ["enter the forest", "approach the castle"]; '
                        "when provided, it overrides Prompt."
                    ),
                ),
                _field(
                    "image",
                    "Reference Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=str(STABLE_VIDEO_INFINITY_DEMO_ROOT / "frame.jpg"),
                ),
                _field(
                    "num_clips",
                    "Maximum Clips",
                    kind="integer",
                    target="call_kwargs",
                    description=(
                        "Official upper bound; normal prompt streams stop after each prompt has been repeated twice."
                    ),
                ),
                _field("num_frames", "Frames per Clip", kind="integer", target="call_kwargs"),
                _field(
                    "num_motion_frames",
                    "Continuation Frames",
                    kind="integer",
                    target="call_kwargs",
                    description="Tail frames reused to condition the next autoregressive clip.",
                ),
                _field(
                    "num_inference_steps",
                    "Steps per Clip",
                    kind="integer",
                    target="call_kwargs",
                ),
                _field(
                    "cfg_scale_text",
                    "Text CFG",
                    kind="number",
                    target="call_kwargs",
                ),
                _field("seed", "Base Seed", kind="integer", target="call_kwargs", default=0),
                _field("seed_stride", "Seed Stride", kind="integer", target="call_kwargs", default=42),
                _field(
                    "prompt_repeat_times",
                    "Prompt Repeats",
                    kind="integer",
                    target="call_kwargs",
                    default=2,
                ),
                _field(
                    "ref_pad_num",
                    "Reference Padding",
                    kind="integer",
                    target="call_kwargs",
                    default=-1,
                    description="-1 preserves the official full-reference-padding behavior.",
                ),
                _field("height", "Height", kind="integer", target="call_kwargs"),
                _field("width", "Width", kind="integer", target="call_kwargs"),
                _field("fps", "FPS", kind="integer", target="call_kwargs"),
                _field("output_path", "Output Path", kind="path", target="call_kwargs"),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "seed": 0,
                "seed_stride": 42,
                "prompt_repeat_times": 2,
                "ref_pad_num": -1,
            },
        ),
    ),
    notes=(
        "Workspace dispatches this task to the shared cu128 environment and resolves both weight roots from runtime defaults.",
        "Checkpoint-backed parity still requires the Wan2.1 base and SVI LoRA to be staged locally.",
    ),
)
