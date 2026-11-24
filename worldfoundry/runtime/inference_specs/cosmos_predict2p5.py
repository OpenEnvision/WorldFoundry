# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``cosmos_predict2p5`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field

COSMOS_PREDICT2P5_VALIDATION_PROMPT = (
    "A nighttime city bus terminal gradually shifts from stillness to subtle movement, "
    "with realistic lighting, stable geometry, and smooth camera motion."
)


COSMOS_PREDICT2P5_VALIDATION_CALL_KWARGS = {
    "num_frames": 93,
    "fps": 16,
    "height": 704,
    "width": 1280,
    "num_inference_steps": 35,
    "guidance_scale": 7.0,
    "seed": 42,
    "output_type": "pt",
}


COSMOS_PREDICT2P5_OFFICIAL_BASE_JSONL = str(_TEST_CASES_ROOT / "cosmos-predict2p5" / "base" / "robot_pouring.jsonl")


COSMOS_PREDICT2P5_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cosmos-predict2p5",
    display_name="Cosmos Predict2.5",
    default_variant_id="default",
    default_task_id="text-to-world-validation",
    aliases=("cosmos-predict2.5", "cosmos-predict-2.5", "nvidia/Cosmos-Predict2.5-2B"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="2B Default",
            status="configured",
            call_kwargs=COSMOS_PREDICT2P5_VALIDATION_CALL_KWARGS,
            aliases=("2b", "default", "validation"),
            notes=("Uses the catalog-resolved Cosmos Predict2.5 2B checkpoint and required Reason1/Wan components.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="text-to-world-validation",
            label="Text-to-World Validation",
            description=(
                "Run a prompt-only Cosmos Predict2.5 validation job. The official robot_pouring "
                "demo is a video2world recipe and the local media assets are Git LFS pointers in "
                "this checkout, so the default Studio task avoids silently pretending to run it."
            ),
            aliases=("default", "validation", "text-to-world", "interactive-video"),
            inputs=(
                _field("prompt", "Prompt", target="prompt", required=True, default=COSMOS_PREDICT2P5_VALIDATION_PROMPT),
                _field(
                    "input_path",
                    "Optional Image",
                    kind="path",
                    target="input_path",
                    default="",
                    description="Leave empty for text-to-world; provide an image path for image-to-world conditioning.",
                ),
                _field("negative_prompt", "Negative Prompt", target="call_kwargs"),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=93),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("height", "Height", kind="integer", target="call_kwargs", default=704),
                _field("width", "Width", kind="integer", target="call_kwargs", default=1280),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=35),
                _field("guidance_scale", "Guidance", kind="number", target="call_kwargs", default=7.0),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("output_type", "Output Type", target="call_kwargs", default="pt", choices=("pt", "pil")),
                _field("use_kerras_sigma", "Karras Sigma", kind="boolean", target="call_kwargs", default=True),
                _field("pad_mode", "Pad Mode", target="call_kwargs", default="repeat", choices=("repeat", "zero")),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=COSMOS_PREDICT2P5_VALIDATION_CALL_KWARGS,
        ),
    ),
    notes=(
        "Official docs run examples/inference.py with assets/base/robot_pouring.jsonl.",
        f"In this checkout that file is not materialized by Git LFS: {COSMOS_PREDICT2P5_OFFICIAL_BASE_JSONL}",
    ),
)
