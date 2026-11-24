# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``astra`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field, _first_existing_path

ASTRA_FALLBACK_FIXTURE = _TEST_CASES_ROOT / "astra" / "condition_images" / "garden_1.png"


ASTRA_OFFICIAL_FIXTURE = _first_existing_path(
    ASTRA_FALLBACK_FIXTURE,
)


ASTRA_OFFICIAL_PROMPT = (
    "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, "
    "charming, and inviting atmosphere. The scene shows a picturesque open square paved with red "
    "bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and "
    "dark-painted facades. On the right side, a lush arrangement of potted plants and blooming "
    "flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands "
    "prominently near the center-right, contributing to the timeless character of the street. "
    "Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. "
    "Bicycles are visible along the edges of the buildings, reinforcing the urban yet leisurely "
    "feel. The sky is bright blue with scattered clouds, and soft sun flares enter the frame from "
    "the left, enhancing the scene's inviting, peaceful mood."
)


ASTRA_OFFICIAL_CALL_KWARGS = {
    "frames_per_generation": 8,
    "total_frames_to_generate": 24,
    "num_inference_steps": 50,
    "start_frame": 0,
    "initial_condition_frames": 1,
    "modality_type": "sekai",
}


ASTRA_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="astra",
    display_name="Astra",
    default_variant_id="default",
    default_task_id="official-sekai-image",
    aliases=("kwai-vgi/astra", "KwaiVGI/Astra"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=ASTRA_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "sekai"),
            notes=(
                "Matches the executable arguments from Astra infer_demo.sh except upstream cam_type, which is represented by direction tokens in this wrapper.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-sekai-image",
            label="Official Sekai Image",
            description="Run the Astra README image-conditioned sekai demo with the packaged garden fixture.",
            aliases=("default", "official-demo", "interactive-video", "image-to-video"),
            inputs=(
                _field(
                    "image",
                    "Condition Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=ASTRA_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=ASTRA_OFFICIAL_PROMPT),
                _field(
                    "interactions",
                    "Camera Direction",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward",),
                ),
                _field(
                    "frames_per_generation", "Frames Per Generation", kind="integer", target="call_kwargs", default=8
                ),
                _field(
                    "total_frames_to_generate", "Generated Frames", kind="integer", target="call_kwargs", default=24
                ),
                _field("num_inference_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field("start_frame", "Start Frame", kind="integer", target="call_kwargs", default=0),
                _field("initial_condition_frames", "Condition Frames", kind="integer", target="call_kwargs", default=1),
                _field(
                    "modality_type",
                    "Modality",
                    target="call_kwargs",
                    default="sekai",
                    choices=("sekai", "nuscenes", "openx"),
                ),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=ASTRA_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The default Astra demo uses the packaged in-tree garden image fixture.",),
)
