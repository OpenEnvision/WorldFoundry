# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``flashworld`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import GENERIC_IMAGE_FIXTURE, _artifact, _field

FLASHWORLD_CALL_KWARGS = {
    "num_frames": 16,
    "fps": 15,
    "image_height": 480,
    "image_width": 704,
    "return_video": True,
}


FLASHWORLD_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="flashworld",
    display_name="FlashWorld",
    default_variant_id="default",
    default_task_id="official-camera-scene",
    aliases=("flash-world",),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=FLASHWORLD_CALL_KWARGS,
            aliases=("official", "demo"),
            notes=("Uses the in-tree FlashWorld pipeline with action-derived camera path.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-camera-scene",
            label="Camera Scene",
            description="Run FlashWorld image-conditioned 3D scene generation.",
            aliases=("default", "official-demo", "image-to-3d"),
            inputs=(
                _field(
                    "input_path",
                    "Input Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default="A coherent explorable 3D scene."),
                _field(
                    "interactions",
                    "Actions",
                    kind="interaction_tokens",
                    target="params",
                    default=("forward", "camera_l", "right", "camera_zoom_in"),
                ),
                _field("num_frames", "Frames", kind="integer", target="call_kwargs", default=16),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=15),
                _field("image_height", "Height", kind="integer", target="call_kwargs", default=480),
                _field("image_width", "Width", kind="integer", target="call_kwargs", default=704),
                _field("return_video", "Return Video", kind="boolean", target="call_kwargs", default=True),
            ),
            outputs=(
                _artifact("model", "generated_3d_asset", required=True, preview=True),
                _artifact("video", "video", required=False, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=FLASHWORLD_CALL_KWARGS,
        ),
    ),
    notes=("Default image is the WorldFoundry studio fixture; official FlashWorld examples are JSON prompts.",),
)
