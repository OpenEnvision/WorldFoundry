# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``warp_as_history`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _artifact, _field

WARP_AS_HISTORY_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="warp-as-history",
    display_name="Warp-as-History",
    default_variant_id="default",
    default_task_id="official-demo-video",
    aliases=("wah", "warp_as_history", "yyfz233/warp-as-history"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs={"num_frames": 33, "fps": 16, "height": 384, "width": 640},
            aliases=("official", "demo"),
            notes=("Uses the vendored official demo CSV when no custom first frame is provided.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-demo-video",
            label="Official Demo Video",
            description="Run the vendored official Warp-as-History demo CSV through the in-tree runtime.",
            aliases=("default", "interactive-video", "demo"),
            inputs=(
                _field("prompt", "Prompt", target="prompt"),
                _field(
                    "frames",
                    "Frames",
                    kind="integer",
                    target="params",
                    default=33,
                    choices=("33",),
                ),
                _field("fps", "FPS", kind="integer", target="params", default=16, choices=("16",)),
                _field("height", "Height", kind="integer", target="params", default=384),
                _field("width", "Width", kind="integer", target="params", default=640),
                _field("seed", "Seed", kind="integer", target="params", default=42),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={"num_frames": 33, "fps": 16, "height": 384, "width": 640, "seed": 42},
        ),
    ),
    notes=("Custom Warp-as-History runs can still pass images plus runtime kwargs through advanced CLI/API paths.",),
)
