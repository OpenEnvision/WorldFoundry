# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``cut3r`` model family."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _artifact, _field

CUT3R_OFFICIAL_FIXTURE = str(Path(__file__).resolve().parents[2] / "data" / "test_cases" / "cut3r" / "examples" / "001")


CUT3R_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="cut3r",
    display_name="CUT3R",
    default_variant_id="default",
    default_task_id="official-export",
    aliases=("cut3r-512", "cut3r_512_dpt_4_64"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs={"size": 512, "vis_threshold": 1.5},
            aliases=("official", "demo"),
            notes=("Uses the staged local CUT3R checkpoint when configured in the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-export",
            label="Official Export",
            description="Run the CUT3R official demo-style export on a sequence directory.",
            aliases=("official", "official-demo", "interactive-video"),
            inputs=(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=CUT3R_OFFICIAL_FIXTURE,
                ),
                _field("size", "Size", kind="integer", target="call_kwargs", default=512),
                _field("vis_threshold", "Visibility Threshold", kind="number", target="call_kwargs", default=1.5),
                _field("revisit", "Revisit", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("official_export", "metadata", required=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={"task_type": "cut3r_official_export", "size": 512, "vis_threshold": 1.5},
        ),
        InferenceTaskProfile(
            task_id="two-stage-3dgs",
            label="Two-stage 3DGS",
            description="Reconstruct a point cloud and render a short 3DGS preview video.",
            aliases=("cut3r_two_stage_3dgs", "two-stage", "3dgs"),
            inputs=(
                _field("input_path", "Input Path", kind="path", target="input_path", required=True),
                _field("interactions", "Interactions", kind="interaction_tokens", target="params"),
                _field("size", "Size", kind="integer", target="call_kwargs", default=512),
                _field("vis_threshold", "Visibility Threshold", kind="number", target="call_kwargs", default=1.5),
                _field("width", "Width", kind="integer", target="params", default=704),
                _field("height", "Height", kind="integer", target="params", default=480),
                _field(
                    "frames_per_interaction", "Frames Per Interaction", kind="integer", target="call_kwargs", default=10
                ),
                _field("num_orbit_frames", "Orbit Frames", kind="integer", target="call_kwargs", default=24),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
                _artifact("scene", "3d"),
            ),
            default_call_kwargs={
                "task_type": "cut3r_two_stage_3dgs",
                "size": 512,
                "vis_threshold": 1.5,
                "image_width": 704,
                "image_height": 480,
            },
        ),
    ),
    notes=("Official export mirrors the upstream demo inputs; 3DGS preview remains available as a separate task.",),
)
