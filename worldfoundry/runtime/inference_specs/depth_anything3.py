# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``depth_anything3`` model family."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _artifact, _field

DEPTH_ANYTHING3_OFFICIAL_FIXTURE = str(
    Path(__file__).resolve().parents[2] / "data" / "test_cases" / "depth_anything_v3" / "examples" / "SOH"
)


DEPTH_ANYTHING3_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="depth-anything-v3",
    display_name="Depth Anything 3",
    default_variant_id="large",
    default_task_id="official-soh-export",
    aliases=("da3", "depth-anything-3", "depthanything3"),
    variants=(
        InferenceVariantSpec(
            variant_id="large",
            label="DA3 Large",
            status="configured",
            call_kwargs={"export_format": "mini_npz", "process_res": 504},
            aliases=("default", "official", "large-1.1"),
            notes=("Uses the staged DA3-LARGE checkpoint when configured in the Studio catalog.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-soh-export",
            label="Official SOH Export",
            description="Run the official Depth Anything 3 SOH-style image sequence export.",
            aliases=("default", "official", "official-demo", "interactive-video", "depth"),
            inputs=(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=DEPTH_ANYTHING3_OFFICIAL_FIXTURE,
                ),
                _field("export_format", "Export Format", target="call_kwargs", default="mini_npz"),
                _field("process_res", "Process Resolution", kind="integer", target="call_kwargs", default=504),
                _field("process_res_method", "Resize Method", target="call_kwargs", default="upper_bound_resize"),
                _field("max_frames", "Max Frames", kind="integer", target="call_kwargs", default=2),
                _field("frame_stride", "Frame Stride", kind="integer", target="call_kwargs", default=1),
                _field("infer_gs", "Infer Gaussians", kind="boolean", target="call_kwargs", default=False),
            ),
            outputs=(
                _artifact("depth", "depth", required=True, preview=True),
                _artifact("point_cloud", "3d"),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "export_format": "mini_npz",
                "process_res": 504,
                "process_res_method": "upper_bound_resize",
                "max_frames": 2,
                "frame_stride": 1,
                "infer_gs": False,
            },
        ),
    ),
    notes=(
        "The default task mirrors the lightweight official SOH demo path; GLB/GS export can be exposed as a separate task.",
    ),
)
