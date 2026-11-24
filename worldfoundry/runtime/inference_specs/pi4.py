# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``pi4`` (pi^4) 4D geometry and motion model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import GENERIC_IMAGE_FIXTURE, _artifact, _field

PI4_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="pi4",
    display_name="Pi4 (π⁴)",
    default_variant_id="default",
    default_task_id="reconstruction",
    aliases=("pi-4", "yyfz233/Pi4", "yyfz/Pi4"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Pi4 0.5B (Official)",
            checkpoints=(
                InferenceCheckpointRef(
                    role="primary",
                    uri="yyfz233/Pi4",
                    required=True,
                    status="public",
                ),
            ),
            status="configured",
            call_kwargs={"max_size": 512, "stride": 1, "conf_threshold": 0.05},
            aliases=("official", "pi4-0.5b"),
            notes=(
                "Loads the single-file yyfz233/Pi4 model.safetensors checkpoint (CC BY-NC 4.0, non-commercial).",
                "Uses optional flex_gemm CUDA sparse kernels when installed, falling back to a checkpoint-identical pure-PyTorch sparse 3D refiner.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="reconstruction",
            label="3D Geometry Reconstruction",
            description="Reconstruct metric-scale 3D point clouds, depth maps, and camera-to-world poses from RGB images or videos.",
            aliases=("default", "geometry", "point-cloud", "3d-reconstruction"),
            inputs=(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=GENERIC_IMAGE_FIXTURE,
                    description="Image file, image directory, or video file.",
                ),
                _field("stride", "Frame Stride", kind="integer", target="call_kwargs", default=1),
                _field("max_frames", "Max Frames", kind="integer", target="call_kwargs", default=None),
                _field("max_size", "Max Image Size", kind="integer", target="call_kwargs", default=512),
                _field("conf_threshold", "Confidence Threshold", kind="number", target="call_kwargs", default=0.05),
            ),
            outputs=(
                _artifact("point_cloud", "3d", required=True, preview=True),
                _artifact("depth", "depth"),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "task_type": "reconstruction",
                "stride": 1,
                "max_size": 512,
                "conf_threshold": 0.05,
            },
        ),
        InferenceTaskProfile(
            task_id="4d-tracking",
            label="4D Geometry & Dense Tracking",
            description="Jointly reconstruct 3D metric geometry, dense 2D/3D point tracks from a query frame, and sequence-level moving-object masks.",
            aliases=("tracking", "motion", "4d-reconstruction", "interactive-video"),
            inputs=(
                _field(
                    "input_path",
                    "Input Path",
                    kind="path",
                    target="input_path",
                    required=True,
                    description="Multi-frame image directory or video file (requires at least 2 frames).",
                ),
                _field("query_frame", "Query Frame Index", kind="integer", target="call_kwargs", default=0),
                _field("stride", "Frame Stride", kind="integer", target="call_kwargs", default=1),
                _field("max_frames", "Max Frames", kind="integer", target="call_kwargs", default=150),
                _field("max_size", "Max Image Size", kind="integer", target="call_kwargs", default=512),
                _field("conf_threshold", "Confidence Threshold", kind="number", target="call_kwargs", default=0.05),
            ),
            outputs=(
                _artifact("point_cloud", "3d", required=True, preview=True),
                _artifact("tracks", "metadata", required=True),
                _artifact("depth", "depth"),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs={
                "task_type": "tracking",
                "query_frame": 0,
                "stride": 1,
                "max_frames": 150,
                "max_size": 512,
                "conf_threshold": 0.05,
            },
        ),
    ),
    notes=(
        "Pi4 code and weights are licensed under CC BY-NC 4.0 (non-commercial use only).",
        "Tracking requires at least two input frames; omit query_frame for geometry-only inference.",
    ),
)

__all__ = ["PI4_INFERENCE_SPEC"]
