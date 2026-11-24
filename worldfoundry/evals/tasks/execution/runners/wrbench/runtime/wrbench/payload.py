"""Payload contract returned by unified camera adapters.

``CameraPayload`` bundles the model-native control payload, the OpenCV target
trajectory used for evaluation, and provenance metadata. ``ModelControlTimeline``
records how the model-native control samples map back to benchmark frames (dense
or sparse), which is what makes "unified" honest across models that consume pose
matrices, pose text, latent embeddings, geometry, or action tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wrbench.trajectory import CameraTrajectory


@dataclass(frozen=True)
class ModelControlTimeline:
    """Compact description of model-native camera control sampling."""

    schema_version: int
    control_sample_kind: str
    payload_type: str
    requested_frame_count: int
    target_frame_count: int
    control_sample_count: int
    source_frame_indices: list[int]
    model_control_indices: list[int]
    sampling_rule: str
    stride_hint: float | None
    coordinate_convention: str
    target_c2w_is_model_effective: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self, *, max_indices: int = 16) -> dict[str, Any]:
        indices = list(self.source_frame_indices)
        control_indices = list(self.model_control_indices)
        truncated = len(indices) > max_indices
        return {
            "schema_version": int(self.schema_version),
            "control_sample_kind": self.control_sample_kind,
            "payload_type": self.payload_type,
            "requested_frame_count": int(self.requested_frame_count),
            "target_frame_count": int(self.target_frame_count),
            "control_sample_count": int(self.control_sample_count),
            "source_frame_indices": indices[:max_indices],
            "source_frame_indices_truncated": bool(truncated),
            "model_control_indices": control_indices[:max_indices],
            "model_control_indices_truncated": len(control_indices) > max_indices,
            "sampling_rule": self.sampling_rule,
            "stride_hint": self.stride_hint,
            "coordinate_convention": self.coordinate_convention,
            "target_c2w_is_model_effective": bool(self.target_c2w_is_model_effective),
            **self.extra,
        }


@dataclass(frozen=True)
class CameraPayload:
    payload_type: str
    payload: dict[str, Any]
    target_trajectory: CameraTrajectory
    official_camera_entrypoint: str
    coordinate_notes: str
    calibration_status: str
    metadata: dict[str, Any] = field(default_factory=dict)
