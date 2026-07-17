"""Canonical OpenCV camera-to-world trajectory and sidecar writer.

``CameraTrajectory`` is the model-agnostic intermediate representation: a stack
of ``(N, 4, 4)`` OpenCV camera-to-world poses plus per-frame intrinsics. Every
adapter consumes this and emits a model-native payload. ``write_target_artifacts``
emits the auditable sidecars that downstream WRBench evaluation
relies on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OPENCV_C2W = "opencv_c2w"
BENCHMARK_TARGET_ROLE = "benchmark_intended_control"
CANONICAL_PROFILE = "canonical_60deg"


def _as_pose_stack(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4) or arr.shape[0] == 0:
        raise ValueError("c2w must have shape (N, 4, 4)")
    return arr


def _as_intrinsics(value: Any, n_frames: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (3, 3):
        arr = np.repeat(arr[None], n_frames, axis=0)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3) or arr.shape[0] != n_frames:
        raise ValueError("intrinsics must have shape (N, 3, 3) or (3, 3)")
    return arr


def _relative_yaw_peak_deg(c2w: np.ndarray) -> float:
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    if len(yaw) == 0:
        return 0.0
    return float(yaw[int(np.argmax(np.abs(yaw)))])


def _control_family(camera_type: str) -> str:
    if camera_type in {"yaw_LR", "yaw_RL"}:
        return "rotation"
    if camera_type in {"pan_LR", "pan_RL"}:
        return "translation"
    if camera_type == "static":
        return "static"
    return "unknown"


def _control_profile(camera_type: str, yaw_peak: float) -> str:
    family = _control_family(camera_type)
    if family == "rotation":
        if abs(abs(float(yaw_peak)) - 60.0) <= 1e-6:
            return "canonical_60deg"
        return f"diagnostic_{abs(float(yaw_peak)):g}deg"
    if family == "translation":
        return "canonical_pan"
    if family == "static":
        return "canonical_static"
    return "unknown"


def _image_size_from_intrinsics(intrinsics: np.ndarray) -> list[int]:
    cx = float(intrinsics[0, 0, 2])
    cy = float(intrinsics[0, 1, 2])
    if cx > 1.0 and cy > 1.0:
        return [int(round(cx * 2.0)), int(round(cy * 2.0))]
    return [832, 480]


def _fov_from_intrinsics(intrinsics: np.ndarray, width: int) -> float:
    fx = float(intrinsics[0, 0, 0])
    if fx <= 0:
        return 60.0
    if fx <= 2.0:
        return float(np.degrees(2.0 * np.arctan(1.0 / (2.0 * fx))))
    return float(np.degrees(2.0 * np.arctan(float(width) / (2.0 * fx))))


@dataclass(frozen=True)
class CameraTrajectory:
    """OpenCV camera-to-world trajectory plus intrinsics."""

    c2w: np.ndarray
    intrinsics: np.ndarray
    camera_type: str
    fps: int = 16
    source: str = "camera_trajectory"
    conversion_mode: str = "direct_pose"
    coordinate_convention: str = OPENCV_C2W

    def __post_init__(self) -> None:
        c2w = _as_pose_stack(self.c2w)
        intrinsics = _as_intrinsics(self.intrinsics, len(c2w))
        object.__setattr__(self, "c2w", c2w)
        object.__setattr__(self, "intrinsics", intrinsics)
        if self.coordinate_convention != OPENCV_C2W:
            raise ValueError("CameraTrajectory stores only opencv_c2w poses")

    @property
    def frame_count(self) -> int:
        return int(self.c2w.shape[0])

    def to_c2w(self) -> np.ndarray:
        return self.c2w.copy()

    def to_w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w).astype(np.float32)

    def resample(self, frame_count: int) -> "CameraTrajectory":
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if frame_count == self.frame_count:
            return self
        src = np.linspace(0, self.frame_count - 1, self.frame_count)
        dst = np.linspace(0, self.frame_count - 1, frame_count)
        c2w = np.empty((frame_count, 4, 4), dtype=np.float32)
        intr = np.empty((frame_count, 3, 3), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                c2w[:, r, c] = np.interp(dst, src, self.c2w[:, r, c])
        for r in range(3):
            for c in range(3):
                intr[:, r, c] = np.interp(dst, src, self.intrinsics[:, r, c])
        c2w[:, 3, :] = np.array([0, 0, 0, 1], dtype=np.float32)
        return CameraTrajectory(
            c2w=c2w,
            intrinsics=intr,
            camera_type=self.camera_type,
            fps=self.fps,
            source=self.source,
            conversion_mode=self.conversion_mode,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "coordinate_convention": self.coordinate_convention,
            "camera_type": self.camera_type,
            "fps": self.fps,
            "source": self.source,
            "conversion_mode": self.conversion_mode,
            "frame_count": self.frame_count,
            "c2w": self.c2w.tolist(),
            "intrinsics": self.intrinsics.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "CameraTrajectory":
        return cls(
            c2w=payload["c2w"],
            intrinsics=payload["intrinsics"],
            camera_type=str(payload.get("camera_type") or ""),
            fps=int(payload.get("fps") or 16),
            source=str(payload.get("source") or "camera_trajectory"),
            conversion_mode=str(payload.get("conversion_mode") or "direct_pose"),
            coordinate_convention=str(payload.get("coordinate_convention") or OPENCV_C2W),
        )

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out

    def write_target_artifacts(
        self,
        video_path: str | Path,
        *,
        conversion_mode: str | None = None,
        target_certification_status: str,
        target_certification_basis: str,
        adapter_provenance: str = "deterministic_adapter",
        evidence_level: str = "benchmark_intent",
        model_payload_type: str | None = None,
        source_action: str | None = None,
        yaw_peak_deg: float | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        fov: float | None = None,
        trajectory_sampling_rule: str = "resolved model-frame go-return trajectory",
        extra_sidecar: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        video = Path(video_path)
        target_path = video.with_suffix(video.suffix + ".target_c2w.npy")
        trajectory_path = video.with_suffix(video.suffix + ".camera_trajectory.json")
        sidecar_path = video.with_suffix(video.suffix + ".camera.json")
        samples_path = video.with_suffix(video.suffix + ".model_control_samples.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, self.to_c2w())
        self.write_json(trajectory_path)
        peak = yaw_peak_deg
        if peak is None and extra_sidecar:
            peak = extra_sidecar.get("yaw_peak_deg", extra_sidecar.get("target_yaw_peak_deg"))
        if peak is None:
            peak = abs(_relative_yaw_peak_deg(self.c2w))
        size = list(image_size) if image_size is not None else _image_size_from_intrinsics(self.intrinsics)
        fov_value = float(fov) if fov is not None else _fov_from_intrinsics(self.intrinsics, int(size[0]))
        direction = source_action or self.camera_type
        family = _control_family(str(direction))
        profile = _control_profile(str(direction), float(peak))
        sidecar: dict[str, Any] = {
            "target_pose_path": str(target_path),
            "trajectory_c2w_path": str(target_path),
            "camera_trajectory_path": str(trajectory_path),
            "target_source": self.source,
            "target_coordinate_convention": OPENCV_C2W,
            "target_certification_status": target_certification_status,
            "target_certification_basis": target_certification_basis,
            "target_role": BENCHMARK_TARGET_ROLE,
            "control_family": family,
            "control_direction": direction,
            "control_profile": profile,
            "evidence_level": evidence_level,
            "canonical_profile": "yaw_go_return_60deg" if profile == CANONICAL_PROFILE else profile,
            "adapter_provenance": adapter_provenance,
            "model_payload_type": model_payload_type or (conversion_mode or self.conversion_mode),
            "certification_reason": target_certification_basis,
            "source_action": direction,
            "yaw_peak_deg": float(peak),
            "num_frames": self.frame_count,
            "fps": float(self.fps),
            "image_size": size,
            "fov": fov_value,
            "trajectory_sampling_rule": trajectory_sampling_rule,
            "conversion_mode": conversion_mode or self.conversion_mode,
            "camera_type": self.camera_type,
            "model_frame_count": self.frame_count,
            "source_frame_count": self.frame_count,
            "resampled": False,
            "model_control_samples_path": str(samples_path),
        }
        if extra_sidecar:
            sidecar.update(extra_sidecar)
        samples_path = Path(sidecar.get("model_control_samples_path") or samples_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        if not samples_path.exists():
            samples_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "payload_type": sidecar.get("model_payload_type"),
                        "official_camera_entrypoint": sidecar.get("official_camera_entrypoint", ""),
                        "model_control_timeline": sidecar.get(
                            "model_control_timeline",
                            {
                                "control_sample_kind": "camera_trajectory_c2w",
                                "control_sample_count": self.frame_count,
                                "target_frame_count": self.frame_count,
                                "source": self.source,
                            },
                        ),
                        "target_frame_count": self.frame_count,
                        "target_coordinate_convention": OPENCV_C2W,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        sidecar["model_control_samples_path"] = str(samples_path)
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "target_pose_path": str(target_path),
            "trajectory_c2w_path": str(target_path),
            "camera_trajectory_path": str(trajectory_path),
            "camera_sidecar_path": str(sidecar_path),
            "model_control_samples_path": str(samples_path),
        }
