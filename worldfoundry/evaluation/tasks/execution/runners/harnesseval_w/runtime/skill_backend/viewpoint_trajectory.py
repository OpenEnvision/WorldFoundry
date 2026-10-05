"""Local MegaSAM runner for Viewpoint Trajectory."""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..io import value_digest
from ..skills.skill_viewpoint_trajectory import BACKEND_OUTPUT_SCHEMA, SAMPLING
from .common import _validate_video


BACKEND_ID = "harnesseval.viewpoint_trajectory.megasam"
BACKEND_VERSION = "harnesseval.megasam_cam_c2w"
CONFIG_DIGEST = value_digest(
    {
        "implementation": "worldfoundry_megasam_unidepth_v2_compatible",
        "output": "cam_c2w",
        "sampling": SAMPLING,
    }
)

__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "CONFIG_DIGEST",
    "LocalBackend",
]


class LocalBackend:
    """Run all three MegaSAM stages on one GPU with their networks loaded once."""

    backend_id = BACKEND_ID
    version = BACKEND_VERSION
    config_digest = CONFIG_DIGEST
    execution_mode = "local"

    def __init__(
        self,
        weights_root: Path,
        device: str = "cuda",
        frame_batch_size: int = 1,
    ) -> None:
        self.weights_root = weights_root.resolve()
        self.device = device
        self.frame_batch_size = frame_batch_size
        self.load_count = 0
        self._pipeline: Any | None = None
        self._ready_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._pipeline is not None:
            return
        with self._ready_lock:
            if self._pipeline is not None:
                return
            from worldfoundry.base_models.three_dimensions.slam.megasam_resident import ResidentMegaSamPipeline

            self._pipeline = ResidentMegaSamPipeline(
                weights_root=self.weights_root,
                device=self.device,
                target_fps=float(SAMPLING["target_fps"]),
                frame_batch_size=self.frame_batch_size,
            )
            self.load_count = 1

    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> dict[str, Any]:
        _validate_video(video_path, video_fingerprint)
        self.ensure_ready()
        with self._inference_lock, tempfile.TemporaryDirectory(
            prefix="harnesseval-resident-megasam-"
        ) as directory:
            output = Path(directory) / "poses.npz"
            assert self._pipeline is not None
            self._pipeline.evaluate(video_path, output)
            with np.load(output, allow_pickle=False) as payload:
                poses = np.asarray(payload["cam_c2w"], dtype=np.float32)
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "cam_c2w": poses.tolist(),
            "pose_count": len(poses),
        }
