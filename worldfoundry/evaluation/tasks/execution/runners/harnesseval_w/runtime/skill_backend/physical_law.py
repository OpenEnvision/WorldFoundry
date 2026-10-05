"""Local OpenCV backend for canonical physical-law motion signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import value_digest
from ..skills.skill_physical_law import BACKEND_OUTPUT_SCHEMA


ANALYSIS_WIDTH = 320
ANALYSIS_HEIGHT = 180
SPATIAL_CELLS = 12
FRAME_STRIDE = 2
ACTIVITY_THRESHOLD = 0.30
BACKEND_ID = "opencv_farneback.physical_law_signals"
BACKEND_VERSION = "harnesseval.physical_law_signals"

__all__ = ["LocalBackend"]


def _json_values(values: np.ndarray) -> list[float | None]:
    return [round(float(value), 8) if np.isfinite(value) else None for value in values]


class LocalBackend:
    """Decode every second frame and return raw Farneback motion signals."""

    backend_id = BACKEND_ID
    version = BACKEND_VERSION
    execution_mode = "local"

    def __init__(self) -> None:
        self.config_digest = value_digest(
            {
                "backend_id": self.backend_id,
                "version": self.version,
                "frame_stride": FRAME_STRIDE,
                "resolution": [ANALYSIS_WIDTH, ANALYSIS_HEIGHT],
                "spatial_cells": SPATIAL_CELLS,
                "activity_threshold": ACTIVITY_THRESHOLD,
                "farneback": [0.5, 3, 15, 3, 5, 1.2, 0],
            }
        )

    @staticmethod
    def _read_frames(video_path: Path) -> tuple[list[Any], int, float]:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"video cannot be decoded: {video_path}")
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % FRAME_STRIDE == 0:
                resized = cv2.resize(frame, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
            index += 1
        capture.release()
        return frames, source_count, source_fps

    def infer(self, video_path: Path) -> dict[str, Any]:
        import cv2

        frames, source_count, source_fps = self._read_frames(video_path.resolve())
        sampling = {
            "method": "every_second_original_frame",
            "frame_stride": FRAME_STRIDE,
            "analysis_resolution": [ANALYSIS_WIDTH, ANALYSIS_HEIGHT],
            "source_frame_count": source_count,
            "source_fps": source_fps,
            "sampled_frame_count": len(frames),
        }
        if len(frames) < 3:
            return {
                "schema_version": BACKEND_OUTPUT_SCHEMA,
                "status": "no_signal",
                "signals": None,
                "sampling": sampling,
            }

        pair_count = len(frames) - 1
        motion = np.zeros(pair_count)
        cx = np.full(pair_count, np.nan)
        cy = np.full(pair_count, np.nan)
        left_vy = np.zeros(pair_count)
        right_vy = np.zeros(pair_count)
        onset_col = np.full(SPATIAL_CELLS, np.nan)
        onset_row = np.full(SPATIAL_CELLS, np.nan)
        active_cols = np.zeros(SPATIAL_CELLS)
        column_width = ANALYSIS_WIDTH // SPATIAL_CELLS
        row_height = ANALYSIS_HEIGHT // SPATIAL_CELLS

        for index in range(pair_count):
            flow = cv2.calcOpticalFlowFarneback(
                frames[index],
                frames[index + 1],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            magnitude = np.linalg.norm(flow, axis=2)
            vertical = flow[..., 1]
            motion[index] = float(magnitude.sum())

            strongest = magnitude > np.percentile(magnitude, 98)
            if strongest.sum() > 4:
                y, x = np.nonzero(strongest)
                weights = magnitude[y, x]
                cx[index] = float((x * weights).sum() / weights.sum()) / ANALYSIS_WIDTH
                cy[index] = float((y * weights).sum() / weights.sum()) / ANALYSIS_HEIGHT

            left_magnitude = magnitude[:, : ANALYSIS_WIDTH // 2]
            right_magnitude = magnitude[:, ANALYSIS_WIDTH // 2 :]
            left_weight = float(left_magnitude.sum())
            right_weight = float(right_magnitude.sum())
            if left_weight > 1e-6:
                left_vy[index] = float(
                    (vertical[:, : ANALYSIS_WIDTH // 2] * left_magnitude).sum()
                    / left_weight
                )
            if right_weight > 1e-6:
                right_vy[index] = float(
                    (vertical[:, ANALYSIS_WIDTH // 2 :] * right_magnitude).sum()
                    / right_weight
                )

            for cell_index in range(SPATIAL_CELLS):
                column = magnitude[
                    :, cell_index * column_width : (cell_index + 1) * column_width
                ]
                if column.mean() > ACTIVITY_THRESHOLD:
                    active_cols[cell_index] += 1
                    if np.isnan(onset_col[cell_index]):
                        onset_col[cell_index] = index / pair_count
                row = magnitude[
                    cell_index * row_height : (cell_index + 1) * row_height, :
                ]
                if row.mean() > ACTIVITY_THRESHOLD and np.isnan(onset_row[cell_index]):
                    onset_row[cell_index] = index / pair_count

        active_cols /= pair_count
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "status": "ok",
            "signals": {
                "n": pair_count,
                "motion": _json_values(motion),
                "cx": _json_values(cx),
                "cy": _json_values(cy),
                "onset_col": _json_values(onset_col),
                "onset_row": _json_values(onset_row),
                "left_vy": _json_values(left_vy),
                "right_vy": _json_values(right_vy),
                "active_cols": _json_values(active_cols),
            },
            "sampling": sampling,
        }
