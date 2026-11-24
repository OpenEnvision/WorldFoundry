"""
Segment continuity — TransNetV2 scene cut detection.

Detects undesired hard cuts in generated videos. A good world model should
produce temporally coherent video without abrupt transitions.

Method:
1. TransNetV2 predicts per-frame scene boundary probability
2. Frames exceeding threshold (default 0.5) are candidate cuts
3. per-case: score = 0.0 if has_cut else 1.0 (higher = better)
4. model-level: overall.score = continuity_rate = fraction without cuts
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from worldfoundry.base_models.perception_core.video_quality.transnetv2 import (
    TransNetV2,
)
from worldfoundry.base_models.perception_core.video_quality.transnetv2 import (
    checkpoint_path as transnetv2_checkpoint_path,
)

METRIC_NAME = "segment_continuity"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SCENE_LEN = 10

_transnet_model = None


def _get_transnet_model():
    global _transnet_model
    if _transnet_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransNetV2(device=str(device), weights_path=transnetv2_checkpoint_path())
        _transnet_model = _TransNetV2Wrapper(model.eval().to(device), device)
    return _transnet_model


class _TransNetV2Wrapper:
    """Minimal wrapper matching TransNetV2Torch.predict_frames interface."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict_frames(self, frames: np.ndarray):
        # TransNetV2 expects [B, T, 27, 48, 3] uint8
        inp = torch.from_numpy(frames).unsqueeze(0).to(self.device)
        logits = self.model(inp)
        if isinstance(logits, tuple):
            logits = logits[0]
        single_pred = torch.sigmoid(logits[0, :, 0]).cpu().numpy()
        return single_pred, None


def _read_frames_cv2(video_path: str, width: int = 48, height: int = 27) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)


def detect_scene_cuts(
    video_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
) -> list[int]:
    """Detect scene cuts using TransNetV2. Returns list of cut frame indices."""
    model = _get_transnet_model()
    frames = _read_frames_cv2(video_path)
    if len(frames) == 0:
        return []
    s_pred, _ = model.predict_frames(frames)
    raw_cuts = np.where(s_pred > threshold)[0].tolist()
    if not raw_cuts:
        return []
    filtered = [raw_cuts[0]]
    for cut in raw_cuts[1:]:
        if cut - filtered[-1] >= min_scene_len:
            filtered.append(cut)
    return filtered


def compute_case(
    video_path: str, case_id: str, threshold: float = DEFAULT_THRESHOLD, min_scene_len: int = DEFAULT_MIN_SCENE_LEN
) -> dict[str, Any]:
    """Evaluate a single video for segment continuity."""
    try:
        cut_frames = detect_scene_cuts(video_path, threshold, min_scene_len)
        has_cut = len(cut_frames) > 0
        score = 0.0 if has_cut else 1.0
        return {
            "case_id": str(case_id),
            "video_path": str(video_path),
            "score": score,
            "details": {"has_cut": has_cut, "n_cuts": len(cut_frames), "cut_frames": cut_frames},
            "params": {"threshold": threshold, "min_scene_len": min_scene_len, "method": "transnetv2"},
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": str(case_id),
            "video_path": str(video_path),
            "score": None,
            "details": {"has_cut": False, "n_cuts": 0, "cut_frames": []},
            "params": {"threshold": threshold, "min_scene_len": min_scene_len, "method": "transnetv2"},
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize_model_results(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case results into model-level summary."""
    valid = [r for r in case_records if r.get("score") is not None]
    if not valid:
        return {"score": 0.0, "continuity_rate": 0.0, "n_cases": 0}
    cases_with_cuts = sum(1 for r in valid if r["details"]["has_cut"])
    continuity_rate = 1.0 - cases_with_cuts / len(valid)
    return {
        "score": round(continuity_rate, 4),
        "continuity_rate": round(continuity_rate, 4),
        "cases_with_cuts": cases_with_cuts,
        "n_cases": len(valid),
    }
