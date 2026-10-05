"""Frame extraction helpers for PAWEval evidence packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pawbench.paweval.evidence.frames import EvidenceFrame
from pawbench.paweval.evidence.sampling import FrameSamplingSpec


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def resize_frame(frame: Any, max_side: int) -> Any:
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _unique_indices(indices: list[int], frame_count: int) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for index in indices:
        clipped = min(max(int(index), 0), max(frame_count - 1, 0))
        if clipped in seen:
            continue
        seen.add(clipped)
        unique.append(clipped)
    return unique


def frame_indices_for_sampling(*, frame_count: int, native_fps: float, sampling: FrameSamplingSpec) -> tuple[int, ...]:
    if frame_count <= 0:
        raise ValueError("video reports no frames")
    if native_fps <= 0:
        raise ValueError("video reports no positive native fps")

    duration_s = frame_count / native_fps
    step_s = 1.0 / sampling.fps
    indices: list[int] = []
    t = 0.0
    while t < duration_s:
        indices.append(round(t * native_fps))
        t += step_s

    indices.extend([0, frame_count // 2, frame_count - 1])
    unique = sorted(_unique_indices(indices, frame_count))
    if sampling.max_frames is not None and len(unique) > sampling.max_frames:
        positions = [
            round(i * (len(unique) - 1) / max(sampling.max_frames - 1, 1)) for i in range(sampling.max_frames)
        ]
        unique = _unique_indices([unique[pos] for pos in positions], frame_count)
    return tuple(unique)


def phase_for_index(index: int, *, frame_count: int) -> str:
    if index <= 0:
        return "initial"
    if index >= frame_count - 1:
        return "terminal"
    return "action"


def ensure_required_phase_indices(indices: tuple[int, ...], *, frame_count: int) -> tuple[tuple[str, int], ...]:
    pairs = [(phase_for_index(index, frame_count=frame_count), index) for index in indices]
    phases = {phase for phase, _ in pairs}
    required = {
        "initial": 0,
        "action": max(frame_count // 2, 0),
        "terminal": max(frame_count - 1, 0),
    }
    for phase, index in required.items():
        if phase not in phases:
            pairs.append((phase, index))
    return tuple(sorted(pairs, key=lambda item: (item[1], item[0])))


def extract_video_frames(
    *,
    sample_id: str,
    video_path: Path,
    output_dir: Path,
    sampling: FrameSamplingSpec,
    max_side: int = 768,
) -> tuple[EvidenceFrame, ...]:
    import cv2

    sample_dir = output_dir / safe_name(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        indices = frame_indices_for_sampling(frame_count=frame_count, native_fps=native_fps, sampling=sampling)
        phase_indices = ensure_required_phase_indices(indices, frame_count=frame_count)
        frames: list[EvidenceFrame] = []
        for ordinal, (phase, index) in enumerate(phase_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {index} from {video_path}")
            out_path = sample_dir / f"{ordinal:03d}_{phase}.jpg"
            saved = cv2.imwrite(str(out_path), resize_frame(frame, max_side), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not saved:
                raise RuntimeError(f"failed to write sampled frame: {out_path}")
            frames.append(
                EvidenceFrame(
                    frame_id=f"{sample_id}__f{ordinal:03d}_{phase}",
                    phase=phase,
                    uri="file://" + str(out_path.resolve()),
                    timestamp_s=round(index / native_fps, 3) if native_fps > 0 else None,
                )
            )
        return tuple(frames)
    finally:
        capture.release()
