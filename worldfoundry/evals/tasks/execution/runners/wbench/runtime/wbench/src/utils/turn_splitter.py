"""
Turn splitter — computes per-turn frame boundaries based on model-specific formulas.

Different video generation models produce multi-turn videos with varying frame
allocation strategies (overlap, condition frames, VAE tails, etc.). This module
registers precise splitting formulas for each model.

Usage:
    from src.utils.turn_splitter import split_turns

    bounds = split_turns("model_a", total_frames=205, chunk_lengths=[4,4,4,4])
    # [(0, 61), (61, 109), (109, 157), (157, 205)]

    # Unregistered models fall back to equal splitting
    bounds = split_turns("unknown", total_frames=484, chunk_lengths=[4,4,4,4])
    # [(0, 121), (121, 242), (242, 363), (363, 484)]
"""

import math
import os
from dataclasses import dataclass, field


def _split_equal(total_frames: int, n_turns: int) -> list[tuple[int, int]]:
    """Equal splitting (fallback for API models)."""
    bounds = []
    for i in range(n_turns):
        s = round(i * total_frames / n_turns)
        e = round((i + 1) * total_frames / n_turns)
        bounds.append((s, max(s + 1, e)))
    return bounds


# ═══════════════════════════════════════════════════════════════════════════════
# Model-specific splitting formulas
# ═══════════════════════════════════════════════════════════════════════════════


def _split_model_a(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model A: 15fps, first segment int(s*15)+1, continuation int(s*15)+1-13 (overlap 13)"""
    segs = [int(s * 15) + 1 for s in chunk_lengths]
    bounds = [(0, segs[0])]
    for i in range(1, len(segs)):
        start = bounds[-1][1]
        bounds.append((start, start + segs[i] - 13))
    return bounds


def _split_model_b(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model B: 16fps, each segment ceil(s/2)*29, no overlap"""
    bounds, pos = [], 0
    for s in chunk_lengths:
        n_frames = math.ceil(s / 2) * 29
        bounds.append((pos, pos + n_frames))
        pos += n_frames
    return bounds


def _split_model_c(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model C: 16fps, first segment 81 frames, continuation 80 frames"""
    bounds = [(0, 81)]
    for _i in range(1, len(chunk_lengths)):
        start = bounds[-1][1]
        bounds.append((start, start + 80))
    return bounds


def _split_model_d(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model D: 24fps, single-shot, latent_rate=6/s, VAE tail correction"""
    latents = [int(s * 6) for s in chunk_lengths]
    tl = sum(latents)
    if tl % 4 != 0:
        tl += 4 - (tl % 4)
    total_pixel = (tl - 1) * 4 + 1

    bounds, cum = [], 0
    for i, ll in enumerate(latents):
        start = cum * 4
        cum += ll
        end = cum * 4 if i < len(latents) - 1 else total_pixel
        bounds.append((start, end))
    return bounds


def _split_model_e(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model E: 8fps, first segment 29 frames, continuation 32 frames"""
    bounds = [(0, 29)]
    for _i in range(1, len(chunk_lengths)):
        start = bounds[-1][1]
        bounds.append((start, start + 32))
    return bounds


def _split_model_f(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model F: 30fps, first segment has condition frame, each iteration 80 frames"""
    model_chunks = [max(1, round(s * 30 / 80)) for s in chunk_lengths]
    bounds = [(0, 1 + model_chunks[0] * 80)]
    for i in range(1, len(model_chunks)):
        start = bounds[-1][1]
        bounds.append((start, start + model_chunks[i] * 80))
    return bounds


def _split_model_g(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model G: 12fps, single-shot, proportional by duration"""
    bounds, cum = [], 0.0
    for s in chunk_lengths:
        start = int(cum * 12)
        cum += s
        end = int(cum * 12)
        bounds.append((start, end))
    return bounds


def _split_model_h(total_frames: int, chunk_lengths: list[float]) -> list[tuple[int, int]]:
    """Model H: 20fps, FramePack sliding window, first latent*4+1, rest latent*4"""
    latents = [round(s * 5) for s in chunk_lengths]
    tl = 1 + sum(latents)
    total_pixel = (tl - 1) * 4 + 1

    bounds = [(0, latents[0] * 4 + 1)]
    for i in range(1, len(latents)):
        start = bounds[-1][1]
        bounds.append((start, start + latents[i] * 4))
    if bounds[-1][1] != total_pixel:
        bounds[-1] = (bounds[-1][0], total_pixel)
    return bounds


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_SPLITTER_REGISTRY = {
    "model_a": _split_model_a,
    "model_b": _split_model_b,
    "model_c": _split_model_c,
    "model_d": _split_model_d,
    "model_e": _split_model_e,
    "model_f": _split_model_f,
    "model_g": _split_model_g,
    "model_h": _split_model_h,
}

_EQUAL_SPLIT_MODELS = {
    "i2v_model_1",
    "i2v_model_2",
    "i2v_model_3",
    "i2v_model_4",
}


def split_turns(
    model: str,
    total_frames: int,
    chunk_lengths: list[float] | None = None,
    n_turns: int | None = None,
    case_id=None,
) -> list[tuple[int, int]]:
    """Compute per-turn frame boundaries based on model.

    Args:
        model: Model name
        total_frames: Total video frame count
        chunk_lengths: Per-turn duration list (seconds), required for world models
        n_turns: Number of turns (used when chunk_lengths not provided)
        case_id: Case ID (for CSV-annotated models)

    Returns:
        [(start, end), ...] per-turn frame ranges (left-closed, right-open)
    """
    if n_turns is None:
        n_turns = len(chunk_lengths) if chunk_lengths else 1

    model_lower = model.lower().replace("-", "").replace("_", "")

    for name, func in _SPLITTER_REGISTRY.items():
        if name.replace("-", "").replace("_", "") == model_lower:
            if chunk_lengths is None:
                return _split_equal(total_frames, n_turns)
            bounds = func(total_frames, chunk_lengths)
            if bounds and bounds[-1][1] != total_frames:
                bounds[-1] = (bounds[-1][0], total_frames)
            return bounds

    return _split_equal(total_frames, n_turns)


@dataclass
class TurnInfo:
    """Parsed result for one turn."""

    index: int
    clip_path: str | None = None
    start_frame: int = 0
    end_frame: int = 0

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class ResolvedTurns:
    """Return value of resolve_turns."""

    source: str
    video_path: str
    turns: list[TurnInfo] = field(default_factory=list)
    n_turns: int = 0

    @property
    def bounds(self) -> list[tuple[int, int]]:
        return [(t.start_frame, t.end_frame) for t in self.turns]


def resolve_turns(
    model: str,
    case_id: int,
    video_path: str,
    interactions: list[dict],
    clips_dir: str | None = None,
) -> ResolvedTurns:
    """Unified turn resolution: prefer pre-cut clips, fallback to split_turns."""
    import cv2

    n_turns = len(interactions) if interactions else 1

    if clips_dir is None:
        video_dir = os.path.dirname(video_path)
        model_dir = os.path.dirname(video_dir)
        clips_dir = os.path.join(model_dir, "clips")

    clip_case_dir = os.path.join(clips_dir, f"case_{case_id}")
    clip_paths = [os.path.join(clip_case_dir, f"turn_{i}.mp4") for i in range(n_turns)]
    clips_exist = all(os.path.exists(p) for p in clip_paths)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunk_lengths = [inter.get("chunk_length", 4) for inter in interactions] if interactions else None
    bounds = split_turns(model, total_frames, chunk_lengths=chunk_lengths, n_turns=n_turns, case_id=case_id)

    turns = []
    for i, (s, e) in enumerate(bounds):
        turns.append(
            TurnInfo(
                index=i,
                clip_path=clip_paths[i] if clips_exist else None,
                start_frame=s,
                end_frame=e,
            )
        )

    return ResolvedTurns(
        source="clips" if clips_exist else "bounds",
        video_path=video_path,
        turns=turns,
        n_turns=n_turns,
    )
