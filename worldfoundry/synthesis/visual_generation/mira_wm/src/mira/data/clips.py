"""Clip enumeration: turning a full match recording into fixed-length, fps-downsampled windows.

The contract: non-overlapping windows of `clip_len` frames at `target_fps` via an integer stride;
a trailing window shorter than `clip_len` is dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

try:
    from itertools import batched  # type: ignore[attr-defined]  # Python 3.12+; fallback below
except ImportError:  # Python 3.10 / 3.11

    def batched(iterable: Iterable, n: int) -> Iterator[tuple]:
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


def compute_stride(source_fps: float, target_fps: int) -> int:
    """Integer decode stride mapping `source_fps` down to `target_fps`.

    Allows a little slack because measured fps is sometimes slightly off (e.g. 29.97 vs 30).
    Raises if the source fps is not (close to) an integer multiple of the target.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    stride = max(1, round(source_fps / target_fps))
    if abs(target_fps * stride - source_fps) > 0.5:
        raise ValueError(f"Source fps {source_fps} not an integer multiple of target fps {target_fps}")
    return stride


def compute_clip_frame_indices(
    total_frames: int, source_fps: float, clip_len: int, target_fps: int
) -> tuple[list[list[int]], int]:
    """Return (list_of_clips, stride) where each clip is `clip_len` source-frame indices.

    Indices step by `stride` so the decoded clip plays at `target_fps`. The final short window
    (if any) is dropped.
    """
    stride = compute_stride(source_fps, target_fps)
    indices = list(range(0, total_frames, stride))
    clips = [list(x) for x in batched(indices, clip_len)]
    if clips and len(clips[-1]) < clip_len:
        clips.pop()
    return clips, stride
