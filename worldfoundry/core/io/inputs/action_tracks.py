"""Reusable loading and normalization for frame-aligned action tracks.

Robot / world-model recipes store per-frame continuous actions as JSON
(``{"action": [[...], ...], "state": ...}``). :func:`load_action_track`
reads that mapping; :func:`normalize_action_track` (below) pads or
truncates rows to ``num_frames`` and a fixed width so a short clip
does not crash the conditioner. This is JSON I/O only — value scaling
lives in :mod:`worldfoundry.core.action_normalization`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.formats.serialization import read_json

# ──────────────────────────────────────────────────────────────────────────
# JSON action tracks — pad/truncate to num_frames; no value scaling here
# ──────────────────────────────────────────────────────────────────────────


def load_action_track(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Load an action-track mapping from memory or JSON."""

    payload = read_json(source) if isinstance(source, (str, Path)) else source
    if not isinstance(payload, Mapping):
        raise TypeError("action track must be a JSON object or mapping")
    return {str(key): value for key, value in payload.items()}


def _normalize_rows(rows: Any, *, num_frames: int, width: int, name: str) -> list[list[float]]:
    """Pad short tracks with zeros; reject row-width mismatches instead of silent crop."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError(f"action track {name!r} must be a sequence")
    normalized: list[list[float]] = []
    for index in range(num_frames):
        row = rows[index] if index < len(rows) else [0.0] * width
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"action track {name!r} row {index} must be a sequence")
        if len(row) != width:
            raise ValueError(f"action track {name!r} row {index} has width {len(row)}; expected {width}")
        normalized.append([float(value) for value in row])
    return normalized


def normalize_action_track(
    source: Mapping[str, Any] | str | Path,
    *,
    num_frames: int,
    action_dim: int = 23,
    camera_dim: int = 2,
) -> dict[str, list[list[float]]]:
    """Validate and pad/truncate a keyboard/camera action track to ``num_frames``.

    Missing trailing rows become zeros. A missing ``camera`` key is
    omitted from the result; ``keyboard`` is required.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    payload = load_action_track(source)
    if "keyboard" not in payload:
        raise KeyError("action track is missing required 'keyboard' rows")
    result = {
        "keyboard": _normalize_rows(payload["keyboard"], num_frames=num_frames, width=action_dim, name="keyboard")
    }
    if payload.get("camera") is not None:
        result["camera"] = _normalize_rows(payload["camera"], num_frames=num_frames, width=camera_dim, name="camera")
    return result


def action_track_tensors(
    source: Mapping[str, Any] | str | Path,
    *,
    num_frames: int,
    action_dim: int = 23,
    camera_dim: int = 2,
):
    """Return batched float32 keyboard and optional camera tensors.

    Shapes are ``[1, T, action_dim]`` and ``[1, T, camera_dim]`` (or
    ``None`` when the track has no camera rows).
    """

    import torch

    payload = normalize_action_track(
        source,
        num_frames=num_frames,
        action_dim=action_dim,
        camera_dim=camera_dim,
    )
    keyboard = torch.tensor(payload["keyboard"], dtype=torch.float32).unsqueeze(0)
    camera_rows = payload.get("camera")
    camera = torch.tensor(camera_rows, dtype=torch.float32).unsqueeze(0) if camera_rows is not None else None
    return keyboard, camera


__all__ = ["action_track_tensors", "load_action_track", "normalize_action_track"]
