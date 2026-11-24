# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Danze Chen. Adapted from Danzer1xxxxChan/H3-World, revision
# f0c7be2acbbde8256b2473f6e7b088f58272acae (Apache-2.0).
"""Rule table translating H3-World keyboard states into per-latent English instructions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _np() -> Any:
    import numpy as np

    return np


H3_WORLD_KEYS9: tuple[str, ...] = ("W", "A", "S", "D", "I", "J", "K", "L", "F")
_KEY_TO_INDEX: dict[str, int] = {key: index for index, key in enumerate(H3_WORLD_KEYS9)}

H3_WORLD_ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "still": (),
    "forward": ("W",),
    "back": ("S",),
    "strafe-left": ("A",),
    "strafe-right": ("D",),
    "tilt-down": ("I",),
    "tilt-up": ("K",),
    "pan-left": ("J",),
    "pan-right": ("L",),
    "pan-left-fast": ("J", "F"),
    "pan-right-fast": ("L", "F"),
}

_MOTION: dict[str, str] = {
    "W": "walks forward",
    "S": "walks backward",
    "A": "strafes left",
    "D": "strafes right",
}
_MOTION_ORDER: tuple[str, ...] = ("W", "S", "A", "D")
_MOTION_IDLE = "stands still"
_PAN_KEY: dict[str, str] = {"J": "left", "L": "right"}
_TILT_KEY: dict[str, str] = {"I": "tilts down", "K": "tilts up"}
_FAST_KEY = "F"
_CAMERA_IDLE = "holds steady"
_CAMERA_FOLLOW = "follows him"


def _purify(on: dict[str, bool], pairs: tuple[tuple[str, str], ...]) -> None:
    for left, right in pairs:
        if on[left] and on[right]:
            on[left] = on[right] = False


def _motion_clause(on: dict[str, bool]) -> str:
    _purify(on, (("W", "S"), ("A", "D")))
    words = [_MOTION[name] for name in _MOTION_ORDER if on[name]]
    return " and ".join(words) if words else _MOTION_IDLE


def _camera_clause(on: dict[str, bool], *, moving: bool) -> str:
    _purify(on, (("J", "L"), ("I", "K")))
    parts: list[str] = []
    for key, side in _PAN_KEY.items():
        if on[key]:
            parts.append(f"pans {side} {'sharply' if on[_FAST_KEY] else 'slowly'}")
    for key, word in _TILT_KEY.items():
        if on[key]:
            parts.append(word)
    if parts:
        return " and ".join(parts)
    return _CAMERA_FOLLOW if moving else _CAMERA_IDLE


def annotate_from_keys9(keys9: np.ndarray) -> list[str]:
    """Convert a ``[latent_t, 9]`` binary key array into per-latent H3-World clauses."""
    np = _np()
    array = np.asarray(keys9)
    if array.ndim != 2 or array.shape[1] != len(H3_WORLD_KEYS9):
        raise ValueError(f"expected keys9 shape [latent_t, {len(H3_WORLD_KEYS9)}], got {list(array.shape)}")
    out: list[str] = []
    for row in array:
        on = {name: bool(row[i] > 0) for i, name in enumerate(H3_WORLD_KEYS9)}
        motion = _motion_clause(dict(on))
        camera = _camera_clause(dict(on), moving=motion != _MOTION_IDLE)
        out.append(f"the man {motion}, camera {camera}")
    return out


def null_action_script(latent_t: int) -> list[str]:
    """Zero-action CFG reference script with identical sentence shape."""

    if isinstance(latent_t, bool) or int(latent_t) <= 0:
        raise ValueError(f"latent_t must be a positive integer, got {latent_t!r}")
    return [f"the man {_MOTION_IDLE}, camera {_CAMERA_IDLE}"] * int(latent_t)


def _keys_to_row(keys: Sequence[str]) -> np.ndarray:
    np = _np()
    row = np.zeros(len(H3_WORLD_KEYS9), dtype=np.int64)
    for raw in keys:
        token = str(raw).strip()
        if not token:
            continue
        upper = token.upper()
        if upper in _KEY_TO_INDEX:
            row[_KEY_TO_INDEX[upper]] = 1
        elif token.lower() in H3_WORLD_ACTION_PRESETS:
            for preset_key in H3_WORLD_ACTION_PRESETS[token.lower()]:
                row[_KEY_TO_INDEX[preset_key]] = 1
        else:
            raise ValueError(
                f"Unsupported H3-World key or preset {raw!r}; "
                f"keys={H3_WORLD_KEYS9}, presets={sorted(H3_WORLD_ACTION_PRESETS)}"
            )
    return row


def _preset_to_row(preset: str) -> np.ndarray:
    normalized = preset.strip().lower()
    if normalized not in H3_WORLD_ACTION_PRESETS:
        raise ValueError(f"Unsupported H3-World action_preset {preset!r}; supported: {sorted(H3_WORLD_ACTION_PRESETS)}")
    return _keys_to_row(H3_WORLD_ACTION_PRESETS[normalized])


def _interactions_to_script(interactions: Any, *, latent_t: int) -> list[str]:
    """Normalize presets, key combos, scheduled segments, or [T, 9] arrays into sentences."""
    np = _np()
    if isinstance(interactions, str):
        token = interactions.strip()
        if token.lower() in H3_WORLD_ACTION_PRESETS:
            row = _preset_to_row(token)
        elif "+" in token or "," in token:
            parts = [part.strip() for part in token.replace("+", ",").split(",") if part.strip()]
            row = _keys_to_row(parts)
        elif token.upper() in _KEY_TO_INDEX:
            row = _keys_to_row([token])
        else:
            raise ValueError(
                f"Unsupported H3-World interaction {interactions!r}; "
                f"supported presets: {sorted(H3_WORLD_ACTION_PRESETS)}"
            )
        return annotate_from_keys9(np.repeat(row[None, :], latent_t, axis=0))

    if isinstance(interactions, np.ndarray):
        if interactions.ndim == 1 and interactions.shape[0] == len(H3_WORLD_KEYS9):
            return annotate_from_keys9(np.repeat(interactions[None, :], latent_t, axis=0))
        if interactions.ndim == 2 and interactions.shape == (latent_t, len(H3_WORLD_KEYS9)):
            return annotate_from_keys9(interactions)
        raise ValueError(
            f"Expected keys9 array of shape ({len(H3_WORLD_KEYS9)},) or ({latent_t}, {len(H3_WORLD_KEYS9)}), "
            f"got {tuple(interactions.shape)}"
        )

    if not isinstance(interactions, Sequence):
        raise ValueError(f"Unsupported H3-World interactions type: {type(interactions).__name__}")
    items = list(interactions)
    if not items:
        return null_action_script(latent_t)

    # Scheduled segments: [{"preset"|"keys": ..., "latents"|"steps": N}, ...]
    if all(isinstance(item, Mapping) for item in items):
        rows: list[np.ndarray] = []
        for idx, segment in enumerate(items):
            if "preset" in segment:
                row = _preset_to_row(str(segment["preset"]))
            elif "keys" in segment:
                raw_keys = segment["keys"]
                row = _keys_to_row([raw_keys] if isinstance(raw_keys, str) else list(raw_keys))
            else:
                raise ValueError(f"interactions[{idx}] segment requires 'preset' or 'keys'")
            count = segment.get("latents", segment.get("steps"))
            if count is None:
                if len(items) == 1:
                    count = latent_t
                else:
                    raise ValueError(f"interactions[{idx}] segment requires 'latents' count")
            if isinstance(count, bool) or int(count) <= 0:
                raise ValueError(f"interactions[{idx}] latents count must be positive, got {count!r}")
            rows.extend([row] * int(count))
        if len(rows) != latent_t:
            raise ValueError(f"Scheduled interaction segments cover {len(rows)} latents, expected {latent_t}")
        return annotate_from_keys9(np.stack(rows, axis=0))

    # Numeric 2-D matrix or 1-D vector
    if all(isinstance(item, (int, float, np.number)) for item in items):
        return _interactions_to_script(np.asarray(items), latent_t=latent_t)
    if all(isinstance(item, Sequence) and not isinstance(item, str) for item in items):
        if len(items) == latent_t and all(
            len(step) == len(H3_WORLD_KEYS9) and all(isinstance(v, (int, float, np.number)) for v in step)
            for step in items
        ):
            return annotate_from_keys9(np.asarray(items))
        if len(items) == latent_t:
            return annotate_from_keys9(np.stack([_keys_to_row(list(step)) for step in items], axis=0))
        raise ValueError(f"Per-latent interactions sequence must have length {latent_t}, got {len(items)}")

    # Sequence of strings: either per-latent presets (len == latent_t) or held key combo (e.g. ["W", "J", "F"])
    if all(isinstance(item, str) for item in items):
        if len(items) == latent_t and all(
            item.strip().lower() in H3_WORLD_ACTION_PRESETS or item.strip().upper() in _KEY_TO_INDEX for item in items
        ):
            return annotate_from_keys9(np.stack([_keys_to_row([item]) for item in items], axis=0))
        row = _keys_to_row(items)
        return annotate_from_keys9(np.repeat(row[None, :], latent_t, axis=0))

    raise ValueError("Unsupported H3-World interactions payload")


def resolve_action_script(
    *,
    latent_t: int,
    action_preset: str | None = None,
    action_script: Sequence[str] | None = None,
    interactions: Any = None,
    default_preset: str = "forward",
) -> list[str]:
    """Resolve user-supplied preset, interactions, or explicit sentences into ``latent_t`` clauses."""
    np = _np()
    provided = sum(value is not None for value in (action_preset, action_script, interactions))
    if provided > 1:
        raise ValueError("Specify at most one of action_preset, action_script, or interactions")
    if action_script is not None:
        if isinstance(action_script, str) or not isinstance(action_script, Sequence):
            raise ValueError("action_script must be a sequence of per-latent English clauses")
        clauses = [str(line).strip() for line in action_script]
        if len(clauses) != latent_t:
            raise ValueError(f"action_script must contain {latent_t} sentences, got {len(clauses)}")
        if any(not clause for clause in clauses):
            raise ValueError("action_script entries must be non-empty strings")
        return clauses
    if interactions is not None:
        return _interactions_to_script(interactions, latent_t=latent_t)
    preset = default_preset if action_preset is None else str(action_preset)
    row = _preset_to_row(preset)
    return annotate_from_keys9(np.repeat(row[None, :], latent_t, axis=0))


__all__ = [
    "H3_WORLD_ACTION_PRESETS",
    "H3_WORLD_KEYS9",
    "annotate_from_keys9",
    "null_action_script",
    "resolve_action_script",
]
