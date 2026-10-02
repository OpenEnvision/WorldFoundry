"""VPT action-sequence bridge for the Open Dreamer world-model runtime.

Open Dreamer is conditioned on Minecraft/VPT action dictionaries: one JSON object
per frame carrying a ``mouse`` and a ``keyboard`` field. WorldFoundry callers
speak the shared interaction vocabulary instead (``forward``, ``camera_l``,
``forward_camera_r``, ...), so this module renders that vocabulary into the VPT
wire format and writes the JSONL file the official rollout entrypoint reads.

The wire format is the public OpenAI Video-Pre-Training contractor format that
Open Dreamer documents as its input contract::

    {"mouse": {"dx": 0.0, "dy": 0.0, "buttons": [], "dwheel": 0.0},
     "keyboard": {"keys": ["key.keyboard.w"]}}

Mouse deltas are raw VPT units. VPT scales them by ``360 / 2400`` degrees per
unit, so this module converts a caller-facing degree step into raw units before
emitting the action.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# VPT scales raw mouse deltas by 360/2400 degrees per unit before binning them
# into the camera action grid. Callers think in degrees, so convert on the way in.
RAW_UNITS_PER_DEGREE = 2400.0 / 360.0

# Per-frame camera step used when an interaction does not carry explicit deltas.
# VPT clips camera motion at 30 degrees per frame; 5 degrees keeps the synthesized
# rollout inside the well-populated centre of the mu-law camera bins.
DEFAULT_CAMERA_STEP_DEGREES = 5.0

DEFAULT_INTERACTIONS: tuple[str, ...] = (
    "forward",
    "forward_camera_l",
    "forward",
    "forward_camera_r",
)

# Interaction token -> VPT keyboard key names.
KEYBOARD_TOKENS: dict[str, tuple[str, ...]] = {
    "forward": ("key.keyboard.w",),
    "back": ("key.keyboard.s",),
    "left": ("key.keyboard.a",),
    "right": ("key.keyboard.d",),
    "jump": ("key.keyboard.space",),
    "sneak": ("key.keyboard.left.shift",),
    "sprint": ("key.keyboard.left.control",),
    "inventory": ("key.keyboard.e",),
    "drop": ("key.keyboard.q",),
    "swap_hands": ("key.keyboard.f",),
    "escape": ("key.keyboard.escape",),
    "hotbar_1": ("key.keyboard.1",),
    "hotbar_2": ("key.keyboard.2",),
    "hotbar_3": ("key.keyboard.3",),
    "hotbar_4": ("key.keyboard.4",),
    "hotbar_5": ("key.keyboard.5",),
    "hotbar_6": ("key.keyboard.6",),
    "hotbar_7": ("key.keyboard.7",),
    "hotbar_8": ("key.keyboard.8",),
    "hotbar_9": ("key.keyboard.9",),
}

# Interaction token -> VPT mouse button index.
MOUSE_BUTTON_TOKENS: dict[str, int] = {
    "attack": 0,
    "use": 1,
    "pick_item": 2,
}

# Interaction token -> (dx, dy) in camera-step units. Positive dx looks right and
# positive dy looks down, matching the VPT contractor recordings.
CAMERA_TOKENS: dict[str, tuple[float, float]] = {
    "camera_l": (-1.0, 0.0),
    "camera_r": (1.0, 0.0),
    "camera_up": (0.0, -1.0),
    "camera_down": (0.0, 1.0),
    "camera_ul": (-1.0, -1.0),
    "camera_ur": (1.0, -1.0),
    "camera_dl": (-1.0, 1.0),
    "camera_dr": (1.0, 1.0),
}

# Spellings that other WorldFoundry operators already accept.
TOKEN_ALIASES: dict[str, str] = {
    "backward": "back",
    "camera_left": "camera_l",
    "camera_right": "camera_r",
    "camera_u": "camera_up",
    "camera_d": "camera_down",
    "cameraup": "camera_up",
    "cameradown": "camera_down",
    "crouch": "sneak",
    "hotbar.1": "hotbar_1",
    "hotbar.2": "hotbar_2",
    "hotbar.3": "hotbar_3",
    "hotbar.4": "hotbar_4",
    "hotbar.5": "hotbar_5",
    "hotbar.6": "hotbar_6",
    "hotbar.7": "hotbar_7",
    "hotbar.8": "hotbar_8",
    "hotbar.9": "hotbar_9",
    "pickitem": "pick_item",
    "run": "sprint",
    "swaphands": "swap_hands",
}

# Tokens that intentionally produce a no-op frame.
IDLE_TOKENS = frozenset({"", "idle", "no_op", "none", "noop", "stay", "still", "wait"})

_ATOMIC_TOKENS = frozenset(KEYBOARD_TOKENS) | frozenset(MOUSE_BUTTON_TOKENS) | frozenset(CAMERA_TOKENS)


def noop_action() -> dict[str, Any]:
    """Return a single VPT action dictionary that presses nothing."""
    return {
        "mouse": {"dx": 0.0, "dy": 0.0, "buttons": [], "dwheel": 0.0},
        "keyboard": {"keys": []},
    }


def _canonical(token: str) -> str:
    """Lower-case and normalize separators for a single interaction token."""
    text = str(token).strip().lower().replace("-", "_").replace(" ", "_")
    return TOKEN_ALIASES.get(text, text)


def split_interaction_token(token: str) -> list[str]:
    """Split a compound interaction name into atomic tokens.

    ``forward_camera_l`` becomes ``["forward", "camera_l"]`` and ``forward_left``
    becomes ``["forward", "left"]``. Unknown names raise so a typo surfaces as a
    clear error instead of a silently idle rollout.
    """
    text = _canonical(token)
    if text in IDLE_TOKENS:
        return []
    if text in _ATOMIC_TOKENS:
        return [text]

    parts = text.split("_")
    resolved: list[str] = []
    index = 0
    while index < len(parts):
        # Prefer the longest atomic token so `camera_l` wins over a bare `camera`.
        for end in range(len(parts), index, -1):
            candidate = _canonical("_".join(parts[index:end]))
            if candidate in _ATOMIC_TOKENS:
                resolved.append(candidate)
                index = end
                break
            if candidate in IDLE_TOKENS:
                index = end
                break
        else:
            known = ", ".join(sorted(_ATOMIC_TOKENS))
            raise ValueError(f"Unknown Open Dreamer interaction {token!r}; known tokens: {known}")
    return resolved


def _apply_tokens(action: dict[str, Any], tokens: Sequence[str], camera_step_degrees: float) -> None:
    """Fold atomic tokens into an in-progress VPT action dictionary."""
    raw_step = camera_step_degrees * RAW_UNITS_PER_DEGREE
    for token in tokens:
        if token in KEYBOARD_TOKENS:
            for key in KEYBOARD_TOKENS[token]:
                if key not in action["keyboard"]["keys"]:
                    action["keyboard"]["keys"].append(key)
        elif token in MOUSE_BUTTON_TOKENS:
            button = MOUSE_BUTTON_TOKENS[token]
            if button not in action["mouse"]["buttons"]:
                action["mouse"]["buttons"].append(button)
        elif token in CAMERA_TOKENS:
            dx, dy = CAMERA_TOKENS[token]
            action["mouse"]["dx"] += dx * raw_step
            action["mouse"]["dy"] += dy * raw_step


def interaction_to_action(
    interaction: Any,
    *,
    camera_step_degrees: float = DEFAULT_CAMERA_STEP_DEGREES,
) -> dict[str, Any]:
    """Render one WorldFoundry interaction as a single VPT action dictionary.

    Args:
        interaction: An interaction token (``"forward_camera_l"``), an iterable of
            tokens, or a mapping. A mapping may already be a VPT action dict, or
            it may use the structured form
            ``{"keys": [...], "buttons": [...], "camera": [dx_deg, dy_deg]}``.
        camera_step_degrees: Degrees of camera motion contributed by one camera
            token. Ignored when the mapping form supplies explicit deltas.

    Returns:
        A VPT action dictionary with ``mouse`` and ``keyboard`` fields.
    """
    action = noop_action()

    if isinstance(interaction, Mapping):
        if "mouse" in interaction or "keyboard" in interaction:
            # Already a VPT action dictionary; normalize the optional fields.
            mouse = dict(interaction.get("mouse") or {})
            keyboard = dict(interaction.get("keyboard") or {})
            action["mouse"] = {
                "dx": float(mouse.get("dx", 0.0)),
                "dy": float(mouse.get("dy", 0.0)),
                "buttons": list(mouse.get("buttons") or []),
                "dwheel": float(mouse.get("dwheel", 0.0) or 0.0),
            }
            action["keyboard"] = {"keys": list(keyboard.get("keys") or [])}
            return action

        tokens: list[str] = []
        for key in ("keys", "buttons", "actions", "interaction", "action"):
            value = interaction.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                tokens.extend(split_interaction_token(value))
            else:
                for item in value:
                    tokens.extend(split_interaction_token(str(item)))
        _apply_tokens(action, tokens, camera_step_degrees)

        camera = interaction.get("camera")
        if camera is not None:
            dx_deg, dy_deg = (float(camera[0]), float(camera[1]))
            action["mouse"]["dx"] += dx_deg * RAW_UNITS_PER_DEGREE
            action["mouse"]["dy"] += dy_deg * RAW_UNITS_PER_DEGREE
        return action

    if isinstance(interaction, str) or not isinstance(interaction, Iterable):
        tokens = split_interaction_token(str(interaction))
    else:
        tokens = []
        for item in interaction:
            tokens.extend(split_interaction_token(str(item)))
    _apply_tokens(action, tokens, camera_step_degrees)
    return action


def _copy_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent copy of a VPT action dictionary."""
    mouse = action["mouse"]
    return {
        "mouse": {
            "dx": mouse["dx"],
            "dy": mouse["dy"],
            "buttons": list(mouse["buttons"]),
            "dwheel": mouse["dwheel"],
        },
        "keyboard": {"keys": list(action["keyboard"]["keys"])},
    }


def _interaction_frames(interaction: Any) -> int | None:
    """Return an explicit per-interaction frame count when the caller supplied one."""
    if isinstance(interaction, Mapping):
        for key in ("frames", "num_frames", "repeat"):
            value = interaction.get(key)
            if value is not None:
                return max(int(value), 1)
    return None


def build_action_dicts(
    interactions: Sequence[Any] | None,
    *,
    context_frames: int,
    horizon: int,
    camera_step_degrees: float = DEFAULT_CAMERA_STEP_DEGREES,
) -> list[dict[str, Any]]:
    """Build the full VPT action sequence for one Open Dreamer rollout.

    The first ``context_frames`` entries cover the observed context clip. The true
    actions behind those frames are unknown to WorldFoundry, so they are emitted
    as no-ops; the remaining ``horizon`` entries hold the requested interactions
    spread evenly across the predicted frames.

    Args:
        interactions: Interaction tokens, mappings, or ``None`` for the default
            forward-and-look schedule.
        context_frames: Number of observed context frames fed to the model.
        horizon: Number of frames the model predicts.
        camera_step_degrees: Degrees of camera motion per camera token.

    Returns:
        A list of ``context_frames + horizon`` VPT action dictionaries.
    """
    if context_frames < 1:
        raise ValueError("context_frames must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    schedule = [item for item in (interactions or ()) if item not in (None, "")]
    if not schedule:
        schedule = list(DEFAULT_INTERACTIONS)

    actions = [noop_action() for _ in range(context_frames)]

    explicit = [_interaction_frames(item) for item in schedule]
    if any(count is not None for count in explicit):
        # Honour caller-supplied frame counts, then pad or trim to the horizon.
        for interaction, count in zip(schedule, explicit):
            action = interaction_to_action(interaction, camera_step_degrees=camera_step_degrees)
            actions.extend(_copy_action(action) for _ in range(count if count is not None else 1))
    else:
        base, remainder = divmod(horizon, len(schedule))
        for index, interaction in enumerate(schedule):
            count = base + (1 if index < remainder else 0)
            if count < 1:
                continue
            action = interaction_to_action(interaction, camera_step_degrees=camera_step_degrees)
            actions.extend(_copy_action(action) for _ in range(count))

    total = context_frames + horizon
    while len(actions) < total:
        actions.append(noop_action())
    return actions[:total]


def write_action_jsonl(path: str | Path, actions: Sequence[Mapping[str, Any]]) -> Path:
    """Write VPT action dictionaries as JSONL and return the resolved path."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action, ensure_ascii=False, sort_keys=True) + "\n")
    return target.resolve()


def count_action_entries(path: str | Path) -> int:
    """Count actions in a caller-supplied VPT JSON array or JSONL file."""
    text = Path(path).expanduser().read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return 0
    if stripped[0] == "[":
        payload = json.loads(text)
        return len(payload) if isinstance(payload, list) else 0
    return sum(1 for line in text.splitlines() if line.strip())


__all__ = [
    "CAMERA_TOKENS",
    "DEFAULT_CAMERA_STEP_DEGREES",
    "DEFAULT_INTERACTIONS",
    "IDLE_TOKENS",
    "KEYBOARD_TOKENS",
    "MOUSE_BUTTON_TOKENS",
    "RAW_UNITS_PER_DEGREE",
    "TOKEN_ALIASES",
    "build_action_dicts",
    "count_action_entries",
    "interaction_to_action",
    "noop_action",
    "split_interaction_token",
    "write_action_jsonl",
]
