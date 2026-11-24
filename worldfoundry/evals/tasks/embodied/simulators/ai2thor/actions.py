"""Discrete action token mapping for AI2-THOR embodied rollouts."""

from __future__ import annotations

from typing import Any

DEFAULT_MOVE_MAGNITUDE = 0.25
DEFAULT_ROTATE_DEGREES = 30

ACTION_TOKENS: tuple[str, ...] = (
    "forward",
    "back",
    "left",
    "right",
    "look_up",
    "look_down",
    "interact",
    "pickup",
    "drop",
    "open",
    "close",
    "pass",
)

TOKEN_ALIASES: dict[str, str] = {
    "move_ahead": "forward",
    "move_back": "back",
    "rotate_left": "left",
    "rotate_right": "right",
    "camera_l": "left",
    "camera_r": "right",
    "camera_u": "look_up",
    "camera_d": "look_down",
    "noop": "pass",
    "stop": "pass",
}


def normalize_action_token(raw: Any) -> str:
    """Normalize a policy token or numeric index into a canonical action token."""
    if isinstance(raw, str):
        token = raw.strip().lower()
        return TOKEN_ALIASES.get(token, token)
    if isinstance(raw, (int, float)):
        index = int(raw)
        if 0 <= index < len(ACTION_TOKENS):
            return ACTION_TOKENS[index]
    raise ValueError(f"Unsupported AI2-THOR action token: {raw!r}")


def resolve_action_token(action: dict[str, Any]) -> str:
    """Extract a discrete token from a normalized or raw policy action payload."""
    if "token" in action:
        return normalize_action_token(action["token"])
    if "discrete_action" in action:
        return normalize_action_token(action["discrete_action"])
    actions = action.get("actions")
    if isinstance(actions, str):
        return normalize_action_token(actions)
    if isinstance(actions, (list, tuple)) and actions:
        first = actions[0]
        if isinstance(first, str):
            return normalize_action_token(first)
        return normalize_action_token(first)
    return "pass"


def token_to_thor_action(
    token: str,
    *,
    rotate_degrees: float = DEFAULT_ROTATE_DEGREES,
    move_magnitude: float = DEFAULT_MOVE_MAGNITUDE,
) -> dict[str, Any]:
    """Map a WorldFoundry discrete token to an AI2-THOR controller action dict."""
    normalized = normalize_action_token(token)
    mapping: dict[str, dict[str, Any]] = {
        "forward": {"action": "MoveAhead", "moveMagnitude": move_magnitude},
        "back": {"action": "MoveBack", "moveMagnitude": move_magnitude},
        "left": {"action": "RotateLeft", "degrees": rotate_degrees},
        "right": {"action": "RotateRight", "degrees": rotate_degrees},
        "look_up": {"action": "LookUp", "degrees": rotate_degrees},
        "look_down": {"action": "LookDown", "degrees": rotate_degrees},
        "interact": {"action": "Pass"},
        "pickup": {"action": "Pickup", "forceAction": True},
        "drop": {"action": "Drop", "forceAction": True},
        "open": {"action": "OpenObject", "forceAction": True},
        "close": {"action": "CloseObject", "forceAction": True},
        "pass": {"action": "Pass"},
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported AI2-THOR token {token!r}. Supported: {', '.join(ACTION_TOKENS)}")
    return dict(mapping[normalized])


__all__ = [
    "ACTION_TOKENS",
    "DEFAULT_MOVE_MAGNITUDE",
    "DEFAULT_ROTATE_DEGREES",
    "normalize_action_token",
    "resolve_action_token",
    "token_to_thor_action",
]
