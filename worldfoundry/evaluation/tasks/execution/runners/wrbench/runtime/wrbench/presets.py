"""Preset camera-motion combinations plus arbitrary go-return builders.

Presets return a :class:`~wrbench.actions.CameraScript`. They are thin, transparent
wrappers over the frame-action grammar so researchers can either use a named
combination (``yaw_LR``) or compose arbitrary angles/translations directly.

Go-return semantics: a ``*_LR`` preset rotates/translates one way for the first
half of the frames, then returns the opposite way for the rest, ending near the
original pose. Use :func:`sweep` for a one-directional motion of any angle.

Compound presets: :func:`arc_LR`, :func:`arc_RL`, and :func:`orbit` combine
simultaneous rotation and translation in a single segment using
:meth:`~wrbench.actions.CameraScript.segment`.
"""

from __future__ import annotations

from typing import Callable

from wrbench.actions import CameraScript


DEFAULT_FRAMES = 81
DEFAULT_YAW_PEAK_DEG = 60.0
DEFAULT_PAN_AMOUNT = 0.5
DEFAULT_DOLLY_AMOUNT = 1.0


def _split(frames: int) -> tuple[int, int]:
    half = max(1, int(frames) // 2)
    rest = max(1, int(frames) - half)
    return half, rest


def static(frames: int = DEFAULT_FRAMES) -> CameraScript:
    return CameraScript().static(frames=int(frames))


def yaw_LR(peak_deg: float = DEFAULT_YAW_PEAK_DEG, frames: int = DEFAULT_FRAMES) -> CameraScript:
    """Yaw left to ``peak_deg`` then return right (go-return)."""
    half, rest = _split(frames)
    return CameraScript().yaw("left", degrees=peak_deg, frames=half).yaw("right", degrees=peak_deg, frames=rest)


def yaw_RL(peak_deg: float = DEFAULT_YAW_PEAK_DEG, frames: int = DEFAULT_FRAMES) -> CameraScript:
    """Yaw right to ``peak_deg`` then return left (go-return)."""
    half, rest = _split(frames)
    return CameraScript().yaw("right", degrees=peak_deg, frames=half).yaw("left", degrees=peak_deg, frames=rest)


def pan_LR(amount: float = DEFAULT_PAN_AMOUNT, frames: int = DEFAULT_FRAMES) -> CameraScript:
    """Pan left then return right (go-return)."""
    half, rest = _split(frames)
    return CameraScript().pan("left", amount=amount, frames=half).pan("right", amount=amount, frames=rest)


def pan_RL(amount: float = DEFAULT_PAN_AMOUNT, frames: int = DEFAULT_FRAMES) -> CameraScript:
    """Pan right then return left (go-return)."""
    half, rest = _split(frames)
    return CameraScript().pan("right", amount=amount, frames=half).pan("left", amount=amount, frames=rest)


def arc_LR(
    peak_deg: float = DEFAULT_YAW_PEAK_DEG,
    dolly_amount: float = DEFAULT_DOLLY_AMOUNT,
    frames: int = DEFAULT_FRAMES,
) -> CameraScript:
    """Arc left (yaw + dolly forward simultaneously) then return (go-return).

    An arc shot keeps a subject roughly in frame while the camera physically
    moves around it.  The camera yaws and dollies forward together during the
    first half, then reverses both to return.

    Example::

        from wrbench.datasets import natural25_first_frame_path

        result = wrbench.compile_camera(
            model="wan22-fun-5b-cam",
            camera=wrbench.presets.arc_LR(peak_deg=40, dolly_amount=0.8),
            image=natural25_first_frame_path("bedroom_cat_bed_jump"),
            out="out.mp4",
        )
    """
    half, rest = _split(frames)
    return (
        CameraScript()
        .segment(half, yaw_left=peak_deg, dolly_forward=dolly_amount)
        .segment(rest, yaw_right=peak_deg, dolly_back=dolly_amount)
    )


def arc_RL(
    peak_deg: float = DEFAULT_YAW_PEAK_DEG,
    dolly_amount: float = DEFAULT_DOLLY_AMOUNT,
    frames: int = DEFAULT_FRAMES,
) -> CameraScript:
    """Arc right (yaw + dolly forward simultaneously) then return (go-return)."""
    half, rest = _split(frames)
    return (
        CameraScript()
        .segment(half, yaw_right=peak_deg, dolly_forward=dolly_amount)
        .segment(rest, yaw_left=peak_deg, dolly_back=dolly_amount)
    )


def orbit(
    peak_deg: float = DEFAULT_YAW_PEAK_DEG,
    pan_amount: float = DEFAULT_PAN_AMOUNT,
    frames: int = DEFAULT_FRAMES,
) -> CameraScript:
    """Orbit left: yaw and pan simultaneously, then return (go-return).

    Combines yaw rotation with lateral pan for an orbiting motion that keeps
    the subject centered while the camera sweeps around it.
    """
    half, rest = _split(frames)
    return (
        CameraScript()
        .segment(half, yaw_left=peak_deg, pan_right=pan_amount)
        .segment(rest, yaw_right=peak_deg, pan_left=pan_amount)
    )


def sweep(kind: str, direction: str, value: float, frames: int = DEFAULT_FRAMES) -> CameraScript:
    """One-directional motion of any angle/amount, e.g. ``sweep('yaw', 'left', 37)``."""
    script = CameraScript()
    if kind in {"yaw", "pitch", "roll"}:
        getattr(script, kind)(direction, degrees=value, frames=int(frames))
    elif kind in {"pan", "dolly", "crane"}:
        getattr(script, kind)(direction, amount=value, frames=int(frames))
    else:
        raise ValueError(f"Unsupported sweep kind {kind!r}")
    return script


def go_return(kind: str, first: str, second: str, value: float, frames: int = DEFAULT_FRAMES) -> CameraScript:
    """Generic go-return for any rotation/translation kind and directions."""
    half, rest = _split(frames)
    script = CameraScript()
    is_rotation = kind in {"yaw", "pitch", "roll"}
    kw = "degrees" if is_rotation else "amount"
    getattr(script, kind)(first, frames=half, **{kw: value})
    getattr(script, kind)(second, frames=rest, **{kw: value})
    return script


PRESETS: dict[str, Callable[..., CameraScript]] = {
    "static": static,
    "yaw_LR": yaw_LR,
    "yaw_RL": yaw_RL,
    "pan_LR": pan_LR,
    "pan_RL": pan_RL,
    "arc_LR": arc_LR,
    "arc_RL": arc_RL,
    "orbit": orbit,
}


def preset_names() -> list[str]:
    return list(PRESETS)


def build_preset(name: str, **kwargs) -> CameraScript:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; valid: {preset_names()}")
    return PRESETS[name](**kwargs)
