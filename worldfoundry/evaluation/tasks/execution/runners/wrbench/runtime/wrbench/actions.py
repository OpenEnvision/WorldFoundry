"""Frame-level camera action grammar for unified camera control.

A ``CameraScript`` is an ordered sequence of frame segments. Each segment
covers a contiguous run of frames and expresses one or more simultaneous
camera intents:

- rotation: ``yaw`` / ``pitch`` / ``roll`` with ``degrees`` (any angle).
- translation: ``pan`` / ``dolly`` / ``crane`` with ``amount`` (any direction).
- ``static``: hold the camera still.

**Compact string form** – single action per segment:

    kind:direction:value@frames[,kind:direction:value@frames,...]

e.g. ``yaw:left:60@40,yaw:right:60@41`` or ``static@81``.

**Compound segments** – simultaneous actions within one time window,
joined by ``+`` inside a comma-separated segment:

    yaw:left:60+dolly:forward:1.0@40,yaw:right:60@41

All actions before the ``@frames`` suffix run at the same time.

**Python segment API** – same result as the compound string, more readable:

    CameraScript().segment(40, yaw_left=60, dolly_forward=1.0)

Keyword format: ``kind_direction=value``, e.g. ``yaw_left=60``,
``dolly_forward=1.0``, ``pan_right=0.5``, ``crane_up=0.3``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


ROTATION_KINDS = {"yaw", "pitch", "roll"}
TRANSLATION_KINDS = {"pan", "dolly", "crane"}
ACTION_KINDS = ROTATION_KINDS | TRANSLATION_KINDS | {"static"}
ROTATION_DIRECTIONS = {
    "yaw": {"left", "right"},
    "pitch": {"up", "down"},
    "roll": {"left", "right", "cw", "ccw"},
}
TRANSLATION_DIRECTIONS = {
    "pan": {"left", "right"},
    "dolly": {"forward", "back", "backward"},
    "crane": {"up", "down"},
}


@dataclass(frozen=True)
class FrameAction:
    """One contiguous frame segment of camera intent.

    When ``simultaneous=True`` the action shares its frame window with the
    *preceding* action in the script (they run at the same time).  All
    simultaneous actions in a group must carry the same ``frames`` value;
    use :meth:`CameraScript.segment` or the ``+`` string syntax to build
    groups correctly.
    """

    kind: str
    direction: str = "none"
    frames: int | None = None
    degrees: float | None = None
    amount: float | None = None
    simultaneous: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower().replace("_", "-")
        direction = str(self.direction or "none").strip().lower()
        if kind not in ACTION_KINDS:
            raise ValueError(f"Unsupported action kind {self.kind!r}; valid choices: {sorted(ACTION_KINDS)}")
        if self.frames is not None and int(self.frames) <= 0:
            raise ValueError("frames must be positive when provided")
        if kind == "static":
            direction = "none"
        elif kind in ROTATION_KINDS:
            valid = ROTATION_DIRECTIONS[kind]
            if direction not in valid:
                raise ValueError(f"Unsupported {kind} direction {direction!r}; valid choices: {sorted(valid)}")
            if self.degrees is None:
                raise ValueError(f"{kind} action requires degrees")
        elif kind in TRANSLATION_KINDS:
            valid = TRANSLATION_DIRECTIONS[kind]
            if direction not in valid:
                raise ValueError(f"Unsupported {kind} direction {direction!r}; valid choices: {sorted(valid)}")
            if self.amount is None:
                raise ValueError(f"{kind} action requires amount")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "direction", direction)
        if self.frames is not None:
            object.__setattr__(self, "frames", int(self.frames))

    def with_frames(self, frames: int) -> "FrameAction":
        return replace(self, frames=int(frames))


def _parse_segment_kwargs(frames: int, kwargs: dict[str, float]) -> list[FrameAction]:
    """Parse ``kind_direction=value`` keyword arguments into :class:`FrameAction` objects."""
    actions: list[FrameAction] = []
    for key, value in kwargs.items():
        sep_idx = key.index("_") if "_" in key else -1
        if sep_idx < 1:
            raise ValueError(
                f"segment kwarg {key!r} must be kind_direction (e.g. yaw_left=60, dolly_forward=1.0)"
            )
        kind = key[:sep_idx]
        direction = key[sep_idx + 1 :]
        if kind in ROTATION_KINDS:
            actions.append(FrameAction(kind, direction, frames=frames, degrees=float(value)))
        elif kind in TRANSLATION_KINDS:
            actions.append(FrameAction(kind, direction, frames=frames, amount=float(value)))
        else:
            raise ValueError(
                f"Unknown motion kind {kind!r} in segment kwarg {key!r}; "
                f"valid: {sorted(ROTATION_KINDS | TRANSLATION_KINDS)}"
            )
    return actions


class CameraScript:
    """Builder for a sequence of frame-level camera actions.

    Individual motion methods (``yaw``, ``pan``, …) append a single action.
    Use :meth:`segment` to combine multiple simultaneous motions into one
    time window.  All methods return ``self`` for chaining.
    """

    def __init__(self, fps: int = 16, actions: list[FrameAction] | None = None) -> None:
        self.fps = int(fps)
        self.actions: list[FrameAction] = list(actions or [])

    @property
    def frame_count(self) -> int | None:
        """Total frame count; simultaneous actions do not add to the total."""
        total = 0
        for action in self.actions:
            if action.simultaneous:
                continue
            if action.frames is None:
                return None
            total += int(action.frames)
        return total

    def extend(self, action: FrameAction) -> "CameraScript":
        self.actions.append(action)
        return self

    def static(self, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("static", frames=frames))

    def yaw(self, direction: str, *, degrees: float, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("yaw", direction, frames=frames, degrees=float(degrees)))

    def pitch(self, direction: str, *, degrees: float, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("pitch", direction, frames=frames, degrees=float(degrees)))

    def roll(self, direction: str, *, degrees: float, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("roll", direction, frames=frames, degrees=float(degrees)))

    def pan(self, direction: str, *, amount: float, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("pan", direction, frames=frames, amount=float(amount)))

    def dolly(self, direction: str, *, amount: float, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("dolly", direction, frames=frames, amount=float(amount)))

    def crane(self, direction: str, *, amount: float, frames: int | None = None) -> "CameraScript":
        return self.extend(FrameAction("crane", direction, frames=frames, amount=float(amount)))

    def segment(self, frames: int, /, **kwargs: float) -> "CameraScript":
        """Add one or more simultaneous motions over a single time window.

        Each keyword argument has the form ``kind_direction=value``::

            # Orbit: yaw left while pushing forward over 40 frames
            CameraScript().segment(40, yaw_left=60, dolly_forward=1.0)

            # Diagonal look: yaw + pitch together
            CameraScript().segment(40, yaw_left=30, pitch_down=15)

            # Chained compound segments (go-return orbit)
            script = (
                CameraScript()
                .segment(40, yaw_left=60, dolly_forward=1.0)
                .segment(41, yaw_right=60, dolly_back=1.0)
            )

        Valid ``kind`` prefixes: ``yaw``, ``pitch``, ``roll`` (rotation) and
        ``pan``, ``dolly``, ``crane`` (translation).

        Valid ``direction`` suffixes match :data:`ROTATION_DIRECTIONS` and
        :data:`TRANSLATION_DIRECTIONS` (e.g. ``yaw_left``, ``dolly_forward``,
        ``crane_up``, ``roll_cw``).

        At least one keyword argument is required.  Static segments can still
        use :meth:`static`.
        """
        if not kwargs:
            raise ValueError("segment() requires at least one motion keyword argument")
        parts = _parse_segment_kwargs(int(frames), kwargs)
        for i, action in enumerate(parts):
            if i > 0:
                action = replace(action, simultaneous=True)
            self.actions.append(action)
        return self

    def to_string(self) -> str:
        """Serialize to the compact string grammar, using ``+`` for simultaneous actions."""
        parts: list[str] = []
        for action in self.actions:
            if action.kind == "static":
                body = "static"
            elif action.kind in ROTATION_KINDS:
                body = f"{action.kind}:{action.direction}:{action.degrees:g}"
            else:
                body = f"{action.kind}:{action.direction}:{action.amount:g}"

            if action.simultaneous and parts:
                # Extend the current group: strip the @frames suffix from the last
                # entry, append the new body, then re-add @frames.
                last = parts[-1]
                at_idx = last.rfind("@")
                if at_idx >= 0:
                    group_body = last[:at_idx]
                    frame_suffix = last[at_idx:]
                else:
                    group_body = last
                    frame_suffix = f"@{action.frames}" if action.frames is not None else ""
                parts[-1] = f"{group_body}+{body}{frame_suffix}"
            else:
                suffix = f"@{action.frames}" if action.frames is not None else ""
                parts.append(f"{body}{suffix}")
        return ",".join(parts)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CameraScript) and self.fps == other.fps and self.actions == other.actions

    def __repr__(self) -> str:
        return f"CameraScript(fps={self.fps}, actions={self.to_string()!r})"


def parse_camera_script(text: str, *, fps: int = 16) -> CameraScript:
    """Parse the compact camera-action grammar into a :class:`CameraScript`.

    Supports single actions per segment::

        yaw:left:60@40,yaw:right:60@41

    And compound segments (simultaneous actions joined with ``+``)::

        yaw:left:60+dolly:forward:1.0@40,yaw:right:60@41

    The ``@frames`` suffix at the end of a comma-separated group applies to
    all ``+``-joined actions within that group.
    """

    script = CameraScript(fps=fps)
    if not str(text).strip():
        raise ValueError("camera script must not be empty")
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        # Extract @frames from the very end of the comma-separated part.
        body, sep, frame_text = part.partition("@")
        frames = int(frame_text) if sep else None

        # Split body on `+` for simultaneous sub-actions.
        sub_parts = [s.strip() for s in body.split("+")]
        for sub_idx, sub in enumerate(sub_parts):
            fields = sub.split(":")
            kind = fields[0].strip().lower()
            simultaneous = sub_idx > 0

            if kind == "static":
                if len(fields) != 1:
                    raise ValueError("static action format is static@N")
                script.actions.append(FrameAction("static", frames=frames, simultaneous=simultaneous))
                continue

            if len(fields) != 3:
                raise ValueError("camera action format is kind:direction:value@frames")
            direction, value = fields[1], float(fields[2])
            if kind in ROTATION_KINDS:
                script.actions.append(
                    FrameAction(kind, direction, frames=frames, degrees=value, simultaneous=simultaneous)
                )
            elif kind in TRANSLATION_KINDS:
                script.actions.append(
                    FrameAction(kind, direction, frames=frames, amount=value, simultaneous=simultaneous)
                )
            else:
                raise ValueError(f"Unsupported action kind {kind!r}")
    return script
