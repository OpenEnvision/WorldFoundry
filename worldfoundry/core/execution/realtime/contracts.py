"""Framework contracts for resident interactive world models.

The browser transport and Studio scheduler must not know a model's latent
stride, causal seed cadence, or preferred control vocabulary.  A model-owned
resident session reports those details through :class:`RealtimeSpec` while
keeping the hot-path API deliberately small: configure once, advance one
chunk, and reset without unloading weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

DEFAULT_REALTIME_CONTROLS = (
    "forward",
    "backward",
    "left",
    "right",
    "camera_up",
    "camera_down",
    "camera_l",
    "camera_r",
)


@dataclass(frozen=True, slots=True)
class RealtimeSpec:
    """Model-owned playback and generation cadence for one resident session."""

    fps: int = 16
    first_chunk_frames: int = 9
    steady_chunk_frames: int = 9
    controls: tuple[str, ...] = DEFAULT_REALTIME_CONTROLS
    transport: str = "in-memory-rgb"
    stateful: bool = True
    input_fps: float | None = None
    first_input_frames: int | None = None
    steady_input_frames: int | None = None

    def __post_init__(self) -> None:
        """Reject non-positive clocks or empty control vocabularies before a session starts."""
        if self.fps < 1:
            raise ValueError("RealtimeSpec.fps must be positive.")
        if self.first_chunk_frames < 1 or self.steady_chunk_frames < 1:
            raise ValueError("RealtimeSpec chunk frame counts must be positive.")
        if self.input_fps is not None and self.input_fps <= 0.0:
            raise ValueError("RealtimeSpec.input_fps must be positive when provided.")
        if self.first_input_frames is not None and self.first_input_frames < 1:
            raise ValueError("RealtimeSpec.first_input_frames must be positive when provided.")
        if self.steady_input_frames is not None and self.steady_input_frames < 1:
            raise ValueError("RealtimeSpec.steady_input_frames must be positive when provided.")
        if not self.controls:
            raise ValueError("RealtimeSpec.controls must not be empty.")

    @property
    def resolved_input_fps(self) -> float:
        """Control/input sampling FPS, defaulting to the playback clock."""

        return float(self.input_fps if self.input_fps is not None else self.fps)

    @property
    def resolved_first_input_frames(self) -> int:
        """First control chunk size, defaulting to first output frames."""

        return int(
            self.first_input_frames
            if self.first_input_frames is not None
            else self.first_chunk_frames
        )

    @property
    def resolved_steady_input_frames(self) -> int:
        """Steady control chunk size, defaulting to steady output frames."""

        return int(
            self.steady_input_frames
            if self.steady_input_frames is not None
            else self.steady_chunk_frames
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe dict; omit unset optional input-clock fields so clients do not treat ``null`` as 0."""
        payload = asdict(self)
        for optional_name in (
            "input_fps",
            "first_input_frames",
            "steady_input_frames",
        ):
            if payload[optional_name] is None:
                payload.pop(optional_name)
        payload["controls"] = list(self.controls)
        return payload

    @classmethod
    def from_payload(
        cls,
        value: Any,
        *,
        fallback: "RealtimeSpec | None" = None,
    ) -> "RealtimeSpec":
        """Parse a model result without letting malformed metadata break play."""

        default = fallback or cls()
        if isinstance(value, Mapping) and isinstance(value.get("realtime_spec"), Mapping):
            value = value["realtime_spec"]
        if not isinstance(value, Mapping):
            return default
        try:
            controls = value.get("controls", default.controls)
            if isinstance(controls, str):
                controls = tuple(item.strip() for item in controls.split(",") if item.strip())
            else:
                controls = tuple(str(item) for item in controls)
            input_fps = value.get("input_fps", default.input_fps)
            first_input_frames = value.get(
                "first_input_frames",
                default.first_input_frames,
            )
            steady_input_frames = value.get(
                "steady_input_frames",
                default.steady_input_frames,
            )
            return cls(
                fps=int(value.get("fps", default.fps)),
                first_chunk_frames=int(
                    value.get("first_chunk_frames", default.first_chunk_frames)
                ),
                steady_chunk_frames=int(
                    value.get("steady_chunk_frames", default.steady_chunk_frames)
                ),
                input_fps=float(input_fps) if input_fps is not None else None,
                first_input_frames=(
                    int(first_input_frames)
                    if first_input_frames is not None
                    else None
                ),
                steady_input_frames=(
                    int(steady_input_frames)
                    if steady_input_frames is not None
                    else None
                ),
                controls=controls or default.controls,
                transport=str(value.get("transport", default.transport)),
                stateful=bool(value.get("stateful", default.stateful)),
            )
        except (TypeError, ValueError):
            return default


@runtime_checkable
class InteractiveWorldPipeline(Protocol):
    """Structural API implemented by model-specific resident adapters."""

    def prepare_realtime(self) -> Mapping[str, Any] | None:
        """Optional one-time setup; return metadata the scheduler may advertise."""
        ...

    def configure_realtime(self, images: Any, prompt: str = "", **kwargs: Any) -> Any:
        """Seed the resident session (first frames / prompt) without unloading weights."""
        ...

    def stream_realtime(self, interactions: list[str], **kwargs: Any) -> Any:
        """Advance one control chunk and return the next RGB frames."""
        ...

    def reset_realtime(self) -> None:
        """Clear session state while keeping loaded weights resident."""
        ...


__all__ = [
    "DEFAULT_REALTIME_CONTROLS",
    "InteractiveWorldPipeline",
    "RealtimeSpec",
]
