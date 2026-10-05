"""Generic frame sampling policy for PAWEval evidence packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_EVIDENCE_FPS = 2.0


@dataclass(frozen=True)
class FrameSamplingSpec:
    mode: str = "fps"
    fps: float = DEFAULT_EVIDENCE_FPS
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool):
            raise ValueError("sampling fps must be numeric")
        fps = float(self.fps)
        if self.mode != "fps":
            raise ValueError(f"unsupported PAWEval sampling mode: {self.mode}")
        if fps <= 0:
            raise ValueError("sampling fps must be > 0")
        object.__setattr__(self, "fps", fps)
        if self.max_frames is not None:
            if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int):
                raise ValueError("sampling max_frames must be an integer when provided")
            if self.max_frames < 3:
                raise ValueError("sampling max_frames must be >= 3 when provided for PAWEval")
            object.__setattr__(self, "max_frames", int(self.max_frames))

def sampling_spec_from_value(value: float | int | Mapping[str, Any] | FrameSamplingSpec | None) -> FrameSamplingSpec:
    if value is None:
        return FrameSamplingSpec()
    if isinstance(value, FrameSamplingSpec):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return FrameSamplingSpec(fps=float(value))
    if isinstance(value, Mapping):
        fps = value.get("fps", DEFAULT_EVIDENCE_FPS)
        if not isinstance(fps, (int, float)) or isinstance(fps, bool):
            raise ValueError("sampling fps must be numeric")
        max_frames = value.get("max_frames")
        if max_frames is not None and (not isinstance(max_frames, int) or isinstance(max_frames, bool)):
            raise ValueError("sampling max_frames must be an integer when provided")
        return FrameSamplingSpec(
            mode=str(value.get("mode") or "fps"),
            fps=float(fps),
            max_frames=max_frames,
        )
    raise TypeError(f"unsupported sampling spec type: {type(value).__name__}")
