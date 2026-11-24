"""Optional real-generation backend hooks.

WRBench compiles camera payloads and sidecars with ``dry_run=True``. Backends
connect the compiled payload to a model's native inference environment when
weights, GPU, and venv are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from wrbench.payload import CameraPayload


@dataclass(frozen=True)
class GenerationRequest:
    """Everything needed to run real generation after ``compile_camera``."""

    model: str
    prompt: str
    payload: CameraPayload
    output_path: Path
    image_path: Path | None = None
    source_video_path: Path | None = None
    work_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    success: bool
    output_path: Path | None = None
    message: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)


class GenerationBackend(Protocol):
    """Model-specific backend that consumes a compiled ``CameraPayload``."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Return (is_available, reason_if_not)."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run real inference. Must not be called in dry-run workflows."""
        ...


class DryRunBackend:
    """Placeholder backend: compile-only, no real generation."""

    name = "dry_run"

    def available(self) -> tuple[bool, str]:
        return True, "dry-run always available"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            success=False,
            message=(
                "Dry-run backend does not invoke model inference. "
                "Use compile_camera(dry_run=True) or wire a model-specific backend."
            ),
        )


def default_backend() -> GenerationBackend:
    return DryRunBackend()
