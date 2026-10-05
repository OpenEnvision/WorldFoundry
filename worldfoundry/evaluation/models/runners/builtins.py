"""Built-in runtime runner definitions for the WorldFoundry model runner registry.

Registers the default ``worldfoundry.pipeline`` runner (and its aliases) plus
the embodied closed-loop runner.  The embodied runner lives in
``worldfoundry.evaluation.tasks`` (a higher layer than ``models``), so it is
declared as a lazy ``module:Class`` target and only imported when a caller
actually resolves it — keeping model-resolution imports free of ``tasks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldfoundry.evaluation.models.import_target import import_dotted_attr
from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner

_EMBODIED_RUNNER_TARGET = "worldfoundry.evaluation.tasks.embodied.rollout_runner:EmbodiedClosedLoopRunner"


@dataclass(frozen=True)
class BuiltinRuntimeRunnerEntry:
    """Configuration entry for a built-in runtime runner.

    Attributes:
        name: Primary identifier for the runner.
        runner_target: Lazy ``module:Class`` import target for the runner.
        runner_class: Optional eagerly-bound runner class (used for
            same-layer runners; cross-layer runners stay lazy via
            ``runner_target``).
        aliases: Alternative lookup keys that resolve to this entry.
        description: Human-readable summary of the runner's purpose.
    """

    name: str
    runner_target: str = ""
    runner_class: type | None = None
    aliases: tuple[str, ...] = ()
    description: str = ""

    def keys(self) -> tuple[str, ...]:
        """Compile all unique lookup keys (name and aliases) for this entry."""
        return tuple(dict.fromkeys((self.name, *self.aliases)))

    def resolve_runner_class(self) -> type:
        """Return the runner class, importing the lazy target on first use."""
        if self.runner_class is not None:
            return self.runner_class
        return import_dotted_attr(self.runner_target)

    def to_dict(self) -> dict[str, Any]:
        """Convert the builtin runner entry to a JSON-friendly dictionary."""
        runner_class = self.runner_target
        if not runner_class and self.runner_class is not None:
            runner_class = f"{self.runner_class.__module__}:{self.runner_class.__qualname__}"
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "runner_class": runner_class,
            "runner_target": self.runner_target,
            "description": self.description,
        }


# ── Builtin runner entries ─────────────────────────────────────────────
BUILTIN_RUNTIME_RUNNERS: tuple[BuiltinRuntimeRunnerEntry, ...] = (
    BuiltinRuntimeRunnerEntry(
        name="worldfoundry.pipeline",
        aliases=(
            "worldfoundry:pipeline",
            "worldfoundry-pipeline",
        ),
        runner_target="worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner",
        runner_class=WorldFoundryPipelineRunner,
        description="WorldFoundry pipeline runner for data-backed pipeline bindings and runtime profiles.",
    ),
    BuiltinRuntimeRunnerEntry(
        name="worldfoundry.embodied-closed-loop",
        aliases=(
            "worldfoundry:embodied-closed-loop",
            "embodied-closed-loop",
            "embodied.rollout",
        ),
        # Lazy target: importing tasks.embodied pulls numpy and the tasks
        # layer, which model resolution must not pay for (models must not
        # depend on tasks at import time).
        runner_target=_EMBODIED_RUNNER_TARGET,
        description="Native embodied simulator closed-loop rollout runner.",
    ),
)


def _runner_key(value: str) -> str:
    """Normalize a runner target key for matching."""
    return value.strip().lower().replace("_", "-")


def get_builtin_runtime_runner_class(name: str) -> type | None:
    """Retrieve a built-in runtime runner class by name or alias.

    Performs case-insensitive, dash-normalised lookup across all
    :data:`BUILTIN_RUNTIME_RUNNERS` entries.  Lazy entries import their
    ``runner_target`` on first resolution.

    Args:
        name: Runner name or alias to search for.

    Returns:
        The matching runner class, or ``None`` if no entry matches.
    """
    key = _runner_key(name)
    for entry in BUILTIN_RUNTIME_RUNNERS:
        if key in {_runner_key(item) for item in entry.keys()}:
            return entry.resolve_runner_class()
    return None


def list_builtin_runtime_runners() -> tuple[BuiltinRuntimeRunnerEntry, ...]:
    """Return the list of all registered built-in runtime runners."""
    return BUILTIN_RUNTIME_RUNNERS


def __getattr__(name: str) -> Any:
    """Lazily expose ``EmbodiedClosedLoopRunner`` (lives in the tasks layer)."""
    if name == "EmbodiedClosedLoopRunner":
        value = import_dotted_attr(_EMBODIED_RUNNER_TARGET)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BUILTIN_RUNTIME_RUNNERS",
    "BuiltinRuntimeRunnerEntry",
    "EmbodiedClosedLoopRunner",
    "WorldFoundryPipelineRunner",
    "get_builtin_runtime_runner_class",
    "list_builtin_runtime_runners",
]
