"""Native embodied simulator interfaces and registry."""

from worldfoundry.evals.tasks.embodied.simulators.base import BaseSimulator, StepResult
from worldfoundry.evals.tasks.embodied.simulators.registry import (
    SIMULATOR_ENTRIES,
    SimulatorEntry,
    get_simulator_entry,
    list_simulator_ids,
    resolve_simulator_class,
)

__all__ = [
    "SIMULATOR_ENTRIES",
    "BaseSimulator",
    "SimulatorEntry",
    "StepResult",
    "get_simulator_entry",
    "list_simulator_ids",
    "resolve_simulator_class",
]
