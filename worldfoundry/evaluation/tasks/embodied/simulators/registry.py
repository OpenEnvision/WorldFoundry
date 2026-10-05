"""Native embodied simulator registry.

Only simulators with an in-tree :class:`BaseSimulator` implementation are listed.
Offline or normalizer-only benchmarks (e.g. BridgeData V2) stay in the
catalog but are not resolved by the closed-loop runner.

Sections:

* **SimulatorEntry** — import path and alias metadata per simulator.
* **SIMULATOR_ENTRIES** — bundled in-tree simulator table.
* **Lookup** — normalize ids, resolve classes dynamically.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulatorEntry:
    """Registry metadata for one supported embodied simulator."""

    benchmark_id: str
    module: str
    class_name: str
    aliases: tuple[str, ...] = ()
    official_config_dir: str | None = None

    @property
    def import_path(self) -> str:
        """Return ``module:class_name`` import path."""
        return f"{self.module}:{self.class_name}"


# ---------------------------------------------------------------------------
# Bundled simulator entries
# ---------------------------------------------------------------------------

SIMULATOR_ENTRIES: tuple[SimulatorEntry, ...] = (
    SimulatorEntry(
        "ai2thor",
        "worldfoundry.evaluation.tasks.embodied.simulators.ai2thor.benchmark",
        "AI2ThorBenchmark",
        aliases=("ai2-thor",),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/ai2thor",
    ),
    SimulatorEntry(
        "behavior1k",
        "worldfoundry.evaluation.tasks.embodied.simulators.behavior1k.benchmark",
        "Behavior1KBenchmark",
        aliases=("behavior-1k", "behavior_1k"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/behavior1k",
    ),
    SimulatorEntry(
        "calvin",
        "worldfoundry.evaluation.tasks.embodied.simulators.calvin.benchmark",
        "CALVINBenchmark",
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/calvin",
    ),
    SimulatorEntry(
        "kinetix",
        "worldfoundry.evaluation.tasks.embodied.simulators.kinetix.benchmark",
        "KinetixBenchmark",
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/kinetix",
    ),
    SimulatorEntry(
        "libero",
        "worldfoundry.evaluation.tasks.embodied.simulators.libero.benchmark",
        "LIBEROBenchmark",
        aliases=("libero-spatial", "libero-object", "libero-goal", "libero-10", "libero-90"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/libero",
    ),
    SimulatorEntry(
        "libero-mem",
        "worldfoundry.evaluation.tasks.embodied.simulators.libero_mem.benchmark",
        "LIBEROMemBenchmark",
        aliases=("libero_mem",),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/libero-mem",
    ),
    SimulatorEntry(
        "libero-plus",
        "worldfoundry.evaluation.tasks.embodied.simulators.libero_plus.benchmark",
        "LIBEROPlusBenchmark",
        aliases=("libero_plus",),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/libero-plus",
    ),
    SimulatorEntry(
        "libero-pro",
        "worldfoundry.evaluation.tasks.embodied.simulators.libero_pro.benchmark",
        "LIBEROProBenchmark",
        aliases=("libero_pro",),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/libero-pro",
    ),
    SimulatorEntry(
        "maniskill2",
        "worldfoundry.evaluation.tasks.embodied.simulators.maniskill2.benchmark",
        "ManiSkill2Benchmark",
        aliases=("maniskill", "mani-skill2", "mani-skill"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/maniskill2",
    ),
    SimulatorEntry(
        "mikasa",
        "worldfoundry.evaluation.tasks.embodied.simulators.mikasa.benchmark",
        "MIKASABenchmark",
        aliases=("mikasa-robo", "mikasa_robo"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/mikasa",
    ),
    SimulatorEntry(
        "molmospaces",
        "worldfoundry.evaluation.tasks.embodied.simulators.molmospaces.benchmark",
        "MolmoSpacesBenchmark",
        aliases=("molmo-spaces", "molmospaces-bench"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/molmospaces",
    ),
    SimulatorEntry(
        "rlbench",
        "worldfoundry.evaluation.tasks.embodied.simulators.rlbench.benchmark",
        "RLBenchBenchmark",
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/rlbench",
    ),
    SimulatorEntry(
        "robocasa",
        "worldfoundry.evaluation.tasks.embodied.simulators.robocasa.benchmark",
        "RoboCasaBenchmark",
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/robocasa",
    ),
    SimulatorEntry(
        "robocerebra",
        "worldfoundry.evaluation.tasks.embodied.simulators.robocerebra.benchmark",
        "RoboCerebraBenchmark",
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/robocerebra",
    ),
    SimulatorEntry(
        "robomme",
        "worldfoundry.evaluation.tasks.embodied.simulators.robomme.benchmark",
        "RoboMMEBenchmark",
        aliases=("robo-mme",),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/robomme",
    ),
    SimulatorEntry(
        "robotwin",
        "worldfoundry.evaluation.tasks.embodied.simulators.robotwin.benchmark",
        "RoboTwinBenchmark",
        aliases=("robotwin2", "robotwin-v2", "robotwin-2.0"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/robotwin",
    ),
    SimulatorEntry(
        "simpler-env",
        "worldfoundry.evaluation.tasks.embodied.simulators.simpler.benchmark",
        "SimplerEnvBenchmark",
        aliases=("simpler", "simplerenv", "simpler_env"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/simpler-env",
    ),
    SimulatorEntry(
        "vlabench",
        "worldfoundry.evaluation.tasks.embodied.simulators.vlabench.benchmark",
        "VLABenchBenchmark",
        aliases=("vla-bench", "vla_bench"),
        official_config_dir="worldfoundry/data/benchmarks/eval_configs/embodied/vlabench",
    ),
)


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def _normalize_id(value: str) -> str:
    """Normalize benchmark id/alias to lowercase hyphen form."""
    return value.strip().lower().replace("_", "-")


def simulator_entry_map() -> dict[str, SimulatorEntry]:
    """Build map from normalized ids/aliases to :class:`SimulatorEntry`."""
    entries: dict[str, SimulatorEntry] = {}
    for entry in SIMULATOR_ENTRIES:
        keys = (entry.benchmark_id, *entry.aliases)
        for key in keys:
            entries[_normalize_id(key)] = entry
    return entries


def get_simulator_entry(benchmark_id: str) -> SimulatorEntry | None:
    """Look up simulator entry by id or alias."""
    return simulator_entry_map().get(_normalize_id(benchmark_id))


def list_simulator_ids() -> tuple[str, ...]:
    """Return primary simulator benchmark ids."""
    return tuple(entry.benchmark_id for entry in SIMULATOR_ENTRIES)


def resolve_simulator_class(benchmark_id: str) -> type[Any]:
    """Import and return simulator class for ``benchmark_id``."""
    entry = get_simulator_entry(benchmark_id)
    if entry is None:
        supported = ", ".join(list_simulator_ids())
        raise KeyError(f"Unsupported embodied simulator benchmark_id={benchmark_id!r}. Supported: {supported}")
    module = importlib.import_module(entry.module)
    value = getattr(module, entry.class_name)
    if not isinstance(value, type):
        raise TypeError(f"Simulator target is not a class: {entry.import_path}")
    return value


__all__ = [
    "SIMULATOR_ENTRIES",
    "SimulatorEntry",
    "get_simulator_entry",
    "list_simulator_ids",
    "resolve_simulator_class",
    "simulator_entry_map",
]
