"""Benchmark-zoo manifest runner: prepare → run → collect → normalize.

Maps checked-in benchmark-zoo YAML entries onto :class:`ManifestBenchmarkRunner`
instances that drive official subprocesses or contract fixtures and emit
``scorecard.json``.

Sections:

* **Constants** — normalizer script maps and embodied track routing.
* **Runtime helpers** — command/env/results-path resolution for subprocess runs.
* **Normalizers** — specialized, embodied, and generic official-result paths.
* **ManifestBenchmarkRunner** — per-benchmark lifecycle orchestrator.
* **Registry** — :class:`BenchmarkRunnerRegistry` lookup and ``run_benchmark_execution``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from worldfoundry.core.io.paths import resolve_worldfoundry_path
from worldfoundry.core.io.serialization import write_json
from worldfoundry.core.time import utc_now_iso
from worldfoundry.evaluation.reporting import inspect_scorecard_runtime_flags
from worldfoundry.runtime.env import benchmark_repo_cache_root

from ....utils import BENCHMARK_ZOO_DIR
from ...catalog.schema import BenchmarkZooEntry, load_entries
from ...catalog.zoo_registry import BenchmarkZooRegistry, UnknownBenchmarkZooKeyError, load_benchmark_zoo_registry
from ...contracts.external import get_external_benchmark_contract
from ..runners._benchmark_metrics import evaluate_external_metric, list_external_metric_evaluators
from ...execution.framework.benchmark_contract_registry import (
    BENCHMARK_CONTRACT_EVALUATOR_KINDS,
    has_benchmark_contract_evaluator,
    write_benchmark_contract_evaluation,
)
from worldfoundry.evaluation.tasks.execution.framework.result_normalizer import OfficialResultsNormalizer
from ..framework.runner_registry import specialized_result_normalizer_scripts
from .interfaces import (
    BenchmarkSample,
    DatasetMaterializationPlan,
    OfficialRunResult,
    OfficialRunStage,
)
from .run_mode import (
    BENCHMARK_RUN_OFFICIAL_MODES as _OFFICIAL_MODES,
)
from .run_mode import (
    normalize_benchmark_run_mode,
)

# ---------------------------------------------------------------------------
# Constants and normalizer routing
# ---------------------------------------------------------------------------

JsonValue = Any
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST_PATH = BENCHMARK_ZOO_DIR
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"

SPECIALIZED_RESULT_NORMALIZER_SCRIPTS: dict[str, tuple[str, str]] = specialized_result_normalizer_scripts()

SPECIALIZED_ARTIFACT_OFFICIAL_RUN_BENCHMARKS: frozenset[str] = frozenset(
    {
        "aigcbench",
        "evalcrafter",
        "ipv-bench",
        "memobench",
        "phyfps-bench-gen",
        "phyeduvideo",
        "physics-iq",
        "physics-iq-verified",
        "phyground",
        "visual-chronometer",
        "world-in-world",
        "wrbench",
    }
)

IN_TREE_RUNTIME_KINDS: frozenset[str] = frozenset(
    {
        "in_tree_artifact_evaluator",
        "in_tree_artifact_metric_importer",
        "in_tree_judge_runtime",
        "in_tree_metric_aggregator",
        "in_tree_official_judge_runtime",
        "in_tree_official_runner",
        "in_tree_official_runtime",
        "in_tree_official_source",
        "in_tree_result_importer",
        "native_closed_loop_simulator",
    }
)

EMBODIED_RESULT_NORMALIZER_TRACKS: dict[str, str] = {
    "behavior1k": "vla",
    "bridgedata-v2": "vla",
    "calvin": "vla",
    "kinetix": "vla",
    "libero": "vla",
    "libero-mem": "vla",
    "libero-para": "vla",
    "libero-plus": "vla",
    "libero-pro": "vla",
    "maniskill": "vla",
    "maniskill2": "vla",
    "metaworld": "vla",
    "mikasa": "vla",
    "molmospaces": "vla",
    "rlbench": "vla",
    "robocasa": "vla",
    "robocerebra": "vla",
    "robomme": "vla",
    "robotwin": "vla",
    "simpler-env": "vla",
    "vlabench": "vla",
}

GENERIC_RESULT_NORMALIZER_BENCHMARKS: frozenset[str] = frozenset(
    {
        "devil-dynamics",
        "t2v-safety-bench",
        "t2vworldbench",
        "videoscience-bench",
        "worldarena",
    }
)


def _requires_external_runtime(*, default: bool, runtime_spec: Mapping[str, Any] | None) -> bool:
    if not isinstance(runtime_spec, Mapping):
        return default
    kind = str(runtime_spec.get("kind") or "").strip()
    if kind in IN_TREE_RUNTIME_KINDS:
        return False
    return default


class BenchmarkExecutionError(ValueError):
    """Base error for benchmark runner failures."""


class BenchmarkExecutionUnavailableError(BenchmarkExecutionError):
    """Raised when a manifest entry has no contract runner surface."""


# ---------------------------------------------------------------------------
# Command, env, and runtime-spec helpers
# ---------------------------------------------------------------------------


def _artifact_path(output_dir: Path, name: str) -> str:
    """Return an absolute path string under ``output_dir``."""
    return str((output_dir / name).resolve())


def _ensure_repo_on_pythonpath(env: dict[str, str]) -> None:
    """Make checked-in runner subprocesses importable from a source checkout."""
    repo_root = str(REPO_ROOT)
    current = env.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if repo_root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([repo_root, *parts])


def _command_kind_for_mode(mode: str) -> str:
    """Map ``mode`` to ``validation`` or ``run`` command kind."""
    return "validation" if mode in {"official-validation", "normalizer"} else "run"


def _command_for_mode(entry: BenchmarkZooEntry, mode: str) -> str | tuple[str, ...] | None:
    """Pick the manifest shell command for ``mode``."""
    if mode in {"official-validation", "normalizer"}:
        return entry.validation_command or entry.run_command
    if mode == "official-run":
        return entry.run_command
    return None


def _workspace_official_run_command(
    benchmark_id: str,
    *,
    output_dir: Path,
    generated_artifact_dir: JsonValue,
    kwargs: Mapping[str, JsonValue],
) -> tuple[str, ...] | None:
    """Build an official command from the canonical in-tree runner registry.

    Catalog commands remain useful documentation, but the executable argument
    contract lives beside the runner implementations.  Using that registry for
    ``official-run`` also covers integrated benchmarks whose catalog entry does
    not duplicate a shell command.
    """
    from ..runners.workspace_registry import (
        CLI_RUNNERS,
        benchmark_key,
        build_workspace_benchmark_command,
    )

    key = benchmark_key(benchmark_id)
    if key not in CLI_RUNNERS:
        return None
    params = dict(kwargs)
    params["run_official"] = True
    params.setdefault("python", sys.executable)
    if generated_artifact_dir is not None:
        params["generated_artifact_dir"] = str(generated_artifact_dir)
    payload = {
        "benchmark_id": key,
        "dataset_root": kwargs.get("dataset_root"),
        "dataset_manifest": kwargs.get("dataset_manifest"),
        "metrics": kwargs.get("metrics", ()),
        "results_path": kwargs.get("official_results_path"),
        "params": params,
    }
    return tuple(build_workspace_benchmark_command(payload, output_dir))


def _resolve_python_command(command: str | tuple[str, ...] | list[str]) -> str | tuple[str, ...]:
    """Rewrite leading ``python`` to ``sys.executable``."""
    if isinstance(command, (tuple, list)) and command and command[0] in {"python", "python3"}:
        return (sys.executable, *command[1:])
    return command


def _command_to_json(command: str | tuple[str, ...] | None) -> JsonValue:
    """Coerce a command to JSON-safe string or list."""
    if command is None:
        return None
    return command if isinstance(command, str) else list(command)


def _subprocess_command(command: str | tuple[str, ...]) -> str | list[str]:
    """Format a command for ``subprocess.run``."""
    return command if isinstance(command, str) else list(command)


def _timeout_seconds(value: JsonValue) -> float | None:
    """Parse timeout kwargs into seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _subprocess_output(value: str | bytes | None) -> str:
    """Decode captured subprocess stdout/stderr bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _env_mapping(value: JsonValue) -> dict[str, str]:
    """Coerce a mapping spec to ``dict[str, str]``."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _optional_text(value: JsonValue) -> str | None:
    """Coerce optional scalar values to strings."""
    if value in (None, ""):
        return None
    return str(value)


def _benchmark_env_prefix(benchmark_id: str) -> str:
    """Build ``WORLDFOUNDRY_<BENCHMARK>_`` env prefix from ``benchmark_id``."""
    normalized = "".join(char if char.isalnum() else "_" for char in benchmark_id).upper()
    return f"WORLDFOUNDRY_{normalized}"


def _benchmark_data_root(
    benchmark_id: str,
    kwargs: Mapping[str, JsonValue],
    env_overrides: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve benchmark dataset root from kwargs or env overrides."""
    prefix = _benchmark_env_prefix(benchmark_id)
    env_overrides = env_overrides or {}
    for key in ("benchmark_data_root", "data_root", "dataset_root"):
        value = kwargs.get(key)
        if value not in (None, ""):
            return Path(str(value))
    for env_name in (f"{prefix}_DATA_ROOT", "WORLDFOUNDRY_BENCHMARK_DATA_ROOT"):
        value = env_overrides.get(env_name) or os.environ.get(env_name)
        if value:
            return Path(value)
    return None


def _default_results_path_from_data_root(benchmark_id: str, data_root: Path | None) -> Path | None:
    """Probe ``data_root`` for conventional results/scores filenames."""
    if data_root is None:
        return None
    normalized = benchmark_id.replace("-", "_")
    candidates = [
        data_root / "results.json",
        data_root / "results.jsonl",
        data_root / "scores.json",
        data_root / "scores.jsonl",
        data_root / "annotations.json",
        data_root / "annotations.jsonl",
        data_root / normalized / "results.json",
        data_root / benchmark_id / "results.json",
    ]
    if benchmark_id == "phyground":
        candidates = [
            data_root / "annotations",
            data_root / "scores.json",
            data_root / "results.json",
            data_root / "phyground" / "annotations",
            *candidates,
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _commands_to_json(commands: Iterable[JsonValue]) -> list[JsonValue]:
    """Serialize install-command tuples for runtime reports."""
    result: list[JsonValue] = []
    for command in commands:
        if command is None:
            continue
        if isinstance(command, str):
            result.append(command)
        elif isinstance(command, (tuple, list)):
            result.append(list(command))
        else:
            result.append(command)
    return result


def _runtime_path(value: JsonValue) -> Path:
    """Resolve a runtime path with WorldFoundry path-token support."""
    path = resolve_worldfoundry_path(str(value))
    if not path.is_absolute():
        return REPO_ROOT / path
    return path


def _runtime_root_env(
    runtime: Mapping[str, JsonValue],
    *,
    clone_root: JsonValue = None,
    external_repo_runtime: bool = False,
) -> dict[str, str]:
    """Resolve a runner root into its configured ``root_env`` mapping."""
    root_env = runtime.get("root_env")
    if not root_env:
        return {}
    env_root = os.environ.get(str(root_env))
    if external_repo_runtime:
        explicit_root = clone_root or env_root or runtime.get("clone_dir") or runtime.get("root") or runtime.get("default_root")
    else:
        explicit_root = clone_root or runtime.get("clone_dir") or runtime.get("root") or runtime.get("default_root") or env_root
    if explicit_root:
        root = _runtime_path(explicit_root)
    elif external_repo_runtime:
        cache_root = Path(os.environ.get("WORLDFOUNDRY_EXTERNAL_REPO_CACHE") or benchmark_repo_cache_root())
        if not cache_root.is_absolute():
            cache_root = REPO_ROOT / cache_root
        root = cache_root / str(runtime.get("default_cache_subdir") or "official_repo")
    else:
        root = REPO_ROOT
    return {str(root_env): str(root)}


def _runner_runtime_spec(
    entry: BenchmarkZooEntry,
    *,
    benchmark_id: str,
    kwargs: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Compile runner runtime spec (repo, results path, env) from manifest + kwargs."""
    kwargs = kwargs or {}
    runtime = entry.runner_runtime
    runtime_kind = _optional_text(runtime.get("kind") if isinstance(runtime, Mapping) else None)
    prefix = _benchmark_env_prefix(benchmark_id)
    explicit_clone_root = kwargs.get("clone_root", kwargs.get("clone_dir"))
    clone_root = explicit_clone_root or (
        getattr(entry.runner, "clone_dir", None) if runtime_kind == "external_official_repo" else None
    )
    root_env_mapping = (
        _runtime_root_env(runtime, clone_root=clone_root, external_repo_runtime=runtime_kind == "external_official_repo")
        if isinstance(runtime, Mapping)
        else {}
    )
    root_dir = next(iter(root_env_mapping.values()), None)
    clone_dir = root_dir if runtime_kind == "external_official_repo" else None
    results_path_env = _optional_text(runtime.get("results_path_env") if isinstance(runtime, Mapping) else None)
    env_overrides = _env_mapping(kwargs.get("env_overrides", kwargs.get("env")))
    results_path = _optional_text(kwargs.get("results_path"))
    results_path_source = "results_path" if results_path else None
    if not results_path:
        results_path = _optional_text(kwargs.get("official_results_path"))
        results_path_source = "official_results_path" if results_path else None
    if not results_path:
        results_path = _optional_text(kwargs.get("upstream_results_path"))
        results_path_source = "upstream_results_path" if results_path else None
    if not results_path and results_path_env:
        results_path = _optional_text(env_overrides.get(results_path_env))
        results_path_source = f"env_override:{results_path_env}" if results_path else None
    if not results_path and results_path_env:
        results_path = _optional_text(os.environ.get(results_path_env))
        results_path_source = f"env:{results_path_env}" if results_path else None
    if not results_path:
        data_root = _benchmark_data_root(benchmark_id, kwargs, env_overrides)
        default_results_path = _default_results_path_from_data_root(benchmark_id, data_root)
        if default_results_path is not None:
            results_path = str(default_results_path)
            results_path_source = "benchmark_data_root"

    env = {}
    if hasattr(entry, "runner"):
        env.update(_env_mapping(entry.runner.env))
    env.update(_env_mapping(runtime.get("env") if isinstance(runtime, Mapping) else None))

    return {
        "kind": runtime_kind,
        "repo_url": (
            _optional_text(kwargs.get("repo_url"))
            or _optional_text(os.environ.get(f"{prefix}_REPO_URL"))
            or _optional_text(getattr(entry.runner, "repo_url", None))
            or _optional_text(runtime.get("repo_url") if isinstance(runtime, Mapping) else None)
            or _optional_text(runtime.get("official_repo_url") if isinstance(runtime, Mapping) else None)
        ),
        "repo_revision": (
            _optional_text(kwargs.get("revision", kwargs.get("repo_revision")))
            or _optional_text(os.environ.get(f"{prefix}_REVISION"))
            or _optional_text(getattr(entry.runner, "repo_revision", None))
            or _optional_text(runtime.get("repo_revision") if isinstance(runtime, Mapping) else None)
            or _optional_text(runtime.get("revision") if isinstance(runtime, Mapping) else None)
        ),
        "root_dir": root_dir,
        "clone_dir": clone_dir,
        "root_env": _optional_text(runtime.get("root_env") if isinstance(runtime, Mapping) else None),
        "results_path_env": results_path_env,
        "results_path": results_path,
        "results_path_source": results_path_source,
        "generated_artifact_dir_env": _optional_text(
            runtime.get("generated_artifact_dir_env") if isinstance(runtime, Mapping) else None
        ),
        "env": env,
        "install_commands": _commands_to_json(getattr(entry.runner, "install_commands", ())),
    }


def _runtime_spec_env(
    runtime_spec: Mapping[str, JsonValue],
    *,
    benchmark_id: str,
    generated_artifact_dir: JsonValue = None,
) -> dict[str, str]:
    """Translate runtime spec fields into subprocess env vars."""
    env = _env_mapping(runtime_spec.get("env"))
    prefix = _benchmark_env_prefix(benchmark_id)
    root_env = runtime_spec.get("root_env")
    clone_dir = runtime_spec.get("clone_dir")
    root_dir = runtime_spec.get("root_dir") or clone_dir
    if root_env and root_dir:
        env[str(root_env)] = str(root_dir)
    if runtime_spec.get("repo_url"):
        env["WORLDFOUNDRY_OFFICIAL_REPO_URL"] = str(runtime_spec["repo_url"])
        env[f"{prefix}_REPO_URL"] = str(runtime_spec["repo_url"])
    if runtime_spec.get("repo_revision"):
        env["WORLDFOUNDRY_OFFICIAL_REPO_REVISION"] = str(runtime_spec["repo_revision"])
        env[f"{prefix}_REVISION"] = str(runtime_spec["repo_revision"])
    if clone_dir:
        env["WORLDFOUNDRY_OFFICIAL_REPO_ROOT"] = str(clone_dir)
    result_path_env = runtime_spec.get("results_path_env")
    if result_path_env and runtime_spec.get("results_path"):
        env[str(result_path_env)] = str(runtime_spec["results_path"])
    generated_artifact_dir_env = runtime_spec.get("generated_artifact_dir_env")
    if generated_artifact_dir_env and generated_artifact_dir:
        env[str(generated_artifact_dir_env)] = str(generated_artifact_dir)
    return env


def _apply_benchmark_data_root_env(env: dict[str, str], benchmark_id: str, kwargs: Mapping[str, JsonValue]) -> None:
    """Bind ``WORLDFOUNDRY_*_DATA_ROOT`` env vars from kwargs."""
    benchmark_data_root = kwargs.get("benchmark_data_root", kwargs.get("data_root", kwargs.get("dataset_root")))
    if benchmark_data_root in (None, ""):
        return
    env["WORLDFOUNDRY_BENCHMARK_DATA_ROOT"] = str(benchmark_data_root)
    env[f"{_benchmark_env_prefix(benchmark_id)}_DATA_ROOT"] = str(benchmark_data_root)


def _expected_artifact_checks(entry: BenchmarkZooEntry, output_dir: Path) -> list[dict[str, JsonValue]]:
    """Check manifest ``expected_artifacts`` against ``output_dir``."""
    checks: list[dict[str, JsonValue]] = []
    for item in entry.expected_artifacts:
        if isinstance(item, str):
            relative_path = item
            metadata: Mapping[str, JsonValue] = {}
        elif isinstance(item, Mapping):
            raw_path = item.get("path") or item.get("uri")
            if not raw_path:
                continue
            relative_path = str(raw_path)
            metadata = item
        else:
            continue
        path = output_dir / relative_path
        checks.append(
            {
                "path": relative_path,
                "resolved_path": str(path.resolve()),
                "exists": path.exists(),
                "ok": path.exists(),
                "metadata": dict(metadata),
            }
        )
    return checks


def _write_runtime_report(root: Path, payload: Mapping[str, JsonValue]) -> Path:
    """Write ``runner_runtime_report.json`` under ``root``."""
    report_path = root / "runner_runtime_report.json"
    write_json(report_path, payload)
    return report_path


def _scorecard_normalization_available(scorecard: Mapping[str, JsonValue]) -> bool:
    """Return True when ``scorecard`` exposes at least one normalized metric."""
    if scorecard.get("normalization_ok") is True:
        return True
    evaluation = scorecard.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("available") is True:
        return True
    metrics = scorecard.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    leaderboard = metrics.get("leaderboard")
    if isinstance(leaderboard, Mapping) and bool(leaderboard):
        return True
    per_metric = metrics.get("per_metric")
    if isinstance(per_metric, Mapping):
        return any(isinstance(item, Mapping) and item.get("available") is True for item in per_metric.values())
    return False


# ---------------------------------------------------------------------------
# Official-result normalizers
# ---------------------------------------------------------------------------


def _run_specialized_result_normalizer(
    *,
    benchmark_id: str,
    output_dir: Path,
    results_path: Path | None,
    runtime_spec: Mapping[str, JsonValue],
    generated_artifact_dir: JsonValue = None,
    run_official: bool = False,
    kwargs: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue] | None:
    """Spawn a per-benchmark specialized normalizer subprocess."""
    spec = SPECIALIZED_RESULT_NORMALIZER_SCRIPTS.get(benchmark_id)
    if spec is None:
        return None
    script, result_flag = spec
    script_path = REPO_ROOT / script
    if not script_path.is_file():
        return None
    kwargs = kwargs or {}
    stdout_path = output_dir / "specialized_normalizer_stdout.log"
    stderr_path = output_dir / "specialized_normalizer_stderr.log"
    command = [
        sys.executable,
        str(script_path),
        "--benchmark-id",
        benchmark_id,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if run_official:
        command.append("--run-official")
    if results_path is not None:
        command.extend([result_flag, str(results_path)])
    score_dir = kwargs.get("score_dir") if benchmark_id == "camerabench" else None
    if score_dir not in (None, ""):
        command.extend(["--score-dir", str(score_dir), "--task", "all", "--no-gpt"])
    if benchmark_id in {
        "aigcbench",
        "phyfps-bench-gen",
        "visual-chronometer",
        "physics-iq",
        "physics-iq-verified",
        "physvidbench",
        "phygenbench",
        "videophy",
        "videophy2",
        "phyground",
        "phyeduvideo",
        "world-in-world",
        "mirabench",
        "memobench",
        "ewmbench",
        "evalcrafter",
        "wrbench",
    } and generated_artifact_dir is not None:
        command.extend(["--generated-artifact-dir", str(generated_artifact_dir)])
    if benchmark_id == "phyfps-bench-gen" and kwargs.get("prompt_manifest"):
        command.extend(["--prompt-manifest", str(kwargs["prompt_manifest"])])
    env = os.environ.copy()
    env.update(
        {
            "WORLDFOUNDRY_BENCHMARK_ID": benchmark_id,
            "WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR": str(output_dir),
        }
    )
    _ensure_repo_on_pythonpath(env)
    env.update(
        _runtime_spec_env(
            runtime_spec,
            benchmark_id=benchmark_id,
            generated_artifact_dir=generated_artifact_dir,
        )
    )
    _apply_benchmark_data_root_env(env, benchmark_id, kwargs)
    env.update(_env_mapping(kwargs.get("env_overrides", kwargs.get("env"))))
    timeout = kwargs.get("timeout_seconds", kwargs.get("timeout"))
    timeout_seconds = float(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) else None
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    scorecard_path = output_dir / "scorecard.json"
    if not scorecard_path.is_file():
        return {
            "ok": False,
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": duration_seconds,
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "scorecard": None,
            "error": "specialized normalizer did not write scorecard.json",
        }
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if not isinstance(scorecard, dict):
        return {
            "ok": False,
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": duration_seconds,
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "scorecard": None,
            "error": "specialized normalizer wrote a non-object scorecard.json",
        }
    normalization_ok = scorecard.get("normalization_ok") is True or _scorecard_normalization_available(scorecard)
    scorecard["normalization_ok"] = normalization_ok
    scorecard.setdefault("normalizer_only", scorecard.get("integration_evidence") is not True)
    scorecard["runtime"] = dict(runtime_spec)
    run = scorecard.get("run")
    if isinstance(run, Mapping):
        scorecard["run"] = {
            **dict(run),
            "runner": run.get("runner") or "benchmark_zoo_specialized_result_normalizer",
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": duration_seconds,
        }
    artifacts = dict(scorecard.get("artifacts") or {})
    artifacts["stdout"] = str(stdout_path.resolve())
    artifacts["stderr"] = str(stderr_path.resolve())
    scorecard["artifacts"] = artifacts
    write_json(scorecard_path, scorecard)
    return {
        "ok": normalization_ok,
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "scorecard": scorecard,
        "error": None if normalization_ok else "specialized normalizer produced no available metrics",
    }


def _run_embodied_result_normalizer(
    *,
    benchmark_id: str,
    output_dir: Path,
    results_path: Path,
    runtime_spec: Mapping[str, JsonValue],
    kwargs: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue] | None:
    """Normalize embodied/VLA results in-process via ``embodied.normalizer``."""
    track = EMBODIED_RESULT_NORMALIZER_TRACKS.get(benchmark_id)
    if track is None:
        return None
    kwargs = kwargs or {}
    normalizer_overrides = kwargs.get("normalizers")
    normalizers = dict(normalizer_overrides) if isinstance(normalizer_overrides, Mapping) else {}
    start = time.monotonic()
    try:
        from worldfoundry.evaluation.tasks.embodied.normalizer import normalize_results

        scorecard = normalize_results(
            input_paths=[results_path],
            output_dir=output_dir,
            benchmark_id=benchmark_id,
            track=track,
            normalizers={str(key): str(value) for key, value in normalizers.items()},
        )
        returncode = 0
        error = None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "worldfoundry.evaluation.tasks.embodied.normalizer",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {
                "benchmark_id": benchmark_id,
                "track": track,
                "contract_only": False,
            },
            "evaluation": {
                "available": False,
                "kind": "vla_va_wam_official_result_normalizer",
                "num_results": 0,
            },
            "metrics": {
                "leaderboard": {},
                "per_metric": {},
                "summary": {"normalized_result_rows": 0},
            },
            "artifacts": {"scorecard": _artifact_path(output_dir, "scorecard.json")},
        }
        write_json(output_dir / "scorecard.json", scorecard)
        returncode = 1
        error = str(exc)
    duration_seconds = time.monotonic() - start
    scorecard = dict(scorecard)
    normalization_ok = _scorecard_normalization_available(scorecard)
    scorecard["normalization_ok"] = normalization_ok
    scorecard["normalizer_only"] = True
    scorecard["runtime"] = dict(runtime_spec)
    run = scorecard.get("run")
    if isinstance(run, Mapping):
        scorecard["run"] = {
            **dict(run),
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        }
    write_json(output_dir / "scorecard.json", scorecard)
    return {
        "ok": normalization_ok,
        "command": [
            "worldfoundry-eval",
            "normalize",
            "embodied",
            "--benchmark-id",
            benchmark_id,
            "--track",
            track,
            "--input",
            str(results_path),
            "--output-dir",
            str(output_dir),
        ],
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "stdout_path": None,
        "stderr_path": None,
        "scorecard": scorecard,
        "error": error if error else (None if normalization_ok else "embodied normalizer produced no available metrics"),
    }


def _metric_result_value(result: Mapping[str, JsonValue]) -> JsonValue:
    """Prefer ``normalized_value``, else ``raw_value``."""
    value = result.get("normalized_value")
    return result.get("raw_value") if value is None else value


def _metric_scorecard_entry(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Convert one metric result row into scorecard ``per_metric`` shape."""
    value = _metric_result_value(result)
    valid = result.get("valid") is True and value is not None
    if valid:
        return {
            "available": True,
            "raw_score": result.get("raw_value"),
            "normalized_score": result.get("normalized_value"),
            "coverage": result.get("coverage", 1.0),
        }
    return {
        "available": False,
        "reason": result.get("skip_reason") or "metric_not_available",
        "diagnostics": result.get("diagnostics", {}),
    }


def _metric_raw_row(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Expose raw metric row with ``available`` flag."""
    row = dict(result)
    row["available"] = row.get("valid") is True and _metric_result_value(row) is not None
    if not row["available"]:
        row["reason"] = row.get("skip_reason") or "metric_not_available"
    return row


def _dataset_commands(entry: BenchmarkZooEntry) -> tuple[tuple[str, ...], ...]:
    """Build ``hf download`` commands from manifest dataset refs."""
    commands: list[tuple[str, ...]] = []
    for ref in entry.dataset_refs or (entry.dataset,):
        if not ref.hf_dataset_id:
            continue
        command = ["hf", "download", ref.hf_dataset_id, "--repo-type", "dataset"]
        if ref.revision:
            command.extend(("--revision", ref.revision))
        commands.append(tuple(command))
    return tuple(commands)


def _expected_paths(entry: BenchmarkZooEntry) -> tuple[str, ...]:
    """Collect expected dataset and artifact paths from the manifest."""
    paths: list[str] = []
    for ref in entry.dataset_refs or (entry.dataset,):
        if ref.path and ref.path not in paths:
            paths.append(ref.path)
    for artifact in entry.expected_artifacts:
        if isinstance(artifact, str):
            paths.append(artifact)
        elif isinstance(artifact, Mapping):
            path = artifact.get("path") or artifact.get("uri")
            if path:
                paths.append(str(path))
    return tuple(dict.fromkeys(paths))


# ---------------------------------------------------------------------------
# ManifestBenchmarkRunner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestBenchmarkRunner:
    """Per-benchmark orchestrator backed by one benchmark-zoo manifest entry.

    Execution flow:

    * ``prepare`` — resolve ``mode``, runtime spec, and output workspace.
    * ``run`` — subprocess official command, import existing results, or contract fixture.
    * ``collect`` — harvest generated files and artifact checks.
    * ``normalize`` — specialized, embodied, generic, or contract scorecard paths.
    * ``evaluate`` — full ``prepare → run → collect → normalize`` chain.
    """

    entry: BenchmarkZooEntry
    manifest_path: Path | None = None

    @property
    def benchmark_id(self) -> str:
        """Canonical benchmark identifier."""
        return self.entry.benchmark_id

    def load_manifest(self) -> Mapping[str, JsonValue]:
        """Serialize manifest entry (plus ``manifest_path`` when set)."""
        payload = self.entry.to_dict()
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
        return payload

    def materialization_plan(self) -> DatasetMaterializationPlan:
        """Build dataset download plan from manifest refs."""
        notes = list(self.entry.notes)
        if self.entry.dataset.not_applicable and self.entry.dataset.reason:
            notes.append(f"dataset not applicable: {self.entry.dataset.reason}")
        return DatasetMaterializationPlan(
            benchmark_id=self.benchmark_id,
            dataset_ids=self.entry.hf_dataset_ids,
            commands=_dataset_commands(self.entry),
            expected_paths=_expected_paths(self.entry),
            requires_auth=self.entry.requires_auth,
            notes=tuple(notes),
        )

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        """Yield a contract placeholder sample for integration tracking."""
        contract = get_external_benchmark_contract(self.benchmark_id)
        yield BenchmarkSample(
            sample_id=f"{self.benchmark_id}:manifest",
            inputs={key: None for key in contract.input_keys},
            expected_outputs={key: None for key in contract.output_keys},
            metadata={
                "benchmark_id": self.benchmark_id,
                "integration_status": self.entry.integration_status,
                "verification_status": self.entry.verification_status,
                "contract_only": self.entry.integration_status != "integrated",
            },
        )

    def report_metadata(self) -> Mapping[str, JsonValue]:
        """Runner capability metadata for scorecards and audits."""
        contract = get_external_benchmark_contract(self.benchmark_id)
        metadata: dict[str, JsonValue] = {
            "benchmark_id": self.benchmark_id,
            "runner": "benchmark_zoo_manifest_runner",
            "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
            "manifest_integration_status": self.entry.integration_status,
            "manifest_verification_status": self.entry.verification_status,
            "requires_upstream_runtime": _requires_external_runtime(
                default=contract.requires_upstream_runtime,
                runtime_spec=self.entry.runner_runtime,
            ),
        }
        if self.entry.runner_target:
            metadata["runner_target"] = self.entry.runner_target
        if self.entry.runner_runtime:
            metadata["runner_runtime"] = dict(self.entry.runner_runtime)
        if self.entry.install_profile:
            metadata["install_profile"] = self.entry.install_profile
        if self.manifest_path is not None:
            metadata["manifest_path"] = str(self.manifest_path)
        return metadata

    def prepare(
        self,
        *,
        output_dir: str | Path,
        mode: str = "contract",
        generated_artifact_dir: str | Path | None = None,
        **kwargs: JsonValue,
    ) -> OfficialRunStage:
        """Resolve runtime spec and create output workspace for ``mode``."""
        resolved_mode = normalize_benchmark_run_mode(mode)

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        plan = self.materialization_plan()
        command = _command_for_mode(self.entry, resolved_mode)
        command_kind = _command_kind_for_mode(resolved_mode) if resolved_mode in _OFFICIAL_MODES else None
        runtime_spec = _runner_runtime_spec(self.entry, benchmark_id=self.benchmark_id, kwargs=kwargs)
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="prepare",
            output_dir=root,
            status="ready",
            data={
                "mode": resolved_mode,
                "contract_only": resolved_mode == "contract",
                "command_kind": command_kind,
                "command": _command_to_json(command),
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "kwargs": dict(kwargs),
                "runtime": runtime_spec,
            },
            metadata={
                **self.report_metadata(),
                "materialization_plan": plan.to_dict(),
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
            },
        )

    def run(self, prepared: OfficialRunStage) -> OfficialRunStage:
        """Run official subprocess, import results, or emit contract fixture."""
        data = dict(prepared.data)
        mode = str(data.get("mode", "contract"))
        if mode in _OFFICIAL_MODES:
            return self._run_official(prepared)
        if mode != "contract":
            raise ValueError(f"unsupported benchmark run mode: {mode}")

        run_status = "contract_fixture"
        data.update(
            {
                "run_status": run_status,
                "evidence_level": "contract_fixture_only",
                "official_benchmark_verified": False,
                "integration_evidence": False,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=run_status,
            artifacts=prepared.artifacts,
            data=data,
            metadata=prepared.metadata,
        )

    def _failed_official_run_result(
        self,
        prepared: OfficialRunStage,
        *,
        status: str,
        error: str,
        data: dict[str, JsonValue],
        metadata: Mapping[str, JsonValue],
    ) -> OfficialRunStage:
        """Build a failed ``run`` stage with ``error`` populated."""
        data.update(
            {
                "run_status": status,
                "official_benchmark_verified": False,
                "integration_evidence": False,
                "error": error,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=status,
            artifacts=prepared.artifacts,
            data=data,
            metadata={**metadata, "run_status": status},
        )

    def _run_official(self, prepared: OfficialRunStage) -> OfficialRunStage:
        """Execute official subprocess and capture stdout/stderr logs."""
        data = dict(prepared.data)
        mode = str(data.get("mode", "official-validation"))
        command_kind = str(data.get("command_kind") or _command_kind_for_mode(mode))
        command = _command_for_mode(self.entry, mode)
        kwargs = data.get("kwargs")
        kwargs = kwargs if isinstance(kwargs, Mapping) else {}
        if mode == "official-run":
            workspace_command = _workspace_official_run_command(
                self.benchmark_id,
                output_dir=prepared.output_dir,
                generated_artifact_dir=data.get("generated_artifact_dir"),
                kwargs=kwargs,
            )
            if workspace_command is not None:
                command = workspace_command
        runtime_spec = data.get("runtime")
        runtime_spec = dict(runtime_spec) if isinstance(runtime_spec, Mapping) else _runner_runtime_spec(self.entry, benchmark_id=self.benchmark_id)
        data["runtime"] = runtime_spec
        metadata = {
            **prepared.metadata,
            "runtime": runtime_spec,
            "runner_runtime_spec": runtime_spec,
        }
        results_path = runtime_spec.get("results_path")
        if results_path and Path(str(results_path)).exists():
            data.update(
                {
                    "run_status": "official_results_import",
                    "returncode": 0,
                    "duration_seconds": 0.0,
                    "command": None,
                    "command_kind": command_kind,
                    "workdir": str(REPO_ROOT),
                    "stdout_path": None,
                    "stderr_path": None,
                    "expected_artifact_checks": [],
                    "scorecard_runtime_flags": {
                        "official_benchmark_verified": False,
                        "integration_evidence": False,
                    },
                    "official_benchmark_verified": False,
                    "integration_evidence": False,
                    "error": None,
                }
            )
            return OfficialRunStage(
                benchmark_id=self.benchmark_id,
                stage="run",
                output_dir=prepared.output_dir,
                status="official_results_import",
                artifacts=prepared.artifacts,
                data=data,
                metadata={**metadata, "run_status": "official_results_import"},
            )
        score_dir = kwargs.get("score_dir") or os.environ.get("WORLDFOUNDRY_CAMERABENCH_SCORE_DIR")
        if self.benchmark_id == "camerabench" and mode in {"official-validation", "normalizer"} and score_dir:
            score_dir_path = Path(str(score_dir))
            if score_dir_path.is_dir():
                data.update(
                    {
                        "run_status": "official_score_dir_import",
                        "returncode": 0,
                        "duration_seconds": 0.0,
                        "command": None,
                        "command_kind": command_kind,
                        "workdir": str(REPO_ROOT),
                        "stdout_path": None,
                        "stderr_path": None,
                        "expected_artifact_checks": [],
                        "scorecard_runtime_flags": {
                            "official_benchmark_verified": False,
                            "integration_evidence": False,
                        },
                        "official_benchmark_verified": False,
                        "integration_evidence": False,
                        "error": None,
                    }
                )
                return OfficialRunStage(
                    benchmark_id=self.benchmark_id,
                    stage="run",
                    output_dir=prepared.output_dir,
                    status="official_score_dir_import",
                    artifacts=prepared.artifacts,
                    data=data,
                    metadata={**metadata, "run_status": "official_score_dir_import"},
                )
        if results_path and runtime_spec.get("results_path_source") == "official_results_path":
            return self._failed_official_run_result(
                prepared,
                status="missing_official_results_path",
                error=f"official results path does not exist: {results_path}",
                data=data,
                metadata=metadata,
            )
        if mode == "normalizer":
            error = (
                f"official results path does not exist: {results_path}"
                if results_path
                else "--mode normalizer requires --official-results-path or a configured results path"
            )
            return self._failed_official_run_result(
                prepared,
                status="missing_official_results_path",
                error=error,
                data=data,
                metadata=metadata,
            )
        if runtime_spec.get("kind") == "external_official_repo" and not runtime_spec.get("repo_url"):
            return self._failed_official_run_result(
                prepared,
                status="missing_official_runtime_spec",
                error="missing repo_url for external_official_repo runtime",
                data=data,
                metadata=metadata,
            )
        if command is None:
            return self._failed_official_run_result(
                prepared,
                status="missing_official_command",
                error=f"missing {command_kind}_command",
                data=data,
                metadata=metadata,
            )

        workdir = Path(str(kwargs.get("workdir"))) if kwargs.get("workdir") else REPO_ROOT
        env = os.environ.copy()
        env.update(
            {
                "WORLDFOUNDRY_BENCHMARK_ID": self.benchmark_id,
                "WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR": str(prepared.output_dir),
                "WORLDFOUNDRY_BENCHMARK_COMMAND_KIND": command_kind,
            }
        )
        _ensure_repo_on_pythonpath(env)
        env.setdefault("WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable)
        generated_artifact_dir = data.get("generated_artifact_dir")
        if generated_artifact_dir is not None:
            env["WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"] = str(generated_artifact_dir)
        if kwargs.get("limit") not in (None, ""):
            env["WORLDFOUNDRY_BENCHMARK_LIMIT"] = str(kwargs["limit"])
        if kwargs.get("model_id") not in (None, ""):
            env["WORLDFOUNDRY_MODEL_ID"] = str(kwargs["model_id"])
        if kwargs.get("split") not in (None, ""):
            env["WORLDFOUNDRY_BENCHMARK_SPLIT"] = str(kwargs["split"])
        env.update(
            _runtime_spec_env(
                runtime_spec,
                benchmark_id=self.benchmark_id,
                generated_artifact_dir=generated_artifact_dir,
            )
        )
        _apply_benchmark_data_root_env(env, self.benchmark_id, kwargs)
        if kwargs.get("prompt_manifest"):
            env["WORLDFOUNDRY_PROMPT_MANIFEST"] = str(kwargs["prompt_manifest"])
            if self.benchmark_id == "phyfps-bench-gen":
                env["WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST"] = str(kwargs["prompt_manifest"])
        env.update(_env_mapping(kwargs.get("env_overrides", kwargs.get("env"))))

        resolved_command = _resolve_python_command(command)
        stdout_path = prepared.output_dir / f"{command_kind}_stdout.log"
        stderr_path = prepared.output_dir / f"{command_kind}_stderr.log"
        timeout_seconds = _timeout_seconds(kwargs.get("timeout_seconds", kwargs.get("timeout")))
        start = time.monotonic()
        try:
            completed = subprocess.run(
                _subprocess_command(resolved_command),
                cwd=workdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                shell=isinstance(resolved_command, str),
            )
            duration_seconds = time.monotonic() - start
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            status = "succeeded" if completed.returncode == 0 else "failed"
            if completed.returncode == 0:
                emitted_scorecard_path = prepared.output_dir / "scorecard.json"
                if emitted_scorecard_path.is_file():
                    try:
                        emitted_scorecard = json.loads(emitted_scorecard_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        emitted_scorecard = {}
                    emitted_run = emitted_scorecard.get("run") if isinstance(emitted_scorecard, Mapping) else {}
                    emitted_status = emitted_run.get("status") if isinstance(emitted_run, Mapping) else None
                    if emitted_status in {"official_results_imported", "official_results_normalized"}:
                        status = str(emitted_status)
            error = None if completed.returncode == 0 else f"official command exited with code {completed.returncode}"
            data.update(
                {
                    "run_status": status,
                    "returncode": completed.returncode,
                    "duration_seconds": duration_seconds,
                    "command": _command_to_json(resolved_command),
                    "command_kind": command_kind,
                    "workdir": str(workdir),
                    "runtime": runtime_spec,
                    "stdout_path": str(stdout_path.resolve()),
                    "stderr_path": str(stderr_path.resolve()),
                    "error": error,
                }
            )
            return OfficialRunStage(
                benchmark_id=self.benchmark_id,
                stage="run",
                output_dir=prepared.output_dir,
                status=status,
                artifacts=prepared.artifacts,
                data=data,
                metadata={**metadata, "run_status": status, "returncode": completed.returncode},
            )
        except subprocess.TimeoutExpired as exc:
            duration_seconds = time.monotonic() - start
            stdout_path.write_text(_subprocess_output(exc.stdout), encoding="utf-8")
            stderr_path.write_text(_subprocess_output(exc.stderr), encoding="utf-8")
            status = "timeout"
            error = f"official command timed out after {timeout_seconds} seconds"
        except OSError as exc:
            duration_seconds = time.monotonic() - start
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            status = "failed_to_launch"
            error = str(exc)

        data.update(
            {
                "run_status": status,
                "returncode": None,
                "duration_seconds": duration_seconds,
                "command": _command_to_json(resolved_command),
                "command_kind": command_kind,
                "workdir": str(workdir),
                "runtime": runtime_spec,
                "stdout_path": str(stdout_path.resolve()),
                "stderr_path": str(stderr_path.resolve()),
                "error": error,
            }
        )
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status=status,
            artifacts=prepared.artifacts,
            data=data,
            metadata={**metadata, "run_status": data.get("run_status"), "returncode": data.get("returncode")},
        )

    def collect(self, run_result: OfficialRunStage) -> OfficialRunStage:
        """Harvest output files and expected-artifact checks after ``run``."""
        data = dict(run_result.data)
        generated_files: list[str] = []
        generated_artifact_dir = data.get("generated_artifact_dir")
        if generated_artifact_dir is not None:
            artifact_root = Path(str(generated_artifact_dir))
            if artifact_root.exists():
                generated_files = [str(path) for path in sorted(artifact_root.rglob("*")) if path.is_file()]

        data["generated_files"] = generated_files
        data["generated_file_count"] = len(generated_files)
        data["output_files"] = [
            str(path)
            for path in sorted(run_result.output_dir.rglob("*"))
            if path.is_file()
        ]
        data["expected_artifact_checks"] = _expected_artifact_checks(self.entry, run_result.output_dir)
        data["scorecard_runtime_flags"] = inspect_scorecard_runtime_flags(run_result.output_dir / "scorecard.json")
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="collect",
            output_dir=run_result.output_dir,
            status="collected",
            artifacts=run_result.artifacts,
            data=data,
            metadata=run_result.metadata,
        )

    def normalize(self, collected: OfficialRunStage) -> OfficialRunResult:
        """Normalize collected outputs into ``scorecard.json`` (official or contract)."""
        data = dict(collected.data)
        mode = str(data.get("mode", "contract"))
        mode = normalize_benchmark_run_mode(mode)
        if mode in _OFFICIAL_MODES:
            return self._normalize_official(collected, mode=mode)

        root = collected.output_dir
        contract = get_external_benchmark_contract(self.benchmark_id)
        generated_files = [str(path) for path in data.get("generated_files") or ()]
        generated_artifact_dir = data.get("generated_artifact_dir")
        contract_only = mode == "contract"
        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}

        if contract_only and has_benchmark_contract_evaluator(self.benchmark_id):
            evaluator_kwargs = dict(extra_metadata)
            result = write_benchmark_contract_evaluation(
                benchmark_id=self.benchmark_id,
                display_name=self.entry.name or contract.display_name,
                official_metric_ids=contract.metric_ids,
                output_dir=root,
                generated_artifact_dir=None if generated_artifact_dir is None else str(generated_artifact_dir),
                manifest=self.load_manifest(),
                runner="benchmark_zoo_manifest_runner",
                mode=mode,
                **evaluator_kwargs,
            )
            return OfficialRunResult(
                benchmark_id=self.benchmark_id,
                output_dir=root,
                scorecard_path=root / "scorecard.json",
                raw_results_path=root / "raw_metric_table.jsonl",
                official_benchmark_verified=False,
                integration_evidence=False,
                artifacts=result["artifacts"],
                metadata={
                    "mode": mode,
                    "contract_only": contract_only,
                    "manifest_integration_status": self.entry.integration_status,
                    "manifest_verification_status": self.entry.verification_status,
                    "requires_upstream_runtime": contract.requires_upstream_runtime,
                    "evaluator": BENCHMARK_CONTRACT_EVALUATOR_KINDS[self.benchmark_id],
                    **extra_metadata,
                },
            )

        generated_artifact_manifest: JsonValue = {
            "generated_files": generated_files,
        }
        if generated_artifact_dir is not None:
            generated_artifact_manifest["generated_artifact_dir"] = str(generated_artifact_dir)
        task_metadata: dict[str, JsonValue] = {
            "benchmark_id": self.benchmark_id,
            "expected_output_keys": list(contract.output_keys),
            "metric_ids": list(contract.metric_ids),
        }
        if self.entry.expected_artifacts:
            task_metadata["required_artifacts"] = [
                item if isinstance(item, str) else item.get("path", item.get("uri"))
                for item in self.entry.expected_artifacts
                if isinstance(item, str) or isinstance(item, Mapping)
            ]

        metric_results = [
            evaluate_external_metric(
                self.benchmark_id,
                metric_id,
                generated_artifact_manifest=generated_artifact_manifest,
                task_metadata=task_metadata,
                reference={"contract": contract.to_dict(), "manifest": self.load_manifest()},
                sample_id=f"{self.benchmark_id}:contract",
                artifact_base_dir=generated_artifact_dir,
            ).to_dict()
            for metric_id in contract.metric_ids
        ]

        raw_rows = [_metric_raw_row(result) for result in metric_results]
        per_metric = {str(result["metric_id"]): _metric_scorecard_entry(result) for result in metric_results}
        blocked_count = sum(1 for result in metric_results if result.get("skip_reason") in {"judge_api_required", "judge_required", "external_runtime_required"})
        failed_count = sum(1 for result in metric_results if result.get("valid") is not True and result.get("skip_reason") not in {"judge_api_required", "judge_required", "external_runtime_required"})
        available_count = sum(1 for result in raw_rows if result.get("available") is True)

        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "run": {
                "status": "contract_fixture",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_manifest_runner",
                "mode": mode,
            },
            "benchmark": {
                "benchmark_id": self.benchmark_id,
                "name": self.entry.name or contract.display_name,
                "contract_only": contract_only,
                "evidence_level": "contract_fixture_only",
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
            },
            "dataset": {
                "hf_dataset_ids": list(self.entry.hf_dataset_ids),
                "requires_auth": self.entry.requires_auth,
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "generated_file_count": len(generated_files),
            },
            "metrics": {
                "per_metric": per_metric,
                "summary": {
                    "sample_count": len(generated_files),
                    "successful_samples": len(generated_files),
                    "failed_samples": 0,
                    "available_metric_count": available_count,
                    "blocked_metric_count": blocked_count,
                    "failed_metric_count": failed_count,
                },
            },
            "evaluation": {
                "available": available_count > 0,
                "kind": "external_metric_registry",
                "evidence_level": "contract_fixture_only",
                "skip_count": len(contract.metric_ids) - available_count,
                "blocked_count": blocked_count,
                "failed_count": failed_count,
            },
            "artifacts": {
                "scorecard": _artifact_path(root, "scorecard.json"),
                "benchmark_contract": _artifact_path(root, "benchmark_contract.json"),
                "raw_metric_table": _artifact_path(root, "raw_metric_table.jsonl"),
            },
        }
        benchmark_contract = contract.to_dict()
        benchmark_contract["manifest"] = self.load_manifest()
        benchmark_contract["generated_files"] = generated_files
        benchmark_contract["metric_evaluators"] = [
            entry.to_dict()
            for entry in list_external_metric_evaluators(self.benchmark_id)
            if entry.benchmark_id == self.benchmark_id
        ]

        write_json(root / "scorecard.json", scorecard)
        write_json(root / "benchmark_contract.json", benchmark_contract)
        (root / "raw_metric_table.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in raw_rows),
            encoding="utf-8",
        )

        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=root / "scorecard.json",
            raw_results_path=root / "raw_metric_table.jsonl",
            official_benchmark_verified=False,
            integration_evidence=False,
            artifacts=scorecard["artifacts"],
            metadata={
                "mode": mode,
                "contract_only": contract_only,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                **extra_metadata,
            },
        )

    def _normalize_generic_official_results(
        self,
        *,
        root: Path,
        contract: Any,
        data: Mapping[str, JsonValue],
        mode: str,
        results_path: Path,
        runtime_spec: Mapping[str, JsonValue],
        runtime_report_path: Path,
        generated_artifact_dir: JsonValue,
        generated_files: list[str],
    ) -> OfficialRunResult:
        """Normalize official results via :class:`OfficialResultsNormalizer`."""
        normalization = OfficialResultsNormalizer.from_benchmark_entry(self.entry).normalize_file(str(results_path))
        raw_metric_path = root / "raw_metric_table.jsonl"
        raw_metric_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalization.raw_metric_rows()),
            encoding="utf-8",
        )
        per_metric = normalization.scorecard_metrics()
        available_count = sum(1 for item in per_metric.values() if item.get("available") is True)
        blocked_count = len(per_metric) - available_count
        scorecard_path = root / "scorecard.json"
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": available_count > 0,
            "run": {
                "status": data.get("run_status", "normalizer_only"),
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_manifest_runner",
                "mode": mode,
                "command": data.get("command"),
                "returncode": data.get("returncode"),
                "error": data.get("error"),
            },
            "benchmark": {
                "benchmark_id": self.benchmark_id,
                "name": self.entry.name or contract.display_name,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
            },
            "dataset": {
                "hf_dataset_ids": list(self.entry.hf_dataset_ids),
                "requires_auth": self.entry.requires_auth,
                "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                "generated_file_count": len(generated_files),
                "official_results_path": str(results_path),
                "official_result_record_count": len(normalization.records),
            },
            "metrics": {
                "leaderboard": {
                    metric_id: entry["normalized_score"]
                    for metric_id, entry in per_metric.items()
                    if entry.get("available") is True
                },
                "per_metric": per_metric,
                "summary": {
                    "sample_count": len(normalization.records),
                    "successful_samples": len(normalization.records),
                    "failed_samples": 0,
                    "available_metric_count": available_count,
                    "blocked_metric_count": blocked_count,
                },
            },
            "evaluation": {
                "available": available_count > 0,
                "kind": "official_results_normalizer",
                "evidence_level": "normalizer_only",
                "num_results": len(normalization.metric_results),
                "skip_count": blocked_count,
                "blocked_count": blocked_count,
            },
            "runtime": dict(runtime_spec),
            "artifacts": {
                "scorecard": _artifact_path(root, "scorecard.json"),
                "runner_runtime_report": str(runtime_report_path.resolve()),
                "raw_metric_table": str(raw_metric_path.resolve()),
                "stdout": data.get("stdout_path"),
                "stderr": data.get("stderr_path"),
            },
        }
        write_json(scorecard_path, scorecard)
        artifacts = {
            "scorecard": str(scorecard_path.resolve()),
            "runner_runtime_report": str(runtime_report_path.resolve()),
            "raw_metric_table": str(raw_metric_path.resolve()),
        }
        for key, value in (("stdout", data.get("stdout_path")), ("stderr", data.get("stderr_path"))):
            if value:
                artifacts[key] = str(value)
        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=raw_metric_path,
            official_benchmark_verified=False,
            integration_evidence=False,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                "run_status": data.get("run_status"),
                "returncode": data.get("returncode"),
                "runtime": dict(runtime_spec),
                "runner_runtime_spec": dict(runtime_spec),
                **extra_metadata,
                "normalization_ok": available_count > 0,
            },
        )

    def _normalize_official(self, collected: OfficialRunStage, *, mode: str) -> OfficialRunResult:
        """Route official normalization across specialized, embodied, and generic paths."""
        data = dict(collected.data)
        root = collected.output_dir
        contract = get_external_benchmark_contract(self.benchmark_id)
        generated_files = [str(path) for path in data.get("generated_files") or ()]
        generated_artifact_dir = data.get("generated_artifact_dir")
        scorecard_path = root / "scorecard.json"
        scorecard_flags = data.get("scorecard_runtime_flags")
        scorecard_flags = scorecard_flags if isinstance(scorecard_flags, Mapping) else {}
        scorecard_runtime_flags = {
            "official_benchmark_verified": scorecard_flags.get("official_benchmark_verified") is True,
            "integration_evidence": scorecard_flags.get("integration_evidence") is True,
            **dict(scorecard_flags),
        }
        runtime_spec = data.get("runtime")
        runtime_spec = dict(runtime_spec) if isinstance(runtime_spec, Mapping) else _runner_runtime_spec(self.entry, benchmark_id=self.benchmark_id)
        expected_artifact_checks = [
            dict(item)
            for item in data.get("expected_artifact_checks") or ()
            if isinstance(item, Mapping)
        ]
        artifacts_ok = all(item.get("ok") is True for item in expected_artifact_checks)
        command_ok = data.get("returncode") == 0
        official_benchmark_verified = (
            command_ok
            and artifacts_ok
            and scorecard_flags.get("official_benchmark_verified") is True
        )
        integration_evidence = (
            command_ok
            and artifacts_ok
            and scorecard_flags.get("integration_evidence") is True
        )
        runtime_report = {
            "schema_version": "worldfoundry-official-benchmark-runtime-report",
            "benchmark_id": self.benchmark_id,
            "mode": mode,
            "command_kind": data.get("command_kind"),
            "command": data.get("command"),
            "returncode": data.get("returncode"),
            "run_status": data.get("run_status"),
            "duration_seconds": data.get("duration_seconds"),
            "workdir": data.get("workdir"),
            "runtime": runtime_spec,
            "stdout_path": data.get("stdout_path"),
            "stderr_path": data.get("stderr_path"),
            "error": data.get("error"),
            "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
            "generated_file_count": len(generated_files),
            "expected_artifact_checks": expected_artifact_checks,
            "scorecard_runtime_flags": scorecard_runtime_flags,
            "official_benchmark_verified": official_benchmark_verified,
            "integration_evidence": integration_evidence,
            "manifest": self.load_manifest(),
        }
        runtime_report_path = _write_runtime_report(root, runtime_report)

        results_path = runtime_spec.get("results_path")
        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        score_dir = extra_metadata.get("score_dir") if self.benchmark_id == "camerabench" else None
        has_score_dir = score_dir not in (None, "") and Path(str(score_dir)).is_dir()
        has_results_path = bool("results_path" in runtime_spec and runtime_spec["results_path"] and Path(str(runtime_spec["results_path"])).exists())
        if has_results_path or has_score_dir:
            specialized = _run_specialized_result_normalizer(
                benchmark_id=self.benchmark_id,
                output_dir=root,
                results_path=Path(str(results_path)) if has_results_path else None,
                runtime_spec=runtime_spec,
                generated_artifact_dir=generated_artifact_dir,
                run_official=(
                    mode == "official-run"
                    and generated_artifact_dir is not None
                    and self.benchmark_id in SPECIALIZED_ARTIFACT_OFFICIAL_RUN_BENCHMARKS
                ),
                kwargs=extra_metadata,
            )
            if specialized is not None:
                runtime_report["command"] = specialized["command"]
                runtime_report["returncode"] = specialized["returncode"]
                runtime_report["duration_seconds"] = specialized["duration_seconds"]
                runtime_report["stdout_path"] = specialized["stdout_path"]
                runtime_report["stderr_path"] = specialized["stderr_path"]
                runtime_report["run_status"] = (
                    "official_results_normalized" if specialized["ok"] else "official_results_missing_scores"
                )
                scorecard = specialized.get("scorecard")
                scorecard = dict(scorecard) if isinstance(scorecard, Mapping) else {}
                artifacts = dict(scorecard.get("artifacts") or {})
                artifacts["runner_runtime_report"] = str(runtime_report_path.resolve())
                scorecard["runtime"] = runtime_spec
                scorecard["artifacts"] = artifacts
                write_json(scorecard_path, scorecard)
                expected_artifact_checks = _expected_artifact_checks(self.entry, root)
                refreshed_flags = inspect_scorecard_runtime_flags(scorecard_path)
                scorecard_runtime_flags = {
                    "official_benchmark_verified": refreshed_flags.get("official_benchmark_verified") is True,
                    "integration_evidence": refreshed_flags.get("integration_evidence") is True,
                    **refreshed_flags,
                }
                artifacts_ok = all(item.get("ok") is True for item in expected_artifact_checks)
                command_ok = specialized["returncode"] == 0
                runtime_report.update(
                    {
                        "error": specialized.get("error"),
                        "expected_artifact_checks": expected_artifact_checks,
                        "scorecard_runtime_flags": scorecard_runtime_flags,
                        "official_benchmark_verified": (
                            command_ok
                            and artifacts_ok
                            and scorecard_runtime_flags.get("official_benchmark_verified") is True
                        ),
                        "integration_evidence": (
                            command_ok
                            and artifacts_ok
                            and scorecard_runtime_flags.get("integration_evidence") is True
                        ),
                    }
                )
                runtime_report_path = _write_runtime_report(root, runtime_report)
                raw_metric_path = root / "raw_metric_table.jsonl"
                return OfficialRunResult(
                    benchmark_id=self.benchmark_id,
                    output_dir=root,
                    scorecard_path=scorecard_path,
                    raw_results_path=raw_metric_path if raw_metric_path.is_file() else None,
                    official_benchmark_verified=scorecard.get("official_benchmark_verified") is True,
                    integration_evidence=scorecard.get("integration_evidence") is True,
                    artifacts=artifacts,
                    metadata={
                        "mode": mode,
                        "contract_only": False,
                        "normalizer_only": scorecard.get("normalizer_only") is not False,
                        "manifest_integration_status": self.entry.integration_status,
                        "manifest_verification_status": self.entry.verification_status,
                        "requires_upstream_runtime": _requires_external_runtime(
                            default=contract.requires_upstream_runtime,
                            runtime_spec=runtime_spec,
                        ),
                        "run_status": runtime_report["run_status"],
                        "returncode": specialized["returncode"],
                        "runtime": runtime_spec,
                        "runner_runtime_spec": runtime_spec,
                        "specialized_normalizer": True,
                        "specialized_normalizer_error": specialized.get("error"),
                        **extra_metadata,
                        "normalization_ok": specialized.get("ok") is True,
                    },
                )
            embodied = _run_embodied_result_normalizer(
                benchmark_id=self.benchmark_id,
                output_dir=root,
                results_path=Path(str(results_path)),
                runtime_spec=runtime_spec,
                kwargs=extra_metadata,
            )
            if embodied is not None:
                runtime_report["command"] = embodied["command"]
                runtime_report["returncode"] = embodied["returncode"]
                runtime_report["duration_seconds"] = embodied["duration_seconds"]
                runtime_report["stdout_path"] = embodied["stdout_path"]
                runtime_report["stderr_path"] = embodied["stderr_path"]
                runtime_report["run_status"] = (
                    "official_results_normalized" if embodied["ok"] else "official_results_missing_scores"
                )
                runtime_report_path = _write_runtime_report(root, runtime_report)
                scorecard = embodied.get("scorecard")
                scorecard = dict(scorecard) if isinstance(scorecard, Mapping) else {}
                artifacts = dict(scorecard.get("artifacts") or {})
                artifacts["runner_runtime_report"] = str(runtime_report_path.resolve())
                scorecard["runtime"] = runtime_spec
                scorecard["artifacts"] = artifacts
                write_json(scorecard_path, scorecard)
                raw_metric_path = root / "raw_metric_table.jsonl"
                raw_results_path = root / "raw_results.jsonl"
                return OfficialRunResult(
                    benchmark_id=self.benchmark_id,
                    output_dir=root,
                    scorecard_path=scorecard_path,
                    raw_results_path=(
                        raw_metric_path
                        if raw_metric_path.is_file()
                        else raw_results_path if raw_results_path.is_file() else None
                    ),
                    official_benchmark_verified=scorecard.get("official_benchmark_verified") is True,
                    integration_evidence=scorecard.get("integration_evidence") is True,
                    artifacts=artifacts,
                    metadata={
                        "mode": mode,
                        "contract_only": False,
                        "normalizer_only": True,
                        "manifest_integration_status": self.entry.integration_status,
                        "manifest_verification_status": self.entry.verification_status,
                        "requires_upstream_runtime": contract.requires_upstream_runtime,
                        "run_status": runtime_report["run_status"],
                        "returncode": embodied["returncode"],
                        "runtime": runtime_spec,
                        "runner_runtime_spec": runtime_spec,
                        "embodied_normalizer": True,
                        "embodied_normalizer_error": embodied.get("error"),
                        **extra_metadata,
                        "normalization_ok": embodied.get("ok") is True,
                    },
                )
            if self.benchmark_id in GENERIC_RESULT_NORMALIZER_BENCHMARKS:
                return self._normalize_generic_official_results(
                    root=root,
                    contract=contract,
                    data=data,
                    mode=mode,
                    results_path=Path(str(results_path)),
                    runtime_spec=runtime_spec,
                    runtime_report_path=runtime_report_path,
                    generated_artifact_dir=generated_artifact_dir,
                    generated_files=generated_files,
                )
            if has_benchmark_contract_evaluator(self.benchmark_id):
                evaluator_kwargs = dict(extra_metadata)
                evaluator_kwargs.pop("official_results_path", None)
                evaluator_kwargs.pop("judge_results_path", None)
                result = write_benchmark_contract_evaluation(
                    benchmark_id=self.benchmark_id,
                    display_name=self.entry.name or contract.display_name,
                    official_metric_ids=contract.metric_ids,
                    output_dir=root,
                    generated_artifact_dir=None if generated_artifact_dir is None else str(generated_artifact_dir),
                    manifest=self.load_manifest(),
                    runner="benchmark_zoo_manifest_runner",
                    mode=mode,
                    official_results_path=str(results_path),
                    **evaluator_kwargs,
                )
                scorecard_path = root / "scorecard.json"
                raw_metric_path = root / "raw_metric_table.jsonl"
                scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
                artifacts = dict(result.get("artifacts") or {})
                artifacts["runner_runtime_report"] = str(runtime_report_path.resolve())
                scorecard["runtime"] = runtime_spec
                scorecard["artifacts"] = {**dict(scorecard.get("artifacts") or {}), **artifacts}
                write_json(scorecard_path, scorecard)
                return OfficialRunResult(
                    benchmark_id=self.benchmark_id,
                    output_dir=root,
                    scorecard_path=scorecard_path,
                    raw_results_path=raw_metric_path,
                    official_benchmark_verified=False,
                    integration_evidence=False,
                    artifacts=artifacts,
                    metadata={
                        "mode": mode,
                        "contract_only": False,
                        "normalizer_only": True,
                        "manifest_integration_status": self.entry.integration_status,
                        "manifest_verification_status": self.entry.verification_status,
                        "requires_upstream_runtime": contract.requires_upstream_runtime,
                        "run_status": data.get("run_status"),
                        "returncode": data.get("returncode"),
                        "runtime": runtime_spec,
                        "runner_runtime_spec": runtime_spec,
                        **extra_metadata,
                        "normalization_ok": _scorecard_normalization_available(scorecard),
                    },
                )
            return self._normalize_generic_official_results(
                root=root,
                contract=contract,
                data=data,
                mode=mode,
                results_path=Path(str(results_path)),
                runtime_spec=runtime_spec,
                runtime_report_path=runtime_report_path,
                generated_artifact_dir=generated_artifact_dir,
                generated_files=generated_files,
            )

        if not scorecard_path.is_file():
            scorecard = {
                "schema_version": SCORECARD_SCHEMA_VERSION,
                "official_benchmark_verified": False,
                "integration_evidence": False,
                "run": {
                    "status": data.get("run_status", "failed"),
                    "started_at": utc_now_iso(),
                    "runner": "benchmark_zoo_manifest_runner",
                    "mode": mode,
                    "command": data.get("command"),
                    "returncode": data.get("returncode"),
                    "error": data.get("error") or "official runtime did not write scorecard.json",
                },
                "benchmark": {
                    "benchmark_id": self.benchmark_id,
                    "name": self.entry.name or contract.display_name,
                    "contract_only": False,
                    "manifest_integration_status": self.entry.integration_status,
                    "manifest_verification_status": self.entry.verification_status,
                    "requires_upstream_runtime": contract.requires_upstream_runtime,
                },
                "dataset": {
                    "hf_dataset_ids": list(self.entry.hf_dataset_ids),
                    "requires_auth": self.entry.requires_auth,
                    "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
                    "generated_file_count": len(generated_files),
                },
                "metrics": {
                    "per_metric": {
                        metric_id: {
                            "available": False,
                            "reason": "official_runtime_scorecard_missing",
                        }
                        for metric_id in contract.metric_ids
                    },
                    "summary": {
                        "sample_count": len(generated_files),
                        "successful_samples": 0,
                        "failed_samples": len(generated_files),
                    },
                },
                "evaluation": {
                    "available": False,
                    "kind": mode,
                    "skip_count": len(contract.metric_ids),
                },
                "runtime": runtime_spec,
                "artifacts": {
                    "scorecard": _artifact_path(root, "scorecard.json"),
                    "runner_runtime_report": str(runtime_report_path.resolve()),
                    "stdout": data.get("stdout_path"),
                    "stderr": data.get("stderr_path"),
                },
            }
            write_json(scorecard_path, scorecard)

        artifacts = {
            "scorecard": str(scorecard_path.resolve()),
            "runner_runtime_report": str(runtime_report_path.resolve()),
        }
        for key, value in (("stdout", data.get("stdout_path")), ("stderr", data.get("stderr_path"))):
            if value:
                artifacts[key] = str(value)

        kwargs = data.get("kwargs")
        extra_metadata = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        normalizer_only = (
            scorecard_runtime_flags.get("validation_normalizer_only") is True
            or (
                scorecard_runtime_flags.get("normalization_ok") is True
                and not official_benchmark_verified
                and not integration_evidence
            )
        )
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=scorecard_path,
            raw_results_path=runtime_report_path,
            official_benchmark_verified=official_benchmark_verified,
            integration_evidence=integration_evidence,
            artifacts=artifacts,
            metadata={
                "mode": mode,
                "contract_only": False,
                "manifest_integration_status": self.entry.integration_status,
                "manifest_verification_status": self.entry.verification_status,
                "requires_upstream_runtime": contract.requires_upstream_runtime,
                "run_status": data.get("run_status"),
                "returncode": data.get("returncode"),
                "normalizer_only": normalizer_only,
                "runtime": runtime_spec,
                "runner_runtime_spec": runtime_spec,
                **extra_metadata,
                "normalization_ok": scorecard_runtime_flags.get("normalization_ok") is True,
            },
        )

    def evaluate(
        self,
        *,
        output_dir: str | Path,
        mode: str = "contract",
        generated_artifact_dir: str | Path | None = None,
        **kwargs: JsonValue,
    ) -> OfficialRunResult:
        """Run ``prepare → run → collect → normalize``."""
        prepared = self.prepare(
            output_dir=output_dir,
            mode=mode,
            generated_artifact_dir=generated_artifact_dir,
            **kwargs,
        )
        run_result = self.run(prepared)
        collected = self.collect(run_result)
        return self.normalize(collected)


# ---------------------------------------------------------------------------
# BenchmarkRunnerRegistry and public API
# ---------------------------------------------------------------------------


class BenchmarkRunnerRegistry:
    """Lookup table of benchmark-zoo entries with contract runner surfaces."""

    def __init__(
        self,
        entries: Iterable[BenchmarkZooEntry | Mapping[str, JsonValue]],
        *,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.zoo = BenchmarkZooRegistry(entries)
        self.manifest_path = None if manifest_path is None else Path(manifest_path)

    def __contains__(self, key: object) -> bool:
        return key in self.zoo

    def __len__(self) -> int:
        return len(self.zoo)

    def __iter__(self) -> Iterator[BenchmarkZooEntry]:
        return iter(self.list_entries())

    def list_entries(self) -> list[BenchmarkZooEntry]:
        """Return sorted registered benchmark entries."""
        return self.zoo.list()

    def by_integration_status(self, status: str) -> list[BenchmarkZooEntry]:
        """Filter entries by ``integration_status``."""
        return self.zoo.by_integration_status(status)

    def integrated(self) -> list[BenchmarkZooEntry]:
        """Return entries with ``integration_status == integrated``."""
        return self.by_integration_status("integrated")

    def planned(self) -> list[BenchmarkZooEntry]:
        """Return entries with ``integration_status == planned``."""
        return self.by_integration_status("planned")

    def blocked(self) -> list[BenchmarkZooEntry]:
        """Return entries with ``integration_status == blocked``."""
        return self.by_integration_status("blocked")

    def has_contract_runner(self, benchmark_id: str) -> bool:
        """Return True when an external benchmark contract exists."""
        entry = self.zoo.get(benchmark_id)
        try:
            get_external_benchmark_contract(entry.benchmark_id)
        except KeyError:
            return False
        return True

    def has_official_runner(self, benchmark_id: str) -> bool:
        """Return True when manifest marks verified official integration."""
        entry = self.zoo.get(benchmark_id)
        return (
            entry.integration_status == "integrated"
            and entry.verification_status == "verified"
            and entry.official_benchmark_verified
            and entry.integration_evidence
        )

    def has_runner(self, benchmark_id: str) -> bool:
        """Return True when a contract runner is available."""
        return self.has_contract_runner(benchmark_id)

    def get_runner(self, benchmark_id: str) -> ManifestBenchmarkRunner:
        """Return :class:`ManifestBenchmarkRunner` for ``benchmark_id``."""
        try:
            entry = self.zoo.get(benchmark_id)
        except UnknownBenchmarkZooKeyError:
            raise
        if not self.has_runner(benchmark_id):
            raise BenchmarkExecutionUnavailableError(
                f"{entry.benchmark_id!r} has no registered contract runner; "
                f"manifest status is {entry.integration_status}/{entry.verification_status}"
            )
        return ManifestBenchmarkRunner(entry=entry, manifest_path=self.manifest_path)


def build_benchmark_runner_registry(path: str | Path = DEFAULT_MANIFEST_PATH) -> BenchmarkRunnerRegistry:
    """Load benchmark-zoo manifests and build a :class:`BenchmarkRunnerRegistry`."""
    manifest_path = Path(path)
    entries = load_benchmark_zoo_registry(manifest_path).list() if manifest_path.is_dir() else load_entries(manifest_path)
    return BenchmarkRunnerRegistry(entries, manifest_path=manifest_path)


def run_benchmark_execution(
    benchmark_id: str,
    *,
    output_dir: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    mode: str = "contract",
    generated_artifact_dir: str | Path | None = None,
    **kwargs: JsonValue,
) -> OfficialRunResult:
    """Run full benchmark lifecycle for ``benchmark_id``."""
    registry = build_benchmark_runner_registry(manifest_path)
    runner = registry.get_runner(benchmark_id)
    return runner.evaluate(
        output_dir=output_dir,
        mode=mode,
        generated_artifact_dir=generated_artifact_dir,
        **kwargs,
    )
