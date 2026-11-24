"""Benchmark subprocess commands: official-run command selection, runtime roots, environment, and timeouts."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.io.formats.serialization import write_json
from worldfoundry.core.io.paths import resolve_worldfoundry_path
from worldfoundry.runtime.env import benchmark_repo_cache_root

from ....utils import BENCHMARK_ZOO_DIR
from ...catalog.runner_kinds import IN_TREE_RUNTIME_KINDS
from ...catalog.schema import BenchmarkZooEntry

JsonValue = Any


REPO_ROOT = Path(__file__).resolve().parents[5]


DEFAULT_MANIFEST_PATH = BENCHMARK_ZOO_DIR


SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"


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
    from worldfoundry.evals.tasks.catalog.workspace_registry import (
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


def _resolve_python_command(command: str | tuple[str, ...] | list[str]) -> str | tuple[str, ...] | list[str]:
    """Rewrite leading ``python`` to ``sys.executable``."""
    if isinstance(command, (tuple, list)) and command and command[0] in {"python", "python3"}:
        return (sys.executable, *command[1:])
    return command


def _command_to_json(command: str | tuple[str, ...] | list[str] | None) -> JsonValue:
    """Coerce a command to JSON-safe string or list."""
    if command is None:
        return None
    return command if isinstance(command, str) else list(command)


def _subprocess_command(command: str | tuple[str, ...] | list[str]) -> str | list[str]:
    """Format a command for ``subprocess.run``."""
    return command if isinstance(command, str) else list(command)


def _timeout_seconds(value: JsonValue) -> float | None:
    """Parse timeout kwargs into seconds."""
    if value is None:
        from worldfoundry.evals.tasks.execution.framework.official_runner import (
            default_benchmark_timeout,
        )

        return default_benchmark_timeout()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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
        explicit_root = (
            clone_root or env_root or runtime.get("clone_dir") or runtime.get("root") or runtime.get("default_root")
        )
    else:
        explicit_root = (
            clone_root or runtime.get("clone_dir") or runtime.get("root") or runtime.get("default_root") or env_root
        )
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
        _runtime_root_env(
            runtime, clone_root=clone_root, external_repo_runtime=runtime_kind == "external_official_repo"
        )
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
