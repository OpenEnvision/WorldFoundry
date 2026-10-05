"""Load official-runner dispatch from catalog YAML only.

One :class:`OfficialRunnerSpec` per benchmark id. There is no handwritten
``_LEGACY_*`` table. Studio, zoo subprocess, integration metadata, official-result
routing, and scoring-engine membership all read this loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    iter_benchmark_catalog_manifest_paths,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_id import normalize_benchmark_id
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, REPO_ROOT, load_manifest

_LOGGER = logging.getLogger(__name__)

_DEFAULT_RESULTS_FLAG = "--official-results-path"
_DEFAULT_GENERATED_ARG = "--generated-video-dir"
_DEFAULT_INPUT_KIND = "generated_video_dir_or_official_results"


def script_to_module(script: str) -> str:
    return script.removesuffix(".py").replace("/", ".")


def _entry_id(entry: Mapping[str, Any]) -> str | None:
    benchmark_id = entry.get("benchmark_id") or entry.get("id")
    if isinstance(benchmark_id, str) and benchmark_id.strip():
        return normalize_benchmark_id(benchmark_id)
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _opt_str(mapping: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    if key not in mapping:
        return default
    return _nullable_str(mapping[key])


def _opt_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in mapping:
        return default
    return _bool(mapping[key], default)


def parse_scoring_hook(hook: str) -> tuple[str, str]:
    module, sep, attr = hook.rpartition(":")
    if not sep or not module or not attr:
        raise ValueError(f"scoring hook must be module:attr, got {hook!r}")
    return module, attr


@dataclass(frozen=True)
class OfficialRunnerSpec:
    """Catalog-backed official runner, workspace CLI, routing, and scoring membership."""

    benchmark_id: str
    script: str = ""
    module: str = ""
    results_flag: str = _DEFAULT_RESULTS_FLAG
    extra_args: tuple[str, ...] = ()
    pass_benchmark_id: bool = True
    generated_arg: str | None = _DEFAULT_GENERATED_ARG
    dataset_root_arg: str | None = None
    dataset_manifest_arg: str | None = None
    prompt_manifest_arg: str | None = None
    answer_manifest_arg: str | None = None
    metric_arg: str | None = None
    dimension_arg: str | None = None
    limit_arg: str | None = None
    split_arg: str | None = None
    phase_arg: str | None = None
    benchmark_id_arg: str | None = None
    model_arg: str | None = None
    supports_run_official: bool = True
    supports_official_runtime: bool = False
    accepts_generated_artifacts: bool = False
    default_run_official: bool = False
    supports_fixture: bool = False
    default_metrics: tuple[str, ...] = ()
    default_mode: str | None = None
    input_kind: str = _DEFAULT_INPUT_KIND
    workspace_dispatch: str | None = None
    integration_tier: str | None = None
    hf_dataset_id: str | None = None
    judge_model_id: str | None = None
    artifact_official_run: bool = False
    pass_generated_artifact_dir: bool = False
    result_dispatch: str | None = None
    embodied_track: str | None = None
    scoring_engines: tuple[str, ...] = ()
    scoring_hooks: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def results_arg(self) -> str:
        return self.results_flag

    def scoring_hook(self, engine: str) -> str | None:
        for name, target in self.scoring_hooks:
            if name == engine:
                return target
        return None

    def scoring_hook_parts(self, engine: str) -> tuple[str, str] | None:
        hook = self.scoring_hook(engine)
        if hook is None:
            return None
        return parse_scoring_hook(hook)


VideoRunnerSpec = OfficialRunnerSpec
WorkspaceRunnerSpec = OfficialRunnerSpec


class IntegrationTier(str, Enum):
    """Benchmark execution tier inside WorldFoundry."""

    IN_TREE = "in_tree"
    MODEL_BACKED = "model_backed"
    NORMALIZER_ONLY = "normalizer_only"


@dataclass(frozen=True)
class BenchmarkIntegrationSpec:
    """Integration metadata for one video benchmark."""

    benchmark_id: str
    tier: IntegrationTier
    runner_script: str
    hf_dataset_id: str | None = None
    judge_model_id: str | None = None

    @property
    def in_tree_runner(self) -> Path:
        return REPO_ROOT / self.runner_script


@lru_cache(maxsize=1)
def load_catalog_dispatch_entries() -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return ``(benchmark_id, raw_entry)`` for every video and embodied catalog shard."""
    from worldfoundry.evaluation.tasks.catalog.schema import iter_benchmark_zoo_payloads

    entries: list[tuple[str, Mapping[str, Any]]] = []
    for path in iter_benchmark_catalog_manifest_paths(BENCHMARK_ZOO_DIR):
        try:
            payload = load_manifest(path)
            items = iter_benchmark_zoo_payloads(payload)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            _LOGGER.warning("skipping malformed benchmark catalog shard %s: %s", path, exc)
            continue
        for item in items:
            benchmark_id = _entry_id(item)
            if benchmark_id is None:
                continue
            entries.append((benchmark_id, item))
    return tuple(entries)


def clear_catalog_dispatch_cache() -> None:
    load_catalog_dispatch_entries.cache_clear()
    official_runner_registry.cache_clear()
    refresh_compat_views()


def _parse_scoring(runner: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    scoring = _as_mapping(runner.get("scoring")) or {}
    engines = _str_tuple(scoring.get("engines"))
    hooks: list[tuple[str, str]] = []
    raw_hooks = scoring.get("hooks")
    if isinstance(raw_hooks, Mapping):
        for engine, target in raw_hooks.items():
            if target:
                hooks.append((str(engine), str(target)))
    return engines, tuple(hooks)


def _parse_normalization(runner: Mapping[str, Any]) -> dict[str, Any]:
    normalization = _as_mapping(runner.get("normalization")) or {}
    return {
        "artifact_official_run": _bool(normalization.get("artifact_official_run")),
        "pass_generated_artifact_dir": _bool(normalization.get("pass_generated_artifact_dir")),
        "result_dispatch": _nullable_str(normalization.get("dispatch")),
        "embodied_track": _nullable_str(normalization.get("embodied_track")),
    }


@lru_cache(maxsize=1)
def official_runner_registry() -> dict[str, OfficialRunnerSpec]:
    """Every catalog entry that declares an official runner, workspace, or embodied track."""
    registry: dict[str, OfficialRunnerSpec] = {}
    for benchmark_id, entry in load_catalog_dispatch_entries():
        runner = _as_mapping(entry.get("runner")) or {}
        workspace = _as_mapping(runner.get("workspace")) or {}
        integration = _as_mapping(entry.get("integration")) or {}
        official_script = runner.get("official_script")
        script = official_script.strip() if isinstance(official_script, str) else ""
        dispatch = _nullable_str(workspace.get("dispatch"))
        raw_module = workspace.get("module")
        if isinstance(raw_module, str) and raw_module.strip():
            module = raw_module.strip()
        elif script and not dispatch:
            module = script_to_module(script)
        else:
            module = ""
        scoring_engines, scoring_hooks = _parse_scoring(runner)
        normalization = _parse_normalization(runner)
        results_flag = _DEFAULT_RESULTS_FLAG
        if runner.get("results_flag") is not None:
            results_flag = str(runner["results_flag"])
        elif workspace.get("results_arg") is not None:
            results_flag = str(workspace["results_arg"])
        spec = OfficialRunnerSpec(
            benchmark_id=benchmark_id,
            script=script,
            module=module,
            results_flag=results_flag,
            extra_args=_str_tuple(runner["extra_args"]) if "extra_args" in runner else (),
            pass_benchmark_id=_opt_bool(runner, "pass_benchmark_id", True),
            generated_arg=_opt_str(workspace, "generated_arg", _DEFAULT_GENERATED_ARG),
            dataset_root_arg=_opt_str(workspace, "dataset_root_arg"),
            dataset_manifest_arg=_opt_str(workspace, "dataset_manifest_arg"),
            prompt_manifest_arg=_opt_str(workspace, "prompt_manifest_arg"),
            answer_manifest_arg=_opt_str(workspace, "answer_manifest_arg"),
            metric_arg=_opt_str(workspace, "metric_arg"),
            dimension_arg=_opt_str(workspace, "dimension_arg"),
            limit_arg=_opt_str(workspace, "limit_arg"),
            split_arg=_opt_str(workspace, "split_arg"),
            phase_arg=_opt_str(workspace, "phase_arg"),
            benchmark_id_arg=_opt_str(workspace, "benchmark_id_arg"),
            model_arg=_opt_str(workspace, "model_arg"),
            supports_run_official=_opt_bool(workspace, "supports_run_official", True),
            supports_official_runtime=_opt_bool(workspace, "supports_official_runtime", False),
            accepts_generated_artifacts=_opt_bool(workspace, "accepts_generated_artifacts", False),
            default_run_official=_opt_bool(workspace, "default_run_official", False),
            supports_fixture=_opt_bool(workspace, "supports_fixture", False),
            default_metrics=_str_tuple(workspace["default_metrics"]) if "default_metrics" in workspace else (),
            default_mode=_opt_str(workspace, "default_mode"),
            input_kind=_opt_str(workspace, "input_kind", _DEFAULT_INPUT_KIND) or _DEFAULT_INPUT_KIND,
            workspace_dispatch=dispatch,
            integration_tier=_nullable_str(integration.get("tier")),
            hf_dataset_id=_opt_str(integration, "hf_dataset_id"),
            judge_model_id=_opt_str(integration, "judge_model_id"),
            scoring_engines=scoring_engines,
            scoring_hooks=scoring_hooks,
            **normalization,
        )
        if not (
            spec.script
            or spec.module
            or spec.workspace_dispatch
            or spec.result_dispatch == "embodied"
            or spec.embodied_track
        ):
            continue
        registry[benchmark_id] = spec
    return registry


def official_runner_spec(benchmark_id: str) -> OfficialRunnerSpec | None:
    return official_runner_registry().get(normalize_benchmark_id(benchmark_id))


def video_runner_registry() -> dict[str, OfficialRunnerSpec]:
    return {key: spec for key, spec in official_runner_registry().items() if spec.script}


def workspace_runner_registry() -> dict[str, OfficialRunnerSpec]:
    return {key: spec for key, spec in official_runner_registry().items() if spec.module}


def scoring_benchmark_ids(engine: str) -> tuple[str, ...]:
    return tuple(
        key
        for key, spec in official_runner_registry().items()
        if engine in spec.scoring_engines
    )


def scoring_hook_table(engine: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for key, spec in official_runner_registry().items():
        if engine not in spec.scoring_engines:
            continue
        hook = spec.scoring_hook(engine)
        if hook:
            table[key] = hook
    return table


def artifact_official_run_benchmarks() -> frozenset[str]:
    return frozenset(key for key, spec in official_runner_registry().items() if spec.artifact_official_run)


def pass_generated_artifact_dir_benchmarks() -> frozenset[str]:
    return frozenset(key for key, spec in official_runner_registry().items() if spec.pass_generated_artifact_dir)


def embodied_result_normalizer_tracks() -> dict[str, str]:
    tracks: dict[str, str] = {}
    for key, spec in official_runner_registry().items():
        if spec.result_dispatch == "embodied" or spec.embodied_track:
            tracks[key] = spec.embodied_track or "vla"
    return tracks


# Compatibility views used by Studio, zoo, and public API.
VIDEO_RUNNER_REGISTRY: dict[str, OfficialRunnerSpec] = {}
CLI_RUNNERS: dict[str, OfficialRunnerSpec] = {}
BENCHMARK_INTEGRATION_REGISTRY: dict[str, BenchmarkIntegrationSpec] = {}


def refresh_compat_views() -> None:
    VIDEO_RUNNER_REGISTRY.clear()
    VIDEO_RUNNER_REGISTRY.update(video_runner_registry())
    CLI_RUNNERS.clear()
    CLI_RUNNERS.update(workspace_runner_registry())
    BENCHMARK_INTEGRATION_REGISTRY.clear()
    BENCHMARK_INTEGRATION_REGISTRY.update(merge_integration_registry())


def video_runner_overrides() -> dict[str, dict[str, Any]]:
    """Deprecated: catalog is the only source. Kept for existing callers."""
    return {
        key: {"script": spec.script, "results_flag": spec.results_flag}
        for key, spec in video_runner_registry().items()
    }


def integration_overrides() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "tier": spec.integration_tier,
            "runner_script": spec.script,
            "hf_dataset_id": spec.hf_dataset_id,
            "judge_model_id": spec.judge_model_id,
        }
        for key, spec in official_runner_registry().items()
        if spec.integration_tier and spec.script
    }


def workspace_runner_overrides() -> dict[str, dict[str, Any]]:
    return {key: {"module": spec.module} for key, spec in workspace_runner_registry().items()}


def merge_video_runner_registry(legacy: Mapping[str, Any] | None = None) -> dict[str, OfficialRunnerSpec]:
    del legacy
    return dict(video_runner_registry())


def merge_integration_registry(legacy: Mapping[str, Any] | None = None) -> dict[str, BenchmarkIntegrationSpec]:
    del legacy
    merged: dict[str, BenchmarkIntegrationSpec] = {}
    allowed = ", ".join(item.value for item in IntegrationTier)
    for benchmark_id, spec in official_runner_registry().items():
        if not spec.script or not spec.integration_tier:
            continue
        try:
            tier = IntegrationTier(spec.integration_tier)
        except ValueError as exc:
            raise ValueError(
                f"{benchmark_id}: integration.tier must be one of {allowed}; got {spec.integration_tier!r}"
            ) from exc
        merged[benchmark_id] = BenchmarkIntegrationSpec(
            benchmark_id,
            tier,
            spec.script,
            hf_dataset_id=spec.hf_dataset_id,
            judge_model_id=spec.judge_model_id,
        )
    return merged


def video_runner_spec(benchmark_id: str) -> OfficialRunnerSpec | None:
    return VIDEO_RUNNER_REGISTRY.get(normalize_benchmark_id(benchmark_id))


def integration_spec(benchmark_id: str) -> BenchmarkIntegrationSpec | None:
    return BENCHMARK_INTEGRATION_REGISTRY.get(normalize_benchmark_id(benchmark_id))


def specialized_result_normalizer_scripts() -> dict[str, tuple[str, str]]:
    return {benchmark_id: (spec.script, spec.results_flag) for benchmark_id, spec in VIDEO_RUNNER_REGISTRY.items()}


def merge_workspace_runner_registry(legacy: Mapping[str, Any] | None = None) -> dict[str, OfficialRunnerSpec]:
    del legacy
    return dict(workspace_runner_registry())


refresh_compat_views()


__all__ = [
    "BENCHMARK_INTEGRATION_REGISTRY",
    "BenchmarkIntegrationSpec",
    "CLI_RUNNERS",
    "IntegrationTier",
    "OfficialRunnerSpec",
    "VIDEO_RUNNER_REGISTRY",
    "VideoRunnerSpec",
    "WorkspaceRunnerSpec",
    "artifact_official_run_benchmarks",
    "clear_catalog_dispatch_cache",
    "embodied_result_normalizer_tracks",
    "integration_overrides",
    "integration_spec",
    "load_catalog_dispatch_entries",
    "merge_integration_registry",
    "merge_video_runner_registry",
    "merge_workspace_runner_registry",
    "official_runner_registry",
    "official_runner_spec",
    "parse_scoring_hook",
    "pass_generated_artifact_dir_benchmarks",
    "refresh_compat_views",
    "script_to_module",
    "scoring_benchmark_ids",
    "scoring_hook_table",
    "specialized_result_normalizer_scripts",
    "video_runner_overrides",
    "video_runner_registry",
    "video_runner_spec",
    "workspace_runner_overrides",
    "workspace_runner_registry",
]
