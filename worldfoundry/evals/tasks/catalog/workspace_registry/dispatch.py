"""Workspace benchmark dispatch helpers."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import project_root
from worldfoundry.evals.tasks.catalog.benchmark_id import normalize_benchmark_id
from worldfoundry.evals.tasks.catalog.dispatch import CLI_RUNNERS
from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

GENERIC_EVALUATION_METRICS = {"artifact_count", "required_artifacts_present"}
RESULT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".xlsx", ".xls"}

REPO_ROOT = project_root(__file__)


def benchmark_key(value: str | None) -> str:
    return normalize_benchmark_id(value)


def _payload_get(payload: Any, name: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.replace(",", " ").split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = []
        for item in value:
            raw_values.extend(str(item).replace(",", " ").split())
    else:
        raw_values = [str(value)]
    return [str(item).strip() for item in raw_values if str(item).strip()]


def _truthy_config_value(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        return text in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int_config_value(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _path_looks_like_results(value: str | Path | None) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    return path.is_file() or path.suffix.lower() in RESULT_SUFFIXES


def _runtime_config(payload: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name in ("params", "call_kwargs", "load_kwargs"):
        value = _payload_get(payload, name, {})
        if isinstance(value, Mapping):
            config.update(value)
    return config


def _selected_metrics(payload: Any, config: Mapping[str, Any], fallback: Sequence[str] = ()) -> list[str]:
    values = _string_list(_payload_get(payload, "metrics", ()))
    if not values or set(values).issubset(GENERIC_EVALUATION_METRICS):
        values = _string_list(
            config.get("dimensions")
            or config.get("dimension")
            or config.get("metrics")
            or config.get("metric")
            or fallback
        )
    return values


def _result_and_video_inputs(payload: Any, config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    results_path = _first_non_empty(
        _payload_get(payload, "results_path"), config.get("from_upstream_results"), config.get("official_results_path")
    )
    generated_path = _first_non_empty(
        config.get("videos_path"),
        config.get("generated_video_dir"),
        config.get("generated_artifact_dir"),
        config.get("videos_dir"),
    )
    if results_path and not _path_looks_like_results(results_path) and generated_path is None:
        generated_path = results_path
        results_path = None
    return results_path, generated_path


def _catalog_metric_ids(benchmark_id: str) -> list[str]:
    try:
        from worldfoundry.evals.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
        from worldfoundry.evals.utils import BENCHMARK_ZOO_DIR

        entry = load_benchmark_zoo_registry(BENCHMARK_ZOO_DIR).get(benchmark_id)
    except Exception:
        return []
    return [metric.metric_id for metric in entry.metrics]


@lru_cache(maxsize=1)
def _vbench_plus_plus_dimensions() -> dict[str, list[str]]:
    files = {
        "i2v": bundled_benchmark_asset("vbench-plus-plus", "i2v", "vbench2_i2v_full_info.json"),
        "long": bundled_benchmark_asset("vbench-plus-plus", "long", "VBench_full_info.json"),
        "trustworthiness": bundled_benchmark_asset("vbench-plus-plus", "trustworthiness", "vbench2_trustworthy.json"),
    }
    dimensions: dict[str, list[str]] = {}
    for variant, path in files.items():
        values: set[str] = set()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = []
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, Mapping):
                    continue
                raw_dimension = row.get("dimension")
                if isinstance(raw_dimension, str):
                    values.add(raw_dimension)
                elif isinstance(raw_dimension, Sequence):
                    for item in raw_dimension:
                        if isinstance(item, str) and item:
                            values.add(item)
        dimensions[variant] = sorted(values)
    return dimensions


@lru_cache(maxsize=1)
def workspace_benchmark_runtime_hints() -> dict[str, dict[str, Any]]:
    hints: dict[str, dict[str, Any]] = {}
    try:
        from worldfoundry.evals.tasks.execution.runners.vbench.vbench_official_impl import (
            vbench_dimensions_payload,
        )

        vbench = vbench_dimensions_payload()
        vbench_metrics = _catalog_metric_ids("vbench")
        hints["vbench"] = {
            **vbench,
            "metrics": vbench_metrics or vbench.get("dimensions", []),
            "default_metrics": ["aesthetic_quality"],
            "input_kind": "generated_video_dir_or_official_results",
            "supports_existing_results": True,
            "supports_official_runtime": True,
            "accepts_generated_artifacts": True,
            "runner_module": "worldfoundry.evals.tasks.execution.runners.vbench.run_vbench_official_runner",
        }
    except Exception:
        pass

    try:
        from worldfoundry.evals.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
            VBENCH2_CATEGORY_GROUPS,
            VBENCH2_DIMENSION_FILES,
        )

        dimensions = sorted(VBENCH2_DIMENSION_FILES)
        hints["vbench-2.0"] = {
            "benchmark_id": "vbench-2.0",
            "dimensions": dimensions,
            "metrics": _catalog_metric_ids("vbench-2.0") or ["vbench2_total"],
            "default_metrics": ["diversity"],
            "presets": {name: list(values) for name, values in sorted(VBENCH2_CATEGORY_GROUPS.items())},
            "primary_presets": {name: list(values) for name, values in sorted(VBENCH2_CATEGORY_GROUPS.items())},
            "input_kind": "generated_video_dir_or_official_results",
            "supports_existing_results": True,
            "supports_official_runtime": True,
            "accepts_generated_artifacts": True,
            "runner_module": "worldfoundry.evals.tasks.execution.runners.vbench_2_0.run_vbench_2_0_official_runner",
        }
    except Exception:
        pass

    variant_dimensions = _vbench_plus_plus_dimensions()
    plus_presets = {f"{variant}:{name}": [name] for variant, names in variant_dimensions.items() for name in names}
    plus_dimensions = sorted({name for names in variant_dimensions.values() for name in names})
    hints["vbench-plus-plus"] = {
        "benchmark_id": "vbench-plus-plus",
        "dimensions": plus_dimensions,
        "metrics": _catalog_metric_ids("vbench-plus-plus") or ["vbench_plus_plus_average"],
        "default_metrics": ["temporal_flickering"],
        "presets": {
            **dict(sorted(variant_dimensions.items())),
            **plus_presets,
        },
        "primary_presets": dict(sorted(variant_dimensions.items())),
        "variants": sorted(variant_dimensions),
        "default_variant": "long",
        "input_kind": "generated_video_dir_or_official_results",
        "supports_existing_results": True,
        "supports_official_runtime": True,
        "accepts_generated_artifacts": True,
        "runner_module": "worldfoundry.evals.tasks.execution.runners.vbench_plus_plus.run_vbench_plus_plus_official_runner",
    }

    for benchmark_id, spec in sorted(CLI_RUNNERS.items()):
        metrics = _catalog_metric_ids(benchmark_id)
        defaults = list(spec.default_metrics or metrics[:1])
        hints[benchmark_id] = {
            "benchmark_id": benchmark_id,
            "dimensions": metrics,
            "metrics": metrics,
            "default_metrics": defaults,
            "presets": {"all": metrics, "default": defaults},
            "primary_presets": {"default": defaults},
            "input_kind": spec.input_kind,
            "supports_existing_results": True,
            "supports_official_runtime": spec.supports_official_runtime,
            "accepts_generated_artifacts": spec.accepts_generated_artifacts,
            "supports_fixture": spec.supports_fixture,
            "runner_module": spec.module,
        }
    return hints


def workspace_benchmark_runtime_hint(benchmark_id: str | None) -> dict[str, Any]:
    return dict(workspace_benchmark_runtime_hints().get(benchmark_key(benchmark_id), {}))


def _extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def build_workspace_benchmark_command(payload: Any, output_dir: str | Path) -> list[str]:
    """Build the canonical CLI invocation for an in-tree benchmark runner.

    The same builder is used by Studio/Workspace and the public benchmark-zoo
    ``official-run`` path.  Keeping argument routing here prevents catalog YAML
    commands from drifting away from the actual runner parsers.
    """
    benchmark_id = benchmark_key(_payload_get(payload, "benchmark_id"))
    if benchmark_id not in CLI_RUNNERS:
        raise KeyError(f"unsupported CLI benchmark runner: {benchmark_id}")
    spec = CLI_RUNNERS[benchmark_id]
    config = _runtime_config(payload)
    results_path, generated_path = _result_and_video_inputs(payload, config)
    command = [
        _first_non_empty(config.get("python"), config.get("python_bin")) or sys.executable,
        "-m",
        spec.module,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if spec.benchmark_id_arg:
        command.extend([spec.benchmark_id_arg, benchmark_id])
    if results_path:
        command.extend([spec.results_arg, results_path])
    if generated_path and not spec.accepts_generated_artifacts:
        raise ValueError(
            f"{benchmark_id} does not currently accept generic generated artifacts; "
            "use official result import or a benchmark-specific prepared layout documented by its runner"
        )
    if generated_path and spec.generated_arg:
        command.extend([spec.generated_arg, generated_path])
    model_id = _first_non_empty(
        _payload_get(payload, "model_id"),
        config.get("model_id"),
        config.get("result_model_id"),
        config.get("model_name"),
    )
    if model_id and spec.model_arg:
        command.extend([spec.model_arg, model_id])
    dataset_root = _first_non_empty(_payload_get(payload, "dataset_root"), config.get("dataset_root"))
    if dataset_root and spec.dataset_root_arg:
        command.extend([spec.dataset_root_arg, dataset_root])
    dataset_manifest = _first_non_empty(
        _payload_get(payload, "dataset_manifest"), config.get("dataset_manifest"), config.get("dataset_json")
    )
    if dataset_manifest and spec.dataset_manifest_arg:
        command.extend([spec.dataset_manifest_arg, dataset_manifest])
    prompt_manifest = _first_non_empty(_payload_get(payload, "prompt_manifest"), config.get("prompt_manifest"))
    if prompt_manifest and spec.prompt_manifest_arg:
        command.extend([spec.prompt_manifest_arg, prompt_manifest])
    answer_manifest = _first_non_empty(
        _payload_get(payload, "answer_manifest"),
        config.get("answer_manifest"),
        config.get("prediction_manifest"),
    )
    if answer_manifest and spec.answer_manifest_arg:
        command.extend([spec.answer_manifest_arg, answer_manifest])
    selected = _selected_metrics(payload, config, spec.default_metrics)
    if selected and spec.dimension_arg:
        if benchmark_id == "worldarena":
            command.extend([spec.dimension_arg, *selected])
        elif benchmark_id == "video-bench":
            for dimension in selected:
                command.extend([spec.dimension_arg, dimension])
        else:
            command.extend([spec.dimension_arg, selected[0]])
    if selected and spec.metric_arg:
        if benchmark_id in {"evalcrafter", "physical-ai-bench", "wbench"}:
            command.extend([spec.metric_arg, ",".join(selected)])
        else:
            command.extend([spec.metric_arg, selected[0]])
    run_fixture = _truthy_config_value(config.get("run_fixture"), default=False)
    if run_fixture and spec.supports_fixture:
        command.append("--run-fixture")
    run_official = _truthy_config_value(
        config.get("run_official"),
        default=spec.default_run_official or bool(generated_path and not results_path),
    )
    if run_official and spec.supports_run_official and spec.supports_official_runtime:
        command.append("--run-official")
    limit = config.get("limit", config.get("num_samples"))
    if limit not in (None, "") and spec.limit_arg:
        command.extend([spec.limit_arg, str(_int_config_value(limit, default=1))])
    if config.get("split") not in (None, "") and spec.split_arg:
        command.extend([spec.split_arg, str(config["split"])])
    phase = config.get("phase", config.get("track"))
    if phase not in (None, "") and spec.phase_arg:
        command.extend([spec.phase_arg, str(phase)])
    if config.get("timeout"):
        command.extend(["--timeout", str(_int_config_value(config.get("timeout"), default=7200))])
    command.extend(_extra_args(config.get("extra_args") or config.get("runner_args")))
    return command


def validate_workspace_registry() -> list[str]:
    issues: list[str] = []
    try:
        from worldfoundry.evals.tasks.catalog.dispatch import VIDEO_RUNNER_REGISTRY
    except Exception as exc:  # pragma: no cover - defensive catalog validation
        VIDEO_RUNNER_REGISTRY = {}
        issues.append(f"framework runner registry unavailable: {type(exc).__name__}: {exc}")

    from worldfoundry.evals.tasks.catalog.dispatch import official_runner_registry, script_to_module

    workspace_ids = {key for key, spec in official_runner_registry().items() if spec.module or spec.workspace_dispatch}
    for benchmark_id, video_spec in sorted(VIDEO_RUNNER_REGISTRY.items()):
        if benchmark_id not in workspace_ids:
            issues.append(f"{benchmark_id}: missing Workspace benchmark runtime entry")
            continue
        spec = official_runner_registry().get(benchmark_id)
        if spec is None or spec.workspace_dispatch:
            continue
        expected_module = script_to_module(video_spec.script)
        if spec.module != expected_module:
            issues.append(
                f"{benchmark_id}: Workspace module {spec.module} does not match framework runner {expected_module}"
            )

    for benchmark_id, spec in sorted(CLI_RUNNERS.items()):
        module_path = REPO_ROOT / (spec.module.replace(".", "/") + ".py")
        if not module_path.is_file():
            issues.append(f"{benchmark_id}: runner module not found: {spec.module}")

    # The integration registry is the public benchmark_integration_spec() surface;
    # every framework-registered runner must appear there so the public API never
    # silently returns None for a wired benchmark (review finding ET-01).
    try:
        from worldfoundry.evals.tasks.catalog.dispatch import BENCHMARK_INTEGRATION_REGISTRY
    except Exception as exc:  # pragma: no cover - defensive catalog validation
        issues.append(f"benchmark integration registry unavailable: {type(exc).__name__}: {exc}")
    else:
        for benchmark_id, video_spec in sorted(VIDEO_RUNNER_REGISTRY.items()):
            integration = BENCHMARK_INTEGRATION_REGISTRY.get(benchmark_id)
            if integration is None:
                issues.append(f"{benchmark_id}: missing benchmark integration registry entry")
            elif integration.runner_script != video_spec.script:
                issues.append(
                    f"{benchmark_id}: integration runner script {integration.runner_script} "
                    f"does not match framework runner {video_spec.script}"
                )
        for benchmark_id in sorted(BENCHMARK_INTEGRATION_REGISTRY):
            if benchmark_id not in VIDEO_RUNNER_REGISTRY:
                issues.append(f"{benchmark_id}: integration entry has no framework runner registry entry")
    return issues
