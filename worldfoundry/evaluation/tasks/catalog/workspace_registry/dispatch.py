"""Workspace benchmark dispatch helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from worldfoundry.core.execution.process import read_text_tail, run_logged_subprocess
from worldfoundry.core.io.paths import project_root
from worldfoundry.evaluation.tasks.catalog.benchmark_id import normalize_benchmark_id
from worldfoundry.evaluation.tasks.catalog.dispatch import CLI_RUNNERS
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

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


def _official_spec(benchmark_id: str | None):
    from worldfoundry.evaluation.tasks.catalog.dispatch import official_runner_spec

    return official_runner_spec(benchmark_id or "")


def workspace_benchmark_supported(benchmark_id: str | None) -> bool:
    spec = _official_spec(benchmark_id)
    return bool(spec and (spec.module or spec.workspace_dispatch))


def workspace_benchmark_has_input(payload: Any) -> bool:
    config = _runtime_config(payload)
    results_path, generated_path = _result_and_video_inputs(payload, config)
    return bool(
        _first_non_empty(
            results_path,
            generated_path,
            _payload_get(payload, "dataset_manifest"),
            config.get("dataset_json"),
            config.get("prompt_file"),
            _payload_get(payload, "answer_manifest"),
            config.get("answer_manifest"),
            config.get("run_fixture"),
        )
    )


def _catalog_metric_ids(benchmark_id: str) -> list[str]:
    try:
        from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
        from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR

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
        from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
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
            "runner_module": "worldfoundry.evaluation.tasks.execution.runners.vbench.run_vbench_official_runner",
        }
    except Exception:
        pass

    try:
        from worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import (
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
            "runner_module": "worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.run_vbench_2_0_official_runner",
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
            **{variant: names for variant, names in sorted(variant_dimensions.items())},
            **plus_presets,
        },
        "primary_presets": {variant: names for variant, names in sorted(variant_dimensions.items())},
        "variants": sorted(variant_dimensions),
        "default_variant": "long",
        "input_kind": "generated_video_dir_or_official_results",
        "supports_existing_results": True,
        "supports_official_runtime": True,
        "accepts_generated_artifacts": True,
        "runner_module": "worldfoundry.evaluation.tasks.execution.runners.vbench_plus_plus.run_vbench_plus_plus_official_runner",
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


def _scorecard_to_workspace_result(
    scorecard: Mapping[str, Any],
    *,
    output_dir: str | Path,
    benchmark_id: str,
    request: Mapping[str, Any],
    delegate_runner: str,
) -> dict[str, Any]:
    run = scorecard.get("run", {}) if isinstance(scorecard, Mapping) else {}
    artifacts = scorecard.get("artifacts", {}) if isinstance(scorecard, Mapping) else {}
    dataset = scorecard.get("dataset", {}) if isinstance(scorecard, Mapping) else {}
    generation = scorecard.get("generation", {}) if isinstance(scorecard, Mapping) else {}
    metrics = scorecard.get("metrics", {}) if isinstance(scorecard, Mapping) else {}
    summary = metrics.get("summary", {}) if isinstance(metrics, Mapping) else {}
    leaderboard = metrics.get("leaderboard", {}) if isinstance(metrics, Mapping) else {}
    run_status = str(run.get("status") or "completed").strip().lower()
    output_dir_path = Path(output_dir)
    return {
        "schema_version": "worldfoundry-evaluate-run-result",
        "mode": "existing-results",
        "delegate_runner": delegate_runner,
        "status": run_status,
        "exit_code": int(run.get("workspace_runner_returncode", run.get("returncode") or 0)),
        "output_dir": str(output_dir_path),
        "manifest_path": "",
        "execution_plan_path": "",
        "scorecard_path": str(artifacts.get("scorecard") or output_dir_path / "scorecard.json"),
        "raw_metric_table_path": str(
            artifacts.get("raw_metric_table")
            or artifacts.get("raw_metric_table_path")
            or output_dir_path / "raw_metric_table.jsonl"
        ),
        "upstream_results_path": str(artifacts.get("upstream_results") or ""),
        "sample_count": int(
            summary.get("sample_count") or dataset.get("generated_file_count") or generation.get("successful") or 0
        ),
        "successful_sample_count": int(generation.get("successful") or 0),
        "failed_sample_count": int(generation.get("failed") or 0),
        "artifact_count": int(dataset.get("generated_file_count") or generation.get("successful") or 0),
        "benchmark_id": benchmark_id,
        "leaderboard_metrics": leaderboard,
        "normalization_ok": bool(scorecard.get("normalization_ok")) if isinstance(scorecard, Mapping) else False,
        "official_benchmark_verified": bool(scorecard.get("official_benchmark_verified"))
        if isinstance(scorecard, Mapping)
        else False,
        "normalizer_only": bool(
            scorecard.get("validation", {}).get("normalizer_only")
            if isinstance(scorecard.get("validation"), Mapping)
            else scorecard.get("normalizer_only")
        )
        if isinstance(scorecard, Mapping)
        else False,
        "request": request,
        "scorecard": scorecard,
    }


def _run_classic_vbench(
    payload: Any, output_dir: str | Path, log_callback: Callable[[str, str], None] | None
) -> dict[str, Any]:
    from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import (
        VBenchRunRequest,
        run_vbench,
        split_dimensions,
    )

    config = _runtime_config(payload)
    results_path, generated_path = _result_and_video_inputs(payload, config)
    dimensions = split_dimensions(
        _selected_metrics(payload, config, ("aesthetic_quality",)),
        _string_list(config.get("presets") or config.get("preset")),
    )
    if not dimensions:
        dimensions = split_dimensions(["aesthetic_quality"])
    mode = _first_non_empty(config.get("mode"), config.get("vbench_mode"))
    if mode is None:
        mode = "custom_input" if generated_path and not results_path else "vbench_standard"
    request = VBenchRunRequest(
        output_dir=output_dir,
        videos_path=generated_path,
        dimensions=tuple(dimensions),
        benchmark_id=_payload_get(payload, "benchmark_id") or "vbench",
        vbench_root=_first_non_empty(config.get("vbench_root"), config.get("runtime_root"))
        or VBenchRunRequest.__dataclass_fields__["vbench_root"].default,
        mode=mode,
        prompt=_first_non_empty(config.get("prompt"), _payload_get(payload, "prompt")),
        prompt_file=_first_non_empty(config.get("prompt_file"), config.get("prompt_path")),
        category=_first_non_empty(config.get("category")),
        imaging_quality_preprocessing_mode=_first_non_empty(
            config.get("imaging_quality_preprocessing_mode"),
            config.get("imaging_preprocessing_mode"),
        )
        or "longer",
        full_json_dir=_first_non_empty(
            config.get("full_json_dir"), config.get("prompt_suite_json"), _payload_get(payload, "dataset_manifest")
        ),
        python=_first_non_empty(config.get("python"), config.get("python_bin")) or sys.executable,
        timeout=_int_config_value(config.get("timeout"), default=1800),
        load_ckpt_from_local=_truthy_config_value(config.get("load_ckpt_from_local"), default=False),
        read_frame=_truthy_config_value(config.get("read_frame"), default=False),
        from_upstream_results=results_path,
    )
    if log_callback is not None:
        source = request.from_upstream_results or request.videos_path or "missing-input"
        log_callback(
            "system", f"vbench mode={request.mode} source={source} dimensions={','.join(request.dimensions)}\n"
        )
    scorecard = run_vbench(request)
    return _scorecard_to_workspace_result(
        scorecard,
        output_dir=output_dir,
        benchmark_id=request.benchmark_id,
        request=asdict(request),
        delegate_runner="benchmark_zoo_vbench_official_runner",
    )


def _series_variant_and_mode(
    benchmark_id: str, config: Mapping[str, Any], generated_path: str | None
) -> tuple[str, str]:
    if benchmark_id == "vbench-2.0":
        variant = "vbench2"
        default_mode = "custom_input" if generated_path else "vbench_standard"
    else:
        variant = str(config.get("variant") or config.get("vbench_variant") or "long").strip() or "long"
        if variant not in {"i2v", "long", "trustworthiness"}:
            variant = "long"
        default_mode = (
            "long_custom_input"
            if variant == "long" and generated_path
            else "custom_input"
            if generated_path
            else "vbench_standard"
        )
    mode = str(config.get("mode") or config.get("vbench_mode") or default_mode)
    return variant, mode


def _run_vbench_series(
    payload: Any, output_dir: str | Path, log_callback: Callable[[str, str], None] | None
) -> dict[str, Any]:
    benchmark_id = benchmark_key(_payload_get(payload, "benchmark_id")) or "vbench-plus-plus"
    config = _runtime_config(payload)
    results_path, generated_path = _result_and_video_inputs(payload, config)
    variant, mode = _series_variant_and_mode(benchmark_id, config, generated_path)
    selected = _selected_metrics(payload, config, ("diversity",) if variant == "vbench2" else ("temporal_flickering",))
    if not selected:
        selected = ["diversity" if variant == "vbench2" else "temporal_flickering"]
    module = (
        "worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.run_vbench_2_0_official_runner"
        if benchmark_id == "vbench-2.0"
        else "worldfoundry.evaluation.tasks.execution.runners.vbench_plus_plus.run_vbench_plus_plus_official_runner"
    )
    command = [
        _first_non_empty(config.get("python"), config.get("python_bin")) or sys.executable,
        "-m",
        module,
        "--benchmark-id",
        benchmark_id,
        "--variant",
        variant,
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(_int_config_value(config.get("timeout"), default=7200)),
        "--json",
    ]
    if results_path:
        command.extend(["--official-results-path", results_path])
    if generated_path:
        command.extend(["--videos-path", generated_path])
    for dimension in selected:
        command.extend(["--dimension", dimension])
    dataset_root = _first_non_empty(_payload_get(payload, "dataset_root"), config.get("vbench2_dataset_root"))
    if benchmark_id == "vbench-2.0" and dataset_root and Path(dataset_root).exists():
        command.extend(["--vbench2-dataset-root", dataset_root])
    full_json_dir = _first_non_empty(
        config.get("full_json_dir"), config.get("prompt_suite_json"), _payload_get(payload, "dataset_manifest")
    )
    if full_json_dir:
        command.extend(["--full-json-dir", full_json_dir])
    if config.get("prompt") or _payload_get(payload, "prompt"):
        command.extend(["--prompt", str(config.get("prompt") or _payload_get(payload, "prompt"))])
    if config.get("prompt_file"):
        command.extend(["--prompt-file", str(config["prompt_file"])])
    if config.get("category"):
        command.extend(["--category", str(config["category"])])
    if variant == "trustworthiness" and _truthy_config_value(config.get("custom_input"), default=bool(generated_path)):
        command.append("--custom-input")
    return _run_cli_command(
        command,
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        delegate_runner="benchmark_zoo_vbench_series_official_runner",
        request={"command": command, "selected_metrics": selected, "variant": variant, "mode": mode},
        log_callback=log_callback,
    )


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


def _workspace_subprocess_timeout(config: Mapping[str, Any], *, grace_seconds: float = 60.0) -> float | None:
    """Parent-side backstop for delegated runner subprocesses.

    The delegated CLI usually enforces its own ``--timeout`` (defaulting to
    ``WORLDFOUNDRY_BENCHMARK_TIMEOUT``); the parent bound adds ``grace_seconds``
    on top so the child can finish writing its failed scorecard first. Returns
    None (unbounded, the historical behavior) when neither the workspace config
    nor the environment declares a timeout.
    """

    configured = config.get("timeout")
    if configured not in (None, ""):
        return float(_int_config_value(configured, default=7200)) + grace_seconds
    env_default = os.environ.get("WORLDFOUNDRY_BENCHMARK_TIMEOUT")
    if env_default:
        try:
            return float(env_default) + grace_seconds
        except ValueError:
            return None
    return None


def _run_cli_benchmark(
    payload: Any, output_dir: str | Path, log_callback: Callable[[str, str], None] | None
) -> dict[str, Any]:
    benchmark_id = benchmark_key(_payload_get(payload, "benchmark_id"))
    spec = CLI_RUNNERS[benchmark_id]
    config = _runtime_config(payload)
    command = build_workspace_benchmark_command(payload, output_dir)
    selected = _selected_metrics(payload, config, spec.default_metrics)
    run_fixture = "--run-fixture" in command
    run_official = "--run-official" in command
    return _run_cli_command(
        command,
        output_dir=output_dir,
        benchmark_id=benchmark_id,
        delegate_runner=f"benchmark_zoo_{benchmark_id.replace('-', '_')}_official_runner",
        request={
            "command": command,
            "selected_metrics": selected,
            "run_official": run_official,
            "run_fixture": run_fixture,
        },
        log_callback=log_callback,
        timeout=_workspace_subprocess_timeout(config),
    )


def _run_cli_command(
    command: list[str],
    *,
    output_dir: str | Path,
    benchmark_id: str,
    delegate_runner: str,
    request: Mapping[str, Any],
    log_callback: Callable[[str, str], None] | None,
    timeout: float | None = None,
) -> dict[str, Any]:
    from uuid import uuid4

    from worldfoundry.evaluation.tasks.execution.orchestration.run_session import run_stage, session_paths, utcnow_iso
    from worldfoundry.evaluation.utils import read_json_object, write_json

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if log_callback is not None:
        log_callback("system", f"{benchmark_id} runner={' '.join(command)}\n")
    stdout_log = output_dir_path / "workspace_runner_stdout.log"
    stderr_log = output_dir_path / "workspace_runner_stderr.log"
    paths = session_paths(output_dir_path)
    run_id = f"workspace-{uuid4().hex[:12]}"
    with run_stage(paths, "evaluate", manifest={
        "run_id": run_id, "runner": delegate_runner, "benchmark": {"benchmark_id": benchmark_id},
    }):
        try:
            completed = run_logged_subprocess(
                command,
                stdout_path=stdout_log,
                stderr_path=stderr_log,
                cwd=REPO_ROOT,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            tail = read_text_tail(stderr_log, max_lines=40) or read_text_tail(stdout_log, max_lines=40)
            raise RuntimeError(
                f"{benchmark_id} runner timed out after {timeout}s and was terminated; tail={tail}"
            ) from None
        scorecard_path = output_dir_path / "scorecard.json"
        if not scorecard_path.is_file():
            tail = read_text_tail(stderr_log, max_lines=40) or read_text_tail(stdout_log, max_lines=40)
            raise RuntimeError(
                f"{benchmark_id} runner did not write scorecard.json; exit={completed.returncode}; tail={tail}"
            )
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        run = scorecard.setdefault("run", {})
        run.setdefault("run_id", run_id)
        run["workspace_runner_returncode"] = completed.returncode
        if completed.returncode != 0:
            run["status"] = "failed"
            scorecard.setdefault("eligibility", {}).update(
                score_valid=False, leaderboard_valid=False, leaderboard_eligible=False,
            )
            # Keep partial scores, but discard any completed reports emitted before the failure.
            for key in ("run_summary", "report"):
                paths[key].unlink(missing_ok=True)
        write_json(scorecard_path, scorecard)
        result = _scorecard_to_workspace_result(
            scorecard,
            output_dir=output_dir_path,
            benchmark_id=benchmark_id,
            request=request,
            delegate_runner=delegate_runner,
        )
        manifest = read_json_object(paths["manifest"])
        manifest.update(run_id=run["run_id"], status=result["status"], exit_code=result["exit_code"],
                        finished_at=utcnow_iso())
        write_json(paths["manifest"], manifest)
        result["manifest_path"] = str(paths["manifest"])
        return result


def run_workspace_benchmark(
    payload: Any,
    output_dir: str | Path,
    *,
    log_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    benchmark_id = benchmark_key(_payload_get(payload, "benchmark_id"))
    spec = _official_spec(benchmark_id)
    dispatch = spec.workspace_dispatch if spec else None
    if dispatch == "vbench_classic":
        return _run_classic_vbench(payload, output_dir, log_callback)
    if dispatch == "vbench_series":
        return _run_vbench_series(payload, output_dir, log_callback)
    if spec is not None and spec.module:
        return _run_cli_benchmark(payload, output_dir, log_callback)
    raise KeyError(f"unsupported Workspace benchmark runner: {benchmark_id}")


def validate_workspace_registry() -> list[str]:
    issues: list[str] = []
    try:
        from worldfoundry.evaluation.tasks.catalog.dispatch import VIDEO_RUNNER_REGISTRY
    except Exception as exc:  # pragma: no cover - defensive catalog validation
        VIDEO_RUNNER_REGISTRY = {}
        issues.append(f"framework runner registry unavailable: {type(exc).__name__}: {exc}")

    from worldfoundry.evaluation.tasks.catalog.dispatch import official_runner_registry, script_to_module

    workspace_ids = {
        key for key, spec in official_runner_registry().items() if spec.module or spec.workspace_dispatch
    }
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
                f"{benchmark_id}: Workspace module {spec.module} "
                f"does not match framework runner {expected_module}"
            )

    for benchmark_id, spec in sorted(CLI_RUNNERS.items()):
        module_path = REPO_ROOT / (spec.module.replace(".", "/") + ".py")
        if not module_path.is_file():
            issues.append(f"{benchmark_id}: runner module not found: {spec.module}")

    # The integration registry is the public benchmark_integration_spec() surface;
    # every framework-registered runner must appear there so the public API never
    # silently returns None for a wired benchmark (review finding ET-01).
    try:
        from worldfoundry.evaluation.tasks.catalog.dispatch import BENCHMARK_INTEGRATION_REGISTRY
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
