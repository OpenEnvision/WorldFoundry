"""Workspace-facing dispatch for in-tree benchmark runners.

This module keeps Studio/Workspace evaluation glue separate from the web app.
Each benchmark runner remains responsible for its own official parsing and
scorecard generation; Workspace only supplies input paths, selected dimensions,
and runtime flags in a consistent way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from worldfoundry.core.io.paths import project_root
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

REPO_ROOT = project_root(__file__)
GENERIC_EVALUATION_METRICS = {"artifact_count", "required_artifacts_present"}
RESULT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".xlsx", ".xls"}


@dataclass(frozen=True)
class WorkspaceRunnerSpec:
    benchmark_id: str
    module: str
    results_arg: str = "--official-results-path"
    generated_arg: str | None = "--generated-video-dir"
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
    # Parser shape and actual capability are intentionally separate.  Some
    # result importers expose a historical --run-official flag without having
    # an evaluator that consumes caller-generated media.
    supports_run_official: bool = True
    supports_official_runtime: bool = False
    accepts_generated_artifacts: bool = False
    default_run_official: bool = False
    supports_fixture: bool = False
    default_metrics: tuple[str, ...] = ()
    default_mode: str | None = None
    input_kind: str = "generated_video_dir_or_official_results"


CLI_RUNNERS: dict[str, WorkspaceRunnerSpec] = {
    "4dworldbench": WorkspaceRunnerSpec(
        "4dworldbench",
        "worldfoundry.evaluation.tasks.execution.runners.four_d_worldbench.run_four_d_worldbench_official_runner",
        dimension_arg="--dimension",
        dataset_manifest_arg="--dataset-json",
        default_metrics=("perceptual_clip_iqa_metrics",),
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "aigcbench": WorkspaceRunnerSpec(
        "aigcbench",
        "worldfoundry.evaluation.tasks.execution.runners.aigcbench.run_aigcbench_official_runner",
        generated_arg=None,
        limit_arg="--limit",
    ),
    "camerabench": WorkspaceRunnerSpec(
        "camerabench",
        "worldfoundry.evaluation.tasks.execution.runners.camerabench.run_camerabench_official_runner",
        generated_arg=None,
        supports_run_official=False,
        dataset_root_arg="--benchmark-data-root",
        input_kind="score_dir_or_official_results",
    ),
    "chronomagic-bench": WorkspaceRunnerSpec(
        "chronomagic-bench",
        "worldfoundry.evaluation.tasks.execution.runners.chronomagic_bench.run_chronomagic_bench_official_runner",
        generated_arg="--generated-video-dir",
        dataset_root_arg="--dataset-root",
        supports_run_official=False,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        default_metrics=("chscore",),
    ),
    "devil-dynamics": WorkspaceRunnerSpec(
        "devil-dynamics",
        "worldfoundry.evaluation.tasks.execution.runners.devil_dynamics.run_devil_dynamics_official_runner",
        generated_arg="--generated-video-dir",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "evalcrafter": WorkspaceRunnerSpec(
        "evalcrafter",
        "worldfoundry.evaluation.tasks.execution.runners.evalcrafter.run_evalcrafter_official_runner",
        generated_arg="--generated-video-dir",
        prompt_manifest_arg="--prompt-manifest",
        metric_arg="--metrics",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        supports_fixture=True,
        default_metrics=("clip_score", "clip_temp_score", "face_consistency_score"),
    ),
    "ewmbench": WorkspaceRunnerSpec(
        "ewmbench",
        "worldfoundry.evaluation.tasks.execution.runners.ewmbench.run_ewmbench_official_runner",
        generated_arg=None,
        limit_arg="--limit",
        supports_fixture=True,
    ),
    "fetv": WorkspaceRunnerSpec(
        "fetv",
        "worldfoundry.evaluation.tasks.execution.runners.fetv.run_fetv_official_runner",
        generated_arg="--generated-video-dir",
        metric_arg="--fetv-metrics",
        limit_arg="--fetv-limit",
        model_arg="--fetv-model-name",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        default_metrics=("clip_score",),
    ),
    "genai-bench": WorkspaceRunnerSpec(
        "genai-bench",
        "worldfoundry.evaluation.tasks.execution.runners.genai_bench.run_genai_bench_official_runner",
        generated_arg=None,
        limit_arg="--limit",
        supports_fixture=True,
        default_metrics=("genai_bench_average",),
    ),
    "ipv-bench": WorkspaceRunnerSpec(
        "ipv-bench",
        "worldfoundry.evaluation.tasks.execution.runners.ipv_bench.run_ipv_bench_official_runner",
        generated_arg=None,
        limit_arg="--limit",
        supports_fixture=True,
    ),
    "iworld-bench": WorkspaceRunnerSpec(
        "iworld-bench",
        "worldfoundry.evaluation.tasks.execution.runners.iworldbench.run_iworldbench_official_runner",
        generated_arg="--generated-video-dir",
        dataset_root_arg="--dataset-root",
        limit_arg="--limit",
        split_arg="--split",
        metric_arg="--metric",
        supports_fixture=True,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "larybench": WorkspaceRunnerSpec(
        "larybench",
        "worldfoundry.evaluation.tasks.execution.runners.larybench.run_larybench_official_runner",
        generated_arg=None,
        dataset_root_arg="--dataset-root",
        split_arg="--split",
        model_arg="--model",
        supports_official_runtime=True,
        default_metrics=("top1_accuracy",),
        input_kind="dataset_root_or_official_results",
    ),
    "mirabench": WorkspaceRunnerSpec(
        "mirabench",
        "worldfoundry.evaluation.tasks.execution.runners.mirabench.run_mirabench_official_runner",
        generated_arg=None,
        supports_fixture=True,
    ),
    "memobench": WorkspaceRunnerSpec(
        "memobench",
        "worldfoundry.evaluation.tasks.execution.runners.memobench.run_memobench_official_runner",
        generated_arg="--generated-video-dir",
        supports_fixture=True,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "phyeduvideo": WorkspaceRunnerSpec(
        "phyeduvideo",
        "worldfoundry.evaluation.tasks.execution.runners.phyeduvideo.run_phyeduvideo_official_runner",
        generated_arg=None,
        limit_arg="--limit",
    ),
    "phyfps-bench-gen": WorkspaceRunnerSpec(
        "phyfps-bench-gen",
        "worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.run_phyfps_bench_gen_official_runner",
        generated_arg="--generated-artifact-dir",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "visual-chronometer": WorkspaceRunnerSpec(
        "visual-chronometer",
        "worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.run_visual_chronometer_official_runner",
        generated_arg="--generated-artifact-dir",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "phygenbench": WorkspaceRunnerSpec(
        "phygenbench",
        "worldfoundry.evaluation.tasks.execution.runners.phygenbench.run_phygenbench_official_runner",
        generated_arg=None,
        limit_arg="--limit",
        supports_fixture=True,
    ),
    "phyground": WorkspaceRunnerSpec(
        "phyground",
        "worldfoundry.evaluation.tasks.execution.runners.phyground.run_phyground_official_runner",
        generated_arg=None,
        limit_arg="--limit",
        supports_fixture=True,
    ),
    "physics-iq": WorkspaceRunnerSpec(
        "physics-iq",
        "worldfoundry.evaluation.tasks.execution.runners.physics_iq.run_physics_iq_official_runner",
        generated_arg="--generated-artifact-dir",
        dataset_root_arg="--dataset-root",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "physics-iq-verified": WorkspaceRunnerSpec(
        "physics-iq-verified",
        "worldfoundry.evaluation.tasks.execution.runners.physics_iq.run_physics_iq_official_runner",
        generated_arg="--generated-artifact-dir",
        dataset_root_arg="--dataset-root",
        limit_arg="--limit",
        benchmark_id_arg="--benchmark-id",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "physical-ai-bench": WorkspaceRunnerSpec(
        "physical-ai-bench",
        "worldfoundry.evaluation.tasks.execution.runners.physical_ai_bench.run_physical_ai_bench_official_runner",
        generated_arg="--generated-video-dir",
        dataset_root_arg="--dataset-root",
        prompt_manifest_arg="--prompt-file",
        answer_manifest_arg="--prediction-manifest",
        metric_arg="--metrics",
        phase_arg="--track",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        default_metrics=(
            "aesthetic_quality",
            "background_consistency",
            "imaging_quality",
            "motion_smoothness",
            "overall_consistency",
            "subject_consistency",
        ),
    ),
    "physvidbench": WorkspaceRunnerSpec(
        "physvidbench",
        "worldfoundry.evaluation.tasks.execution.runners.physvidbench.run_physvidbench_official_runner",
        generated_arg=None,
        limit_arg="--limit",
    ),
    "t2v-compbench": WorkspaceRunnerSpec(
        "t2v-compbench",
        "worldfoundry.evaluation.tasks.execution.runners.t2v_compbench.run_t2v_compbench_official_runner",
        generated_arg="--generated-video-dir",
        dataset_root_arg="--dataset-root",
        metric_arg="--category",
        model_arg="--model-name",
        supports_run_official=False,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        input_kind="generated_video_root_or_official_results",
    ),
    "t2v-safety-bench": WorkspaceRunnerSpec(
        "t2v-safety-bench",
        "worldfoundry.evaluation.tasks.execution.runners.t2v_safety_bench.run_t2v_safety_bench_official_runner",
        generated_arg="--generated-video-dir",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "t2vworldbench": WorkspaceRunnerSpec(
        "t2vworldbench",
        "worldfoundry.evaluation.tasks.execution.runners.t2vworldbench.run_t2vworldbench_official_runner",
        generated_arg="--generated-video-dir",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "video-bench": WorkspaceRunnerSpec(
        "video-bench",
        "worldfoundry.evaluation.tasks.execution.runners.videobench.run_videobench_official_runner",
        generated_arg="--generated-video-dir",
        dimension_arg="--dimension",
        supports_run_official=False,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "videophy": WorkspaceRunnerSpec(
        "videophy",
        "worldfoundry.evaluation.tasks.execution.runners.videophy.run_videophy_official_runner",
        generated_arg="--generated-video-dir",
        limit_arg="--limit",
        supports_fixture=True,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "videophy2": WorkspaceRunnerSpec(
        "videophy2",
        "worldfoundry.evaluation.tasks.execution.runners.videophy2.run_videophy2_official_runner",
        generated_arg="--generated-video-dir",
        limit_arg="--limit",
        supports_fixture=True,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "videoscience-bench": WorkspaceRunnerSpec(
        "videoscience-bench",
        "worldfoundry.evaluation.tasks.execution.runners.videoscience_bench.run_videoscience_bench_official_runner",
        generated_arg="--generated-video-dir",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "videoscore": WorkspaceRunnerSpec(
        "videoscore",
        "worldfoundry.evaluation.tasks.execution.runners.videoscore.run_videoscore_official_runner",
        generated_arg="--frames-dir",
        dataset_root_arg="--dataset-root",
        supports_run_official=False,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        input_kind="generated_frames_dir_or_official_results",
    ),
    "videoverse": WorkspaceRunnerSpec(
        "videoverse",
        "worldfoundry.evaluation.tasks.execution.runners.videoverse.run_videoverse_official_runner",
        generated_arg="--generated-artifact-dir",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "vmbench": WorkspaceRunnerSpec(
        "vmbench",
        "worldfoundry.evaluation.tasks.execution.runners.vmbench.run_vmbench_official_runner",
        generated_arg="--generated-video-dir",
        supports_fixture=True,
        default_metrics=("vmbench_average",),
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "wbench": WorkspaceRunnerSpec(
        "wbench",
        "worldfoundry.evaluation.tasks.execution.runners.wbench.run_wbench_official_runner",
        generated_arg="--generated-video-dir",
        dataset_root_arg="--dataset-root",
        metric_arg="--metrics",
        phase_arg="--phase",
        model_arg="--model-name",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "world-in-world": WorkspaceRunnerSpec(
        "world-in-world",
        "worldfoundry.evaluation.tasks.execution.runners.world_in_world.run_world_in_world_official_runner",
        generated_arg=None,
        limit_arg="--limit",
        supports_fixture=True,
    ),
    "worldarena": WorkspaceRunnerSpec(
        "worldarena",
        "worldfoundry.evaluation.tasks.execution.runners.worldarena.run_worldarena_official_runner",
        generated_arg=None,
        dimension_arg="--dimension",
    ),
    "worldbench": WorkspaceRunnerSpec(
        "worldbench",
        "worldfoundry.evaluation.tasks.execution.runners.worldbench.run_worldbench_official_runner",
        generated_arg="--generated-video-dir",
        dataset_root_arg="--dataset-root",
        dataset_manifest_arg="--video-manifest",
        answer_manifest_arg="--answer-manifest",
        limit_arg="--limit",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        default_metrics=("foreground_miou", "background_rmse"),
    ),
    "worldmodelbench": WorkspaceRunnerSpec(
        "worldmodelbench",
        "worldfoundry.evaluation.tasks.execution.runners.worldmodelbench.run_worldmodelbench_official_runner",
        generated_arg="--video-dir",
        dataset_root_arg="--data-root",
        supports_run_official=False,
        input_kind="generated_video_dir_or_official_results",
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "worldscore": WorkspaceRunnerSpec(
        "worldscore",
        "worldfoundry.evaluation.tasks.execution.runners.worldscore.run_worldscore_official_runner",
        generated_arg="--stage-dynamic-source",
        dataset_root_arg="--data-path",
        model_arg="--model-name",
        supports_run_official=False,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
        input_kind="worldscore_output_dir_or_official_results",
    ),
    "worldreasonbench": WorkspaceRunnerSpec(
        "worldreasonbench",
        "worldfoundry.evaluation.tasks.execution.runners.worldreasonbench.run_worldreasonbench_official_runner",
        generated_arg="--generated-artifact-dir",
        limit_arg="--limit",
        supports_fixture=True,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
    "wrbench": WorkspaceRunnerSpec(
        "wrbench",
        "worldfoundry.evaluation.tasks.execution.runners.wrbench.run_wrbench_official_runner",
        generated_arg="--generated-artifact-dir",
        limit_arg="--limit",
        supports_fixture=True,
        supports_official_runtime=True,
        accepts_generated_artifacts=True,
    ),
}


def benchmark_key(value: str | None) -> str:
    return str(value or "").strip().lower().replace("_", "-")


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


def workspace_benchmark_supported(benchmark_id: str | None) -> bool:
    key = benchmark_key(benchmark_id)
    return key in {"vbench", "vbench-2.0", "vbench-plus-plus"} or key in CLI_RUNNERS


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
    if run_status in {"failed", "error"} and scorecard.get("normalization_ok") and leaderboard:
        run_status = "normalized"
    output_dir_path = Path(output_dir)
    return {
        "schema_version": "worldfoundry-evaluate-run-result",
        "mode": "existing-results",
        "delegate_runner": delegate_runner,
        "status": run_status,
        "exit_code": int(run.get("returncode") or 0),
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
    )


def _run_cli_command(
    command: list[str],
    *,
    output_dir: str | Path,
    benchmark_id: str,
    delegate_runner: str,
    request: Mapping[str, Any],
    log_callback: Callable[[str, str], None] | None,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if log_callback is not None:
        log_callback("system", f"{benchmark_id} runner={' '.join(command)}\n")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir_path / "workspace_runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir_path / "workspace_runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    scorecard_path = output_dir_path / "scorecard.json"
    if not scorecard_path.is_file():
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-40:])
        raise RuntimeError(
            f"{benchmark_id} runner did not write scorecard.json; exit={completed.returncode}; tail={tail}"
        )
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if isinstance(scorecard, dict):
        scorecard.setdefault("run", {})["workspace_runner_returncode"] = completed.returncode
    result = _scorecard_to_workspace_result(
        scorecard,
        output_dir=output_dir_path,
        benchmark_id=benchmark_id,
        request=request,
        delegate_runner=delegate_runner,
    )
    if (
        completed.returncode != 0
        and not result.get("normalization_ok")
        and result.get("status") not in {"failed", "blocked"}
    ):
        result["status"] = "failed"
        result["exit_code"] = completed.returncode
    return result


def run_workspace_benchmark(
    payload: Any,
    output_dir: str | Path,
    *,
    log_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    benchmark_id = benchmark_key(_payload_get(payload, "benchmark_id"))
    if benchmark_id == "vbench":
        return _run_classic_vbench(payload, output_dir, log_callback)
    if benchmark_id in {"vbench-2.0", "vbench-plus-plus"}:
        return _run_vbench_series(payload, output_dir, log_callback)
    if benchmark_id in CLI_RUNNERS:
        return _run_cli_benchmark(payload, output_dir, log_callback)
    raise KeyError(f"unsupported Workspace benchmark runner: {benchmark_id}")


def validate_workspace_registry() -> list[str]:
    issues: list[str] = []
    try:
        from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
    except Exception as exc:  # pragma: no cover - defensive catalog validation
        VIDEO_RUNNER_REGISTRY = {}
        issues.append(f"framework runner registry unavailable: {type(exc).__name__}: {exc}")

    workspace_ids = set(CLI_RUNNERS) | {"vbench", "vbench-2.0", "vbench-plus-plus"}
    for benchmark_id, video_spec in sorted(VIDEO_RUNNER_REGISTRY.items()):
        if benchmark_id not in workspace_ids:
            issues.append(f"{benchmark_id}: missing Workspace benchmark runtime entry")
            continue
        expected_module = video_spec.script.removesuffix(".py").replace("/", ".")
        if benchmark_id in CLI_RUNNERS and CLI_RUNNERS[benchmark_id].module != expected_module:
            issues.append(
                f"{benchmark_id}: Workspace module {CLI_RUNNERS[benchmark_id].module} "
                f"does not match framework runner {expected_module}"
            )

    for benchmark_id, spec in sorted(CLI_RUNNERS.items()):
        module_path = REPO_ROOT / (spec.module.replace(".", "/") + ".py")
        if not module_path.is_file():
            issues.append(f"{benchmark_id}: runner module not found: {spec.module}")
    return issues
