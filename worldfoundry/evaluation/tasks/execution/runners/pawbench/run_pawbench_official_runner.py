#!/usr/bin/env python3
"""Run the in-tree PAWBench evaluator or normalize official ``metrics.json``.

The upstream evaluator is vendored read-only under ``runtime/pawbench``.
Official runs use caller-provided benchmark data, generated rollout videos,
and an OpenAI-compatible PAWEval judge. Every persistent upstream artifact is
written below ``--output-dir``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.io import scalar_number  # noqa: E402

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = RUNNER_ROOT / "runtime" / "pawbench"
METRIC_IDS = ("calibration_tvd_percent", "coverage_percent")
EXPECTED_SCENES_PER_TRACK = 25
TRACK_METRICS = {
    "calibration": "calibration_tvd_percent",
    "coverage": "coverage_percent",
}

CONFIG = ors.BenchRunnerConfig(
    benchmark_id="pawbench",
    display_name="PAWBench",
    root_env="WORLDFOUNDRY_PAWBENCH_ROOT",
    results_path_env="WORLDFOUNDRY_PAWBENCH_RESULTS_PATH",
    default_repo_subdir="worldfoundry/evaluation/tasks/execution/runners/pawbench/runtime/pawbench",
    metric_order=METRIC_IDS,
    metric_specs={
        "calibration_tvd_percent": {
            "name": "PAW-Calibration TVD",
            "group": "probabilistic_alignment",
            "higher_is_better": False,
        },
        "coverage_percent": {
            "name": "PAW-Coverage",
            "group": "probabilistic_alignment",
            "higher_is_better": True,
        },
    },
    metric_aliases={metric_id: metric_id for metric_id in METRIC_IDS},
    # PAWBench explicitly keeps its calibration and coverage tracks separate.
    average_metric_id="__no_derived_average__",
    official_entry="evaluate.py",
    official_output_globs=("upstream/metrics.json", "**/metrics.json"),
    usage_epilog=(
        "Normalize an official report:\n"
        "  python run_pawbench_official_runner.py --official-results-path <metrics.json-or-run-dir> "
        "--output-dir <out>\n\n"
        "Run the vendored PAWEval evaluator:\n"
        "  python run_pawbench_official_runner.py --run-official --dataset-root <PAWBench-data> "
        "--generated-artifact-dir <rollouts> --model-name <model> --output-dir <out>"
    ),
)


def _official_metrics_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("PAWBench metrics payload must be a JSON object")
    nested = payload.get("metrics")
    metrics = nested if isinstance(nested, Mapping) else payload
    if metrics.get("schema_version") != "pawbench.metrics/v1":
        raise ValueError("PAWBench metrics payload has an unsupported schema_version")
    if metrics.get("status") != "ok" or metrics.get("blockers"):
        raise ValueError("PAWBench metrics are blocked and cannot produce a valid scorecard")
    return metrics


def _single_model_result(track: str, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    tracks = payload.get("tracks")
    track_payload = tracks.get(track) if isinstance(tracks, Mapping) else None
    if not isinstance(track_payload, Mapping) or track_payload.get("status") != "ok" or track_payload.get("blockers"):
        raise ValueError(f"PAWBench {track} track is missing or blocked")
    models = track_payload.get("models")
    if not isinstance(models, Mapping) or len(models) != 1:
        raise ValueError(f"PAWBench {track} track must contain exactly one evaluated model")
    model_name, result = next(iter(models.items()))
    if (
        not isinstance(model_name, str)
        or not isinstance(result, Mapping)
        or result.get("status") != "ok"
        or result.get("blockers")
    ):
        raise ValueError(f"PAWBench {track} model result is missing or blocked")
    return model_name, result


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    """Extract the two official PAWBench track averages without combining them."""
    official = _official_metrics_payload(payload)
    extracted: dict[str, dict[str, Any]] = {}
    selected_model: str | None = None
    for track, metric_id in TRACK_METRICS.items():
        model_name, result = _single_model_result(track, official)
        if selected_model is not None and model_name != selected_model:
            raise ValueError("PAWBench tracks refer to different evaluated models")
        selected_model = model_name
        average = result.get("track_average")
        if not isinstance(average, Mapping) or average.get("name") != metric_id:
            raise ValueError(f"PAWBench {track} track_average is missing {metric_id}")
        raw_score = scalar_number(average.get("value"))
        if raw_score is None or not 0.0 <= raw_score <= 100.0:
            raise ValueError(f"PAWBench {metric_id} must be a percentage in [0, 100]")
        pass_rate = result.get("scene_pass_rate")
        passing_scenes = scalar_number(pass_rate.get("passing_scenes") if isinstance(pass_rate, Mapping) else None)
        sample_count = scalar_number(pass_rate.get("scene_denominator") if isinstance(pass_rate, Mapping) else None)
        pass_rate_value = scalar_number(pass_rate.get("value") if isinstance(pass_rate, Mapping) else None)
        if (
            passing_scenes != EXPECTED_SCENES_PER_TRACK
            or sample_count != EXPECTED_SCENES_PER_TRACK
            or pass_rate_value != 100.0
        ):
            raise ValueError(f"PAWBench {track} track is incomplete; all {EXPECTED_SCENES_PER_TRACK} scenes must pass")
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_score,
            "normalized_score": raw_score / 100.0,
            "source": f"{results_path}#tracks.{track}.models.{model_name}.track_average",
            "sample_count": EXPECTED_SCENES_PER_TRACK,
            "model_name": model_name,
        }
    return extracted


def prepare_upstream_results(
    config: ors.BenchRunnerConfig,
    results_path: Path,
    args: Any,
    output_dir: Path,
) -> Path:
    """Resolve an official PAWBench report directory to ``metrics.json``."""
    del config, args, output_dir
    resolved = results_path.expanduser().resolve()
    if resolved.is_dir():
        direct = resolved / "metrics.json"
        if direct.is_file():
            return direct
        matches = sorted(resolved.rglob("metrics.json"))
        if matches:
            return matches[-1]
    if resolved.name in {"run.json", "rows.jsonl", "checkpoint.jsonl"}:
        sibling = resolved.with_name("metrics.json")
        if sibling.is_file():
            return sibling
    return resolved


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    del repo_root
    return ors.discover_by_globs([output_dir], ("upstream/metrics.json", "**/metrics.json"))


def build_official_command(
    *,
    config: ors.BenchRunnerConfig,
    repo_root: Path,
    generated_video_dir: Path,
    output_dir: Path,
    args: Any,
) -> list[str] | None:
    del config
    data_root = args.dataset_root or os.environ.get("WORLDFOUNDRY_PAWBENCH_DATA_ROOT")
    if not data_root:
        raise ValueError("--dataset-root or WORLDFOUNDRY_PAWBENCH_DATA_ROOT is required")
    data_path = Path(data_root).expanduser().resolve()
    if not (data_path / "manifest.json").is_file():
        raise FileNotFoundError(f"PAWBench manifest.json not found below dataset root: {data_path}")
    rollouts = generated_video_dir.expanduser().resolve()
    if not rollouts.is_dir():
        raise FileNotFoundError(f"PAWBench rollout directory not found: {rollouts}")
    evaluator = repo_root / "evaluate.py"
    if not evaluator.is_file():
        # Catalog orchestration binds the runner root to ``root_env`` while
        # standalone use resolves the vendored runtime directly. Accept both
        # shapes so the same wrapper works through either public surface.
        evaluator = repo_root / "runtime" / "pawbench" / "evaluate.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"vendored PAWBench evaluator not found: {evaluator}")
    if not os.environ.get(args.vlm_api_key_env):
        raise ValueError(f"set {args.vlm_api_key_env} for the PAWEval judge")

    return [
        args.python,
        str(evaluator),
        "--benchmark",
        str(data_path),
        "--videos",
        str(rollouts),
        "--output",
        str((output_dir / "upstream").resolve()),
        "--model",
        args.model_name or rollouts.name or "world-model",
        "--vlm-base-url",
        args.vlm_base_url,
        "--vlm-model",
        args.vlm_model,
        "--vlm-api-key-env",
        args.vlm_api_key_env,
    ]


def build_official_environment(
    *,
    config: ors.BenchRunnerConfig,
    repo_root: Path,
    generated_video_dir: Path,
    output_dir: Path,
    args: Any,
) -> Mapping[str, str]:
    del config, repo_root, generated_video_dir, args
    temporary_root = (output_dir / "upstream_tmp").resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    return {"TMPDIR": str(temporary_root)}


def extend_parser(parser) -> None:
    parser.add_argument(
        "--generated-artifact-dir",
        dest="generated_video_dir",
        type=Path,
        help="Alias for --generated-video-dir used by the unified WorldFoundry CLI.",
    )
    parser.add_argument("--dataset-root", type=Path, help="Downloaded Andrew613/PAWBench dataset root.")
    parser.add_argument("--model-name", help="Video model name recorded in official PAWBench outputs.")
    parser.add_argument(
        "--vlm-base-url",
        default=os.environ.get("WORLDFOUNDRY_PAWBENCH_VLM_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenAI-compatible PAWEval endpoint.",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("WORLDFOUNDRY_PAWBENCH_JUDGE_MODEL", "google/gemini-3.5-flash"),
        help="PAWEval judge model; official runs use google/gemini-3.5-flash.",
    )
    parser.add_argument(
        "--vlm-api-key-env",
        default=os.environ.get("WORLDFOUNDRY_PAWBENCH_VLM_API_KEY_ENV", "OPENROUTER_API_KEY"),
        help="Name of the environment variable carrying the endpoint API key.",
    )


def main(argv: list[str] | None = None) -> int:
    return ors.run_main(
        CONFIG,
        ors.RunnerHooks(
            build_official_command=build_official_command,
            build_official_environment=build_official_environment,
            discover_official_results=discover_official_results,
            extract_metrics=extract_metrics,
            prepare_upstream_results=prepare_upstream_results,
            extend_parser=extend_parser,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
