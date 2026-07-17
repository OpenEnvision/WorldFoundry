#!/usr/bin/env python3
"""Official runner for FETV-EVAL.

FETV combines human ratings (static/temporal quality, alignment) with automatic
CLIP/BLIP scores and FID/FVD distance metrics. When ``--official-results-path``
points at an official FETV-EVAL output directory (``manual_eval_results/``,
``auto_eval_results/``), this runner materializes a CSV before normalization.

Environment variables
---------------------
- ``WORLDFOUNDRY_FETV_RUNTIME_ROOT``: optional override for the vendored FETV-EVAL runtime root
- ``WORLDFOUNDRY_FETV_RESULTS_PATH``: CSV/JSON export or eval output directory
- ``WORLDFOUNDRY_FETV_MODEL_NAME``: model alias inside FETV directory layout
- ``WORLDFOUNDRY_GENERATED_ARTIFACT_DIR``: generated videos for contract checks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric, scalar_number
from worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_official_runtime import (
    OFFICIAL_RUNTIME_ROOT,
    normalize_metric_tokens,
    resolve_fetv_runtime_root,
)

FETV_DEFAULT_MODEL_NAME = "modelscope-t2v"
FETV_MODEL_ALIASES = {
    "modelscope": ("modelscope-t2v", "damo-text2video"),
    "modelscope-t2v": ("modelscope-t2v", "damo-text2video"),
    "damo-text2video": ("modelscope-t2v", "damo-text2video"),
    "text2video-zero": ("text2video-zero", "text2video-zero"),
    "text2video_zero": ("text2video-zero", "text2video-zero"),
    "cogvideo": ("cogvideo", "cogvideo"),
    "zeroscope": ("zeroscope", "zeroscope"),
    "ground-truth": ("ground-truth", "ground-truth"),
    "ground_truth": ("ground-truth", "ground-truth"),
}
FETV_MANUAL_METRIC_FIELDS = {
    "static_quality": "static_quality",
    "temporal_quality": "temporal_quality",
    "alignment": "overall_alignment",
}
FETV_AUTO_METRIC_DIRS = {
    "clip_score": "CLIPScore",
    "blip_score": "BLIPScore",
}
FETV_DISTANCE_METRICS = {
    "fid": ("fid_results", "metric-fid1024_16f.jsonl", "fid1024_16f"),
    "fvd": ("fvd_results", "metric-fvd1024_16f.jsonl", "fvd1024_16f"),
}

CONFIG = ors.build_runner_config_from_contract(
    "fetv",
    root_env="WORLDFOUNDRY_FETV_RUNTIME_ROOT",
    results_path_env="WORLDFOUNDRY_FETV_RESULTS_PATH",
    default_repo_subdir="worldfoundry/evaluation/tasks/execution/runners/fetv/runtime/fetv_eval",
    official_output_globs=("fetv_eval_results_*.csv", "fetv_results*.csv", "fetv_results*.json"),
    official_entry="worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_official_runtime",
    lower_is_better=frozenset({"fid", "fvd"}),
)


def _fetv_model_names(model_name: str | None) -> tuple[str, str]:
    key = (model_name or FETV_DEFAULT_MODEL_NAME).strip().lower()
    return FETV_MODEL_ALIASES.get(key, (key, key))


def _load_fetv_manual_metric_values(path: Path, metrics: dict[str, list[float]]) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        for item in payload.values():
            if not isinstance(item, dict):
                continue
            for source_field, metric_id in FETV_MANUAL_METRIC_FIELDS.items():
                value = scalar_number(item.get(source_field))
                if value is not None:
                    metrics.setdefault(metric_id, []).append(value)
            fine_grained = item.get("fine-grained_alignment")
            if isinstance(fine_grained, dict):
                for value in fine_grained.values():
                    numeric = scalar_number(value)
                    if numeric is not None:
                        metrics.setdefault("fine_grained_alignment", []).append(numeric)


def _load_fetv_auto_metric_values(root: Path, auto_model_name: str, metrics: dict[str, list[float]]) -> None:
    for metric_id, dirname in FETV_AUTO_METRIC_DIRS.items():
        path = root / "auto_eval_results" / dirname / f"auto_eval_results_{auto_model_name}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for value in payload.values():
            numeric = scalar_number(value)
            if numeric is not None:
                metrics.setdefault(metric_id, []).append(numeric)


def _load_fetv_distance_metric_values(root: Path, auto_model_name: str, metrics: dict[str, list[float]]) -> None:
    for metric_id, (dirname, filename, result_key) in FETV_DISTANCE_METRICS.items():
        metric_root = root / "auto_eval_results" / dirname / auto_model_name
        if not metric_root.is_dir():
            continue
        for path in sorted(metric_root.glob(f"*/{filename}")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                results = payload.get("results")
                if not isinstance(results, dict):
                    continue
                value = scalar_number(results.get(result_key))
                if value is not None:
                    metrics.setdefault(metric_id, []).append(value)


def materialize_fetv_eval_directory_results(*, root: Path, output_dir: Path, model_name: str | None) -> Path:
    auto_model_name, manual_model_name = _fetv_model_names(model_name)
    metrics: dict[str, list[float]] = {}
    manual_root = root / "manual_eval_results"
    if manual_root.is_dir():
        for human_dir in sorted(manual_root.glob("human*")):
            path = human_dir / f"manual_eval_results_{manual_model_name}.json"
            if path.is_file():
                _load_fetv_manual_metric_values(path, metrics)

    _load_fetv_auto_metric_values(root, auto_model_name, metrics)
    _load_fetv_distance_metric_values(root, auto_model_name, metrics)

    rows = [
        {
            "metric_id": metric_id,
            "score": score,
            "source": "fetv_eval_directory",
            "model_name": auto_model_name,
            "record_count": len(values),
        }
        for metric_id, values in sorted(metrics.items())
        for score in [mean_numeric(values)]
        if score is not None
    ]
    if not rows:
        raise ValueError(
            f"no FETV-EVAL result files found under {root} for model {auto_model_name!r}/{manual_model_name!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"fetv_eval_results_{auto_model_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric_id", "score", "source", "model_name", "record_count"))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    return ors.extract_tabular_official_metrics(payload, results_path, CONFIG)


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir]
    if repo_root is not None:
        search_roots.append(repo_root)
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)


def prepare_upstream_results(
    config: ors.BenchRunnerConfig,
    results_path: Path,
    args: Any,
    output_dir: Path,
) -> Path:
    del config
    if not results_path.is_dir():
        return results_path
    return materialize_fetv_eval_directory_results(
        root=results_path,
        output_dir=output_dir,
        model_name=getattr(args, "fetv_model_name", None),
    )


def extend_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fetv-model-name", default=os.environ.get("WORLDFOUNDRY_FETV_MODEL_NAME"))
    parser.add_argument(
        "--fetv-runtime-backend",
        choices=("official", "artifact"),
        default=os.environ.get("WORLDFOUNDRY_FETV_RUNTIME_BACKEND", "official"),
        help="Use the vendored FETV-EVAL runtime or the WorldFoundry artifact-score fallback.",
    )
    parser.add_argument(
        "--fetv-runtime-root",
        type=Path,
        default=ors.env_path("WORLDFOUNDRY_FETV_RUNTIME_ROOT", OFFICIAL_RUNTIME_ROOT),
        help="Optional override for the vendored FETV-EVAL runtime root.",
    )
    parser.add_argument(
        "--fetv-metrics",
        nargs="+",
        default=None,
        help="FETV official metric families to run: clip_score, blip_score, fid, fvd, or all.",
    )
    parser.add_argument("--fetv-prompt-file", type=Path, default=ors.env_path("WORLDFOUNDRY_FETV_PROMPT_FILE"))
    parser.add_argument("--fetv-clip-model", default=os.environ.get("WORLDFOUNDRY_FETV_CLIP_MODEL", "ViT-B/32"))
    parser.add_argument(
        "--fetv-is-clip-ft",
        action="store_true",
        default=os.environ.get("WORLDFOUNDRY_FETV_IS_CLIP_FT") == "1",
    )
    parser.add_argument("--fetv-blip-config", type=Path, default=ors.env_path("WORLDFOUNDRY_FETV_BLIP_CONFIG"))
    parser.add_argument(
        "--fetv-fid-generated-video-dir",
        type=Path,
        default=ors.env_path("WORLDFOUNDRY_FETV_FID_GENERATED_VIDEO_DIR"),
    )
    parser.add_argument(
        "--fetv-fid-real-video-dir",
        type=Path,
        default=ors.env_path("WORLDFOUNDRY_FETV_FID_REAL_VIDEO_DIR"),
    )
    parser.add_argument(
        "--fetv-fvd-generated-video-dir",
        type=Path,
        default=ors.env_path("WORLDFOUNDRY_FETV_FVD_GENERATED_VIDEO_DIR"),
    )
    parser.add_argument(
        "--fetv-fvd-real-video-dir",
        type=Path,
        default=ors.env_path("WORLDFOUNDRY_FETV_FVD_REAL_VIDEO_DIR"),
    )
    parser.add_argument("--fetv-seeds", default=os.environ.get("WORLDFOUNDRY_FETV_SEEDS"))
    parser.add_argument("--fetv-cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--fetv-fvd-gpus", default=os.environ.get("WORLDFOUNDRY_FETV_FVD_GPUS"))
    parser.add_argument("--fetv-fvd-resolution", default=os.environ.get("WORLDFOUNDRY_FETV_FVD_RESOLUTION"))
    parser.add_argument("--fetv-max-frame-num", default=os.environ.get("WORLDFOUNDRY_FETV_MAX_FRAME_NUM"))
    parser.add_argument(
        "--fetv-limit",
        type=int,
        default=os.environ.get("WORLDFOUNDRY_FETV_LIMIT"),
        help="Evaluate an exact zero-based prompt prefix; bounded runs are never leaderboard-valid.",
    )
    parser.add_argument(
        "--artifact-score-dir",
        type=Path,
        help="Directory containing in-tree FETV metric artifacts produced by WorldFoundry base-model evaluators.",
    )


def build_official_command(*, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any) -> list[str] | None:
    if args.fetv_runtime_backend == "official":
        runtime_root = resolve_fetv_runtime_root(args.fetv_runtime_root)
        command = [
            args.python,
            str(Path(__file__).resolve().with_name("fetv_official_runtime.py")),
            "--generated-video-dir",
            str(generated_video_dir),
            "--output-dir",
            str(output_dir),
            "--runtime-root",
            str(runtime_root),
            "--model-name",
            args.fetv_model_name or FETV_DEFAULT_MODEL_NAME,
            "--clip-model",
            args.fetv_clip_model,
            "--python",
            args.python,
            "--timeout",
            str(args.timeout),
        ]
        if args.fetv_metrics:
            normalized_metrics = normalize_metric_tokens(args.fetv_metrics)
            command.extend(["--metrics", *normalized_metrics])
        if args.fetv_prompt_file is not None:
            command.extend(["--prompt-file", str(args.fetv_prompt_file)])
        if args.fetv_is_clip_ft:
            command.append("--is-clip-ft")
        if args.fetv_blip_config is not None:
            command.extend(["--blip-config", str(args.fetv_blip_config)])
        if args.fetv_fid_generated_video_dir is not None:
            command.extend(["--fid-generated-video-dir", str(args.fetv_fid_generated_video_dir)])
        if args.fetv_fid_real_video_dir is not None:
            command.extend(["--fid-real-video-dir", str(args.fetv_fid_real_video_dir)])
        if args.fetv_fvd_generated_video_dir is not None:
            command.extend(["--fvd-generated-video-dir", str(args.fetv_fvd_generated_video_dir)])
        if args.fetv_fvd_real_video_dir is not None:
            command.extend(["--fvd-real-video-dir", str(args.fetv_fvd_real_video_dir)])
        if args.fetv_seeds:
            command.extend(["--seeds", args.fetv_seeds])
        if args.fetv_cuda_visible_devices:
            command.extend(["--cuda-visible-devices", args.fetv_cuda_visible_devices])
        if args.fetv_fvd_gpus:
            command.extend(["--fvd-gpus", str(args.fetv_fvd_gpus)])
        if args.fetv_fvd_resolution:
            command.extend(["--fvd-resolution", str(args.fetv_fvd_resolution)])
        if args.fetv_max_frame_num:
            command.extend(["--max-frame-num", str(args.fetv_max_frame_num)])
        if args.fetv_limit is not None:
            command.extend(["--limit", str(args.fetv_limit)])
        return command

    score_dir = args.artifact_score_dir or ors.env_path("WORLDFOUNDRY_FETV_ARTIFACT_SCORE_DIR")
    if score_dir is None:
        score_dir = generated_video_dir
    upstream_results = output_dir / "fetv_results_artifact_scores.json"
    return [
        args.python,
        "-m",
        "worldfoundry.evaluation.tasks.execution.framework.artifact_score_runtime",
        "--benchmark-id",
        config.benchmark_id,
        "--score-dir",
        str(score_dir),
        "--generated-video-dir",
        str(generated_video_dir),
        "--output-path",
        str(upstream_results),
    ]


HOOKS = ors.RunnerHooks(
    build_official_command=build_official_command,
    extract_metrics=extract_metrics,
    discover_official_results=discover_official_results,
    prepare_upstream_results=prepare_upstream_results,
    extend_parser=extend_parser,
)


def main(argv: list[str] | None = None) -> int:
    return ors.run_main(CONFIG, HOOKS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
