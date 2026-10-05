#!/usr/bin/env python3
"""Official runner for PhyFPS-Bench-Gen (Visual Chronometer temporal consistency benchmark)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    resolve_env_path,
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)
from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_metrics import (
    METRIC_ORDER,
    METRIC_SPECS,
    compute_phyfps_metrics,
)
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_predict import (
    PhyFPSPredictConfig,
    load_meta_fps_map,
    parse_results_csv,
    run_phyfps_predict,
)
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_prompts import (
    CANONICAL_PROMPT_COUNT,
    PROMPT_MANIFEST_REL,
    load_prompt_lines,
    video_filename_for_index,
)
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.visual_chronometer_runtime import (
    resolve_chronometer_root,
)
from worldfoundry.evaluation.utils import REPO_ROOT

BENCHMARK_ID = "phyfps-bench-gen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize PhyFPS-Bench-Gen official outputs.")
    parser.add_argument("--benchmark-id", default="phyfps-bench-gen")
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--from-upstream-results", dest="official_results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-official", action="store_true", help="Execute Visual Chronometer PhyFPS prediction in-tree.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-artifact-dir", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--visual-chronometer-root", type=Path)
    parser.add_argument("--phyfps-bench-gen-root", type=Path)
    parser.add_argument("--meta-fps", type=float)
    parser.add_argument("--meta-fps-manifest", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=30)
    parser.add_argument("--device", default=os.environ.get("WORLDFOUNDRY_PHYFPS_DEVICE", "cuda:0"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)




def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None



def _resolve_phyfps_bench_gen_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        resolve_env_path("WORLDFOUNDRY_PHYFPS_BENCH_GEN_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def _default_prompt_manifest() -> Path | None:
    env_manifest = resolve_env_path("WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST")
    if env_manifest is not None and env_manifest.is_file():
        return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    root = _resolve_phyfps_bench_gen_root()
    if root is None:
        return None
    candidate = root / PROMPT_MANIFEST_REL
    return candidate if candidate.is_file() else None


def _prompt_manifest_from_root(root: Path | None) -> Path | None:
    if root is None:
        return None
    candidate = root.expanduser().resolve() / PROMPT_MANIFEST_REL
    return candidate if candidate.is_file() else None


def _video_index(generated_dir: Path | None) -> dict[str, Any]:
    if generated_dir is None:
        return {
            "provided": False,
            "generated_artifact_dir": None,
            "video_count": 0,
            "by_stem": {},
            "duplicate_stems": [],
        }
    by_stem: dict[str, list[str]] = {}
    if generated_dir.exists():
        for path in sorted(generated_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                by_stem.setdefault(path.stem, []).append(str(path))
    return {
        "provided": True,
        "generated_artifact_dir": str(generated_dir),
        "video_count": sum(len(paths) for paths in by_stem.values()),
        "by_stem": dict(by_stem),
        "duplicate_stems": sorted(stem for stem, paths in by_stem.items() if len(paths) > 1),
    }


def _coverage(expected_names: set[str], actual_names: set[str]) -> dict[str, Any]:
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    matched = sorted(expected_names & actual_names)
    return {
        "expected_count": len(expected_names),
        "actual_count": len(actual_names),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "complete": bool(expected_names) and not missing and not unexpected,
        "missing_ids": missing[:50],
        "unexpected_ids": unexpected[:50],
    }


def _manifest_stats(prompt_count: int) -> dict[str, Any]:
    return {
        "prompt_count": prompt_count,
        "canonical_suite": prompt_count == CANONICAL_PROMPT_COUNT,
    }


def _metric_rows(
    *,
    computed: Mapping[str, Any],
    source_path: Path,
    official_runtime_executed: bool,
) -> list[dict[str, Any]]:
    direct_metrics = computed.get("metrics") if isinstance(computed.get("metrics"), Mapping) else {}
    components = computed.get("components") if isinstance(computed.get("components"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = direct_metrics.get(metric_id)
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "source": "phyfps_official_runtime" if official_runtime_executed else "phyfps_results_csv",
                "source_path": str(source_path),
                "components": components,
                "reason": None if score is not None else "score_not_available_in_phyfps_results",
            }
        )
    return rows


def _scorecard(
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    prompt_manifest_path: Path | None,
    manifest_stats: Mapping[str, Any],
    video_coverage: Mapping[str, Any],
    metric_rows: list[dict[str, Any]],
    predict_summary: Mapping[str, Any] | None,
    official_runtime_executed: bool,
    meta_fps_coverage_count: int,
) -> dict[str, Any]:
    available_rows = [row for row in metric_rows if row.get("available") is True]
    per_metric = {str(row["metric_id"]): row for row in metric_rows}
    leaderboard = {
        str(row["metric_id"]): row["normalized_score"]
        for row in available_rows
        if row.get("normalized_score") is not None
    }
    video_complete = (
        video_coverage.get("provided") is True
        and video_coverage.get("complete") is True
        and not video_coverage.get("duplicate_stems")
    )
    full_suite_valid = (
        manifest_stats.get("canonical_suite") is True
        and video_complete
        and len(available_rows) == len(METRIC_ORDER)
        and meta_fps_coverage_count > 0
    )
    normalization_ok = bool(available_rows)
    predict_backend = str((predict_summary or {}).get("backend") or "").lower()
    official_runtime_real = official_runtime_executed and predict_backend != "mock"
    official_verified = official_runtime_real and normalization_ok
    integration_evidence = official_verified and full_suite_valid
    normalizer_only = not official_runtime_executed and not integration_evidence
    evaluation_kind = (
        "phyfps_bench_gen_official_in_tree"
        if official_runtime_executed
        else "phyfps_bench_gen_result_normalizer"
    )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalizer_only": normalizer_only,
        "normalization_ok": normalization_ok,
        "official_results_imported": bool(available_rows) and not official_runtime_executed,
        "run": {
            "status": (
                "official_verified"
                if official_verified and integration_evidence
                else "official_results_normalized"
                if available_rows
                else "official_results_missing_scores"
            ),
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_phyfps_bench_gen_official_runner",
            "command": None if predict_summary is None else predict_summary.get("command"),
            "returncode": 0 if available_rows else 1,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "PhyFPS-Bench-Gen",
            "contract_only": False,
            "requires_upstream_runtime": not official_runtime_real,
            "official_runtime_available": official_runtime_real,
            "official_judge": "Visual Chronometer FPSPredictor",
        },
        "dataset": {
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "official_results_path": str(official_results_path),
            "manifest_stats": dict(manifest_stats),
            "video_coverage": dict(video_coverage),
            "meta_fps_coverage_count": meta_fps_coverage_count,
        },
        "eligibility": {
            "canonical_suite": manifest_stats.get("canonical_suite") is True,
            "full_suite_valid": full_suite_valid,
            "leaderboard_valid": False,
            "reasons": [
                "PhyFPS-Bench-Gen requires Meta FPS values for alignment metrics; provide WORLDFOUNDRY_PHYFPS_META_FPS or a manifest.",
                "Full-suite integration evidence requires 100 generated videos and complete Visual Chronometer PhyFPS outputs.",
            ],
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "alignment": ["avg_error_fps", "pct_error"],
                "consistency": ["inter_video_cv", "intra_video_cv"],
                "aggregate": ["phyfps_bench_gen_average"],
            },
            "per_metric": per_metric,
            "summary": {
                "available_metric_count": len(available_rows),
                "blocked_metric_count": len(metric_rows) - len(available_rows),
                "video_count": video_coverage.get("actual_count"),
                "expected_video_count": video_coverage.get("expected_count"),
            },
        },
        "evaluation": {
            "available": bool(available_rows),
            "kind": evaluation_kind,
            "evidence_level": (
                "official_runtime_executed"
                if official_runtime_real
                else "mock_runtime_executed"
                if predict_backend == "mock"
                else "official_results_normalized"
            ),
            "num_results": len(metric_rows),
            "skip_count": len(metric_rows) - len(available_rows),
            "blocked_count": len(metric_rows) - len(available_rows),
            "predict_summary": dict(predict_summary or {}),
        },
        "artifacts": {
            "scorecard": str((output_dir / "scorecard.json").resolve()),
            "raw_metric_table": str((output_dir / "raw_metric_table.jsonl").resolve()),
            "benchmark_contract": str((output_dir / "benchmark_contract.json").resolve()),
            "per_sample_scores": str((output_dir / "per_sample_scores.jsonl").resolve()),
            "results_csv": str(official_results_path.resolve()),
        },
    }


def normalize_phyfps_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    predict_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    official_results_path = args.official_results_path or resolve_env_path("WORLDFOUNDRY_PHYFPS_BENCH_GEN_RESULTS_PATH")
    if official_results_path is None:
        official_results_path = output_dir / "results.csv"
    prompt_manifest_path = (
        args.prompt_manifest
        or _prompt_manifest_from_root(args.phyfps_bench_gen_root)
        or resolve_env_path("WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST")
        or _default_prompt_manifest()
    )
    generated_artifact_dir = args.generated_artifact_dir or resolve_env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")

    prompt_count = 0
    expected_names: set[str] = set()
    if prompt_manifest_path is not None and Path(prompt_manifest_path).is_file():
        prompts = load_prompt_lines(prompt_manifest_path)
        if args.limit is not None:
            prompts = prompts[: int(args.limit)]
        prompt_count = len(prompts)
        expected_names = {Path(video_filename_for_index(index)).stem for index in range(len(prompts))}

    manifest_stats = _manifest_stats(prompt_count)
    video_index = _video_index(generated_artifact_dir)
    actual_names = set(video_index["by_stem"].keys()) if video_index["provided"] else set()
    if expected_names:
        video_coverage = _coverage(expected_names, actual_names)
    else:
        video_coverage = {
            "expected_count": 0,
            "actual_count": len(actual_names),
            "matched_count": len(actual_names),
            "missing_count": 0,
            "unexpected_count": 0,
            "complete": bool(actual_names),
            "missing_ids": [],
            "unexpected_ids": sorted(actual_names)[:50],
        }
    video_coverage.update(
        {
            "provided": video_index["provided"],
            "generated_artifact_dir": video_index["generated_artifact_dir"],
            "video_count": video_index["video_count"],
            "duplicate_stems": video_index["duplicate_stems"],
        }
    )

    records = parse_results_csv(Path(official_results_path))
    meta_fps_map = load_meta_fps_map(
        explicit_manifest=args.meta_fps_manifest or resolve_env_path("WORLDFOUNDRY_PHYFPS_META_FPS_MANIFEST"),
        default_meta_fps=args.meta_fps or _env_float("WORLDFOUNDRY_PHYFPS_META_FPS"),
        video_names=[record.video for record in records],
    )
    computed = compute_phyfps_metrics(video_records=records, meta_fps_by_video=meta_fps_map)
    rows = _metric_rows(
        computed=computed,
        source_path=Path(official_results_path),
        official_runtime_executed=official_runtime_executed,
    )
    scorecard = _scorecard(
        benchmark_id=str(args.benchmark_id),
        output_dir=output_dir,
        official_results_path=Path(official_results_path),
        prompt_manifest_path=prompt_manifest_path if prompt_manifest_path and Path(prompt_manifest_path).is_file() else None,
        manifest_stats=manifest_stats,
        video_coverage=video_coverage,
        metric_rows=rows,
        predict_summary=predict_summary,
        official_runtime_executed=official_runtime_executed,
        meta_fps_coverage_count=int(computed.get("components", {}).get("meta_fps_coverage_count", 0)),
    )
    strict = args.strict or str(os.environ.get("WORLDFOUNDRY_PHYFPS_BENCH_GEN_STRICT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not scorecard["eligibility"]["full_suite_valid"]:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1

    write_jsonl(output_dir / "raw_metric_table.jsonl", rows)
    write_jsonl(
        output_dir / "per_sample_scores.jsonl",
        [
            {
                "video": record.video,
                "avg_phyfps": record.avg_phyfps,
                "segment_count": len(record.segment_phyfps),
            }
            for record in records
        ],
    )
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "prompt_manifest": None if prompt_manifest_path is None else str(prompt_manifest_path),
            "official_results_path": str(official_results_path),
            "metric_ids": list(METRIC_ORDER),
            "canonical_suite": manifest_stats["canonical_suite"],
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def run_official_phyfps_bench_gen(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_artifact_dir = args.generated_artifact_dir or resolve_env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if generated_artifact_dir is None:
        raise ValueError("--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official")

    backend = (os.environ.get("WORLDFOUNDRY_PHYFPS_PREDICT_BACKEND") or "official").lower()
    config = PhyFPSPredictConfig(
        backend=backend,
        stride=args.stride,
        clip_length=args.clip_length,
        device=args.device,
        python_executable=args.python,
        chronometer_root=resolve_chronometer_root(args.visual_chronometer_root),
        timeout_seconds=args.timeout,
    )
    results_csv = output_dir / "results.csv"
    predict_summary = run_phyfps_predict(
        config=config,
        video_dir=Path(generated_artifact_dir),
        output_csv=results_csv,
        stdout_path=output_dir / "predict_stdout.log",
        stderr_path=output_dir / "predict_stderr.log",
    )
    normalize_args = argparse.Namespace(
        benchmark_id=args.benchmark_id,
        official_results_path=results_csv,
        output_dir=output_dir,
        generated_artifact_dir=generated_artifact_dir,
        prompt_manifest=args.prompt_manifest,
        visual_chronometer_root=args.visual_chronometer_root,
        phyfps_bench_gen_root=args.phyfps_bench_gen_root,
        meta_fps=args.meta_fps,
        meta_fps_manifest=args.meta_fps_manifest,
        stride=args.stride,
        clip_length=args.clip_length,
        device=args.device,
        python=args.python,
        timeout=args.timeout,
        limit=args.limit,
        strict=args.strict,
        json=False,
    )
    return normalize_phyfps_results(
        normalize_args,
        official_runtime_executed=True,
        predict_summary=predict_summary,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_official:
            scorecard = run_official_phyfps_bench_gen(args)
        else:
            scorecard = normalize_phyfps_results(args)
    except Exception as exc:  # noqa: BLE001
        args.output_dir.mkdir(parents=True, exist_ok=True)
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
                "runner": "benchmark_zoo_phyfps_bench_gen_official_runner",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "PhyFPS-Bench-Gen"},
            "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"available_metric_count": 0}},
            "evaluation": {"available": False, "kind": "phyfps_bench_gen_result_normalizer", "blocked_count": len(METRIC_ORDER)},
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", scorecard)
        if args.json:
            print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
        return 1
    if args.json:
        print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
    return int(scorecard.get("run", {}).get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
