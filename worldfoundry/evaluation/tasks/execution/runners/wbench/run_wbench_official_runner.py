#!/usr/bin/env python3
"""Official WBench runner and result normalizer."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.core.time import utc_now_iso  # noqa: E402
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, write_json  # noqa: E402

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_WBENCH_ROOT = RUNNER_ROOT / "runtime" / "wbench"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
OFFICIAL_VIDEO_RE = re.compile(r"^case_(?P<case_id>\d+)_combined\.mp4$", re.IGNORECASE)
MATERIALIZED_VIDEO_RE = re.compile(
    r"^case_(?P<case_id>\d+)(?:__generated_video)?\.mp4$",
    re.IGNORECASE,
)
SUBMISSION_PROTOCOLS = (
    "unverified",
    "text_multi_turn",
    "camera_conditioned",
    "action_conditioned",
)

METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "quality": (
        "aesthetic_quality",
        "imaging_quality",
        "temporal_flickering",
        "dynamic_degree",
        "motion_smoothness",
        "hpsv3_quality",
    ),
    "setting": ("scene_adherence", "subject_adherence"),
    "interaction": (
        "navigation_trajectory",
        "event_edit_adherence",
        "subject_action_adherence",
        "perspective_switch_adherence",
    ),
    "consistency": (
        "background_consistency",
        "segment_continuity",
        "perspective_consistency",
        "subject_consistency",
        "geometric_consistency",
        "photometric_consistency",
        "spatial_consistency",
        "gated_spatial_consistency",
    ),
    "physical": ("visual_plausibility", "causal_fidelity"),
}
DIMENSION_METRICS = tuple(f"{group}_score" for group in METRIC_GROUPS)
METRIC_ORDER = tuple(metric for group in METRIC_GROUPS.values() for metric in group) + DIMENSION_METRICS + (
    "wbench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    metric_id: {
        "name": metric_id.replace("_", " ").title(),
        "higher_is_better": True,
        "group": group,
    }
    for group, metric_ids in METRIC_GROUPS.items()
    for metric_id in metric_ids
}
METRIC_SPECS.update(
    {
        metric_id: {
            "name": metric_id.replace("_", " ").title(),
            "higher_is_better": True,
            "group": metric_id.removesuffix("_score"),
        }
        for metric_id in DIMENSION_METRICS
    }
)
METRIC_SPECS["wbench_average"] = {
    "name": "WBench Average",
    "higher_is_better": True,
    "group": "aggregate",
}

ALIASES = {
    "average": "wbench_average",
    "avg": "wbench_average",
    "overall": "wbench_average",
    "wbench": "wbench_average",
    "wbench_average": "wbench_average",
    "subject_consistency_cross_model": "subject_consistency",
    "gated_spatial": "gated_spatial_consistency",
    "quality": "quality_score",
    "setting": "setting_score",
    "interaction": "interaction_score",
    "consistency": "consistency_score",
    "physical": "physical_score",
}


def canonical_metric_id(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(text, text)


def scalar(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            number = float(text)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        return number / 100.0 if is_percent else number
    if isinstance(value, Mapping):
        for key in ("mean", "score", "normalized_score", "value", "average", "avg"):
            if key in value:
                number = scalar(value[key])
                if number is not None:
                    return number
    return None


def mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores_from_report(payload: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    full = payload.get("full")
    if isinstance(full, Mapping):
        for key, value in full.items():
            metric_id = canonical_metric_id(key)
            if metric_id not in METRIC_ORDER:
                continue
            number = scalar(value)
            if number is not None:
                scores[metric_id] = number
    for container_key in ("metrics", "summary", "leaderboard"):
        container = payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            metric_id = canonical_metric_id(key)
            if metric_id not in METRIC_ORDER:
                continue
            number = scalar(value)
            if number is not None:
                scores[metric_id] = number
    return add_aggregate_scores(scores)


def extract_scores_from_eval_dir(eval_dir: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        metric_id = canonical_metric_id(metric_dir.name)
        if metric_id not in METRIC_ORDER:
            continue
        values: list[float] = []
        for result_file in sorted(metric_dir.glob("case_*.json")):
            try:
                payload = read_json(result_file)
            except Exception:
                continue
            number = scalar(payload)
            if number is None and isinstance(payload, Mapping):
                summary = payload.get("summary")
                if isinstance(summary, Mapping):
                    number = scalar(summary.get(metric_id))
                if number is None and metric_id == "navigation_trajectory":
                    number = scalar(payload.get("NavScore"))
                if number is None and metric_id == "spatial_consistency":
                    number = scalar(payload.get("ret_sim"))
            if number is not None:
                values.append(number)
        score = mean(values)
        if score is not None:
            scores[metric_id] = score
    return add_aggregate_scores(scores)


def add_aggregate_scores(scores: Mapping[str, float]) -> dict[str, float]:
    merged = dict(scores)
    for group, metric_ids in METRIC_GROUPS.items():
        if all(metric_id in merged for metric_id in metric_ids):
            merged.setdefault(f"{group}_score", sum(merged[metric_id] for metric_id in metric_ids) / len(metric_ids))
    if all(metric_id in merged for metric_id in DIMENSION_METRICS):
        merged.setdefault(
            "wbench_average",
            sum(merged[metric_id] for metric_id in DIMENSION_METRICS) / len(DIMENSION_METRICS),
        )
    return merged


def resolve_results_path(args: argparse.Namespace) -> Path | None:
    if args.official_results_path is not None:
        return args.official_results_path
    env = env_path("WORLDFOUNDRY_WBENCH_RESULTS_PATH")
    if env is not None:
        return env
    generated_dir = args.generated_artifact_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    for root in (generated_dir, args.output_dir):
        if root is None:
            continue
        root = root.expanduser().resolve()
        for candidate in (
            root / "report.json",
            root / "evaluation" / "report.json",
            root / "evaluation",
        ):
            if candidate.exists():
                return candidate
    return None


def resolve_wbench_root(args: argparse.Namespace) -> Path:
    explicit = args.wbench_root or env_path("WORLDFOUNDRY_WBENCH_ROOT")
    return (explicit or DEFAULT_WBENCH_ROOT).expanduser().resolve()


def resolve_model_id(args: argparse.Namespace) -> str:
    """Resolve the submitted model id without coupling WBench to a model registry."""

    value = (
        args.model_name
        or os.environ.get("WORLDFOUNDRY_MODEL_ID")
        or os.environ.get("WORLDFOUNDRY_WBENCH_MODEL_NAME")
    )
    if not value:
        raise ValueError(
            "--model-name, WORLDFOUNDRY_MODEL_ID, or "
            "WORLDFOUNDRY_WBENCH_MODEL_NAME is required for --run-official"
        )
    return str(value)


def runtime_model_name(model_id: str) -> str:
    """Return a safe WBench work-directory segment for an arbitrary model id."""

    value = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._-")
    if not value:
        raise ValueError(f"model id does not contain a safe path component: {model_id!r}")
    return value


def resolve_dataset_root(args: argparse.Namespace, *, wbench_root: Path) -> Path:
    """Resolve the official dataset separately from the checked-in runtime source."""

    candidates = (
        args.dataset_root,
        env_path("WORLDFOUNDRY_WBENCH_DATASET_ROOT"),
        wbench_root / "data",
    )
    for candidate in candidates:
        if candidate is None:
            continue
        root = Path(candidate).expanduser().resolve()
        if (root / "cases").is_dir():
            return root
    raise FileNotFoundError(
        "WBench dataset is missing. Pass --dataset-root or set "
        "WORLDFOUNDRY_WBENCH_DATASET_ROOT to a directory containing cases/."
    )


def expected_video_count(dataset_root: Path, submission_protocol: str) -> int:
    """Count the official cases applicable to the declared model interface."""

    cases = tuple(sorted((dataset_root / "cases").glob("case_*.json")))
    if submission_protocol not in {"camera_conditioned", "action_conditioned"}:
        return len(cases)
    return sum(
        1
        for case_path in cases
        if any(
            isinstance(interaction, Mapping) and interaction.get("type") == "navigation"
            for interaction in (read_json(case_path).get("interactions") or ())
        )
    )


def resolve_isolated_work_dir(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    wbench_root: Path,
) -> Path:
    """Keep all mutable WBench runtime state under the requested output directory."""

    work_dir = (args.work_dir or (output_dir / "wbench_work")).expanduser().resolve()
    if work_dir != output_dir and output_dir not in work_dir.parents:
        raise ValueError(f"--work-dir must be inside --output-dir: {work_dir}")
    if work_dir == wbench_root or wbench_root in work_dir.parents:
        raise ValueError(f"WBench work directory must not modify the in-tree runtime: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _official_video_name(path: Path) -> tuple[str, str] | None:
    for pattern in (OFFICIAL_VIDEO_RE, MATERIALIZED_VIDEO_RE):
        match = pattern.fullmatch(path.name)
        if match is not None:
            case_id = match.group("case_id")
            return case_id, f"case_{case_id}_combined.mp4"
    return None


def stage_generated_videos(
    *,
    generated_artifact_dir: Path,
    videos_dir: Path,
    dataset_root: Path,
    staging_manifest_path: Path,
) -> list[dict[str, Any]]:
    """Link generated WBench clips into the official isolated work layout.

    Only filenames that identify an official case are accepted. Arbitrarily
    assigning unnamed videos to cases would silently invalidate evaluation.
    """

    source_root = generated_artifact_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"generated artifact directory is missing: {source_root}")
    candidates: list[tuple[str, Path, str, int, int]] = []
    staged_case_ids: set[str] = set()
    for source in sorted(source_root.rglob("*.mp4")):
        if not source.is_file() or videos_dir == source.parent or videos_dir in source.parents:
            continue
        resolved_name = _official_video_name(source)
        if resolved_name is None:
            continue
        case_id, target_name = resolved_name
        case_path = dataset_root / "cases" / f"case_{case_id}.json"
        if not case_path.is_file():
            raise FileNotFoundError(
                f"generated video {source.name!r} does not match a case in {dataset_root / 'cases'}"
            )
        if case_id in staged_case_ids:
            raise ValueError(f"multiple generated videos resolve to WBench case {case_id}")
        staged_case_ids.add(case_id)
        source_stat = source.stat()
        candidates.append((case_id, source, target_name, source_stat.st_size, source_stat.st_mtime_ns))
    if not candidates:
        raise ValueError(
            "generated artifact directory contains no identifiable WBench videos; "
            "expected case_<id>_combined.mp4, case_<id>.mp4, or "
            "case_<id>__generated_video.mp4"
        )

    candidate_signature = sorted(
        (case_id, str(source.resolve()), size, mtime_ns)
        for case_id, source, _target_name, size, mtime_ns in candidates
    )
    if staging_manifest_path.is_file() and (videos_dir.parent / "evaluation").exists():
        previous = read_json(staging_manifest_path)
        previous_rows = previous.get("videos") if isinstance(previous, Mapping) else None
        if isinstance(previous_rows, list):
            previous_signature = sorted(
                (
                    str(row.get("case_id")),
                    str(row.get("source")),
                    row.get("size"),
                    row.get("mtime_ns"),
                )
                for row in previous_rows
                if isinstance(row, Mapping)
            )
            if previous_signature != candidate_signature:
                raise ValueError(
                    "generated WBench inputs changed while cached evaluation results exist; "
                    "use a new --output-dir to avoid mixing scores from different videos"
                )

    videos_dir.mkdir(parents=True, exist_ok=True)
    for stale in videos_dir.glob("case_*_combined.mp4"):
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    rows: list[dict[str, Any]] = []
    for case_id, source, target_name, size, mtime_ns in candidates:
        target = videos_dir / target_name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(source.resolve())
            method = "symlink"
        except OSError:
            shutil.copy2(source, target)
            method = "copy"
        rows.append(
            {
                "case_id": case_id,
                "source": str(source.resolve()),
                "destination": str(target),
                "method": method,
                "size": size,
                "mtime_ns": mtime_ns,
            }
        )
    write_json(staging_manifest_path, {"videos": rows, "video_count": len(rows)})
    return rows


def generated_video_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES)


def scorecard_from_scores(
    *,
    benchmark_id: str,
    output_dir: Path,
    source_path: Path,
    scores: Mapping[str, float],
    official_runtime_executed: bool,
    runtime_summary: Mapping[str, Any] | None = None,
    generated_artifact_dir: Path | None = None,
    model_id: str | None = None,
    submission_protocol: str = "unverified",
) -> dict[str, Any]:
    per_metric: dict[str, dict[str, Any]] = {}
    for metric_id in METRIC_ORDER:
        score = scores.get(metric_id)
        spec = METRIC_SPECS[metric_id]
        per_metric[metric_id] = {
            "metric_id": metric_id,
            "name": spec["name"],
            "group": spec["group"],
            "available": score is not None,
            "raw_score": score,
            "normalized_score": score,
            "score": score,
            "higher_is_better": spec["higher_is_better"],
            "source": "wbench_official_runtime" if official_runtime_executed else "wbench_results_file",
            "source_path": str(source_path),
            "reason": None if score is not None else "score_not_available_in_wbench_results",
        }
    available = [row for row in per_metric.values() if row["available"]]
    raw_metric_table = output_dir / "raw_metric_table.jsonl"
    raw_metric_table.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in per_metric.values()),
        encoding="utf-8",
    )
    scorecard_path = output_dir / "scorecard.json"
    component_metric_ids = tuple(metric for metric_ids in METRIC_GROUPS.values() for metric in metric_ids)
    missing_component_metrics = [metric for metric in component_metric_ids if metric not in scores]
    interaction_protocol_declared = submission_protocol != "unverified"
    comparability_blockers: list[str] = []
    if not official_runtime_executed:
        comparability_blockers.append("official WBench runtime was not executed by this run")
    if not interaction_protocol_declared:
        comparability_blockers.append(
            "the submitted model did not declare a WBench multi-turn interaction protocol"
        )
    else:
        comparability_blockers.append(
            "the declared WBench multi-turn generation provenance was not mechanically verified"
        )
    if missing_component_metrics:
        comparability_blockers.append(
            f"official metric coverage is partial ({len(missing_component_metrics)} component metrics missing)"
        )
    expected_video_count = (runtime_summary or {}).get("expected_video_count")
    evaluated_video_count = (runtime_summary or {}).get("evaluated_video_count")
    if (
        isinstance(expected_video_count, int)
        and isinstance(evaluated_video_count, int)
        and evaluated_video_count < expected_video_count
    ):
        comparability_blockers.append(
            f"official case coverage is partial ({evaluated_video_count}/{expected_video_count} videos)"
        )
    official_verified = not comparability_blockers and bool(available)
    leaderboard_blockers = list(comparability_blockers)
    leaderboard_blockers.append(
        "leaderboard submission coverage, turns.json boundaries, and official packaging were not validated"
    )
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": official_runtime_executed and bool(available),
        "leaderboard_valid": False,
        "leaderboard_blockers": leaderboard_blockers,
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": bool(available),
        "run": {
            "status": "succeeded" if available else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_wbench_official_runner",
            "returncode": 0 if available else 1,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {"benchmark_id": benchmark_id, "name": "WBench"},
        "model": {
            "model_id": model_id,
            "submission_protocol": submission_protocol,
        },
        "metrics": {
            "leaderboard": {
                metric_id: row["normalized_score"]
                for metric_id, row in per_metric.items()
                if row["available"] and row["normalized_score"] is not None
            },
            "per_metric": per_metric,
            "groups": {group: list(metric_ids) for group, metric_ids in METRIC_GROUPS.items()},
            "summary": {
                "sample_count": evaluated_video_count
                if isinstance(evaluated_video_count, int)
                else generated_video_count(generated_artifact_dir),
                "metric_count": len(METRIC_ORDER),
                "available_metrics": len(available),
                "failed_metrics": len(METRIC_ORDER) - len(available),
            },
        },
        "evaluation": {
            "kind": "wbench_official_in_tree" if official_runtime_executed else "wbench_result_normalizer",
            "available_metric_count": len(available),
            "declared_metric_count": len(METRIC_ORDER),
            "official_runtime_executed": official_runtime_executed,
            "benchmark_comparable": official_verified,
            "interaction_protocol_declared": interaction_protocol_declared,
            "interaction_protocol_verified": False,
            "comparability_blockers": comparability_blockers,
            "missing_component_metrics": missing_component_metrics,
        },
        "dataset": {
            "generated_artifact_dir": None if generated_artifact_dir is None else str(generated_artifact_dir),
            "generated_video_count": generated_video_count(generated_artifact_dir),
            "results_path": str(source_path),
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table.resolve()),
            "official_results_path": str(source_path.resolve()),
        },
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def normalize_wbench_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = resolve_results_path(args)
    if results_path is None:
        raise ValueError("--official-results-path or WORLDFOUNDRY_WBENCH_RESULTS_PATH is required")
    results_path = results_path.expanduser().resolve()
    if results_path.is_dir():
        report_path = results_path / "report.json"
        scores = extract_scores_from_report(read_json(report_path)) if report_path.is_file() else extract_scores_from_eval_dir(results_path)
        source_path = report_path if report_path.is_file() else results_path
    else:
        scores = extract_scores_from_report(read_json(results_path))
        source_path = results_path
    generated_dir = args.generated_artifact_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    model_id = args.model_name or os.environ.get("WORLDFOUNDRY_MODEL_ID") or os.environ.get(
        "WORLDFOUNDRY_WBENCH_MODEL_NAME"
    )
    return scorecard_from_scores(
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        source_path=source_path,
        scores=scores,
        official_runtime_executed=official_runtime_executed,
        runtime_summary=runtime_summary,
        generated_artifact_dir=generated_dir,
        model_id=model_id,
        submission_protocol=args.submission_protocol,
    )


def run_official_wbench(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_wbench_root(args)
    if not (root / "main.py").is_file():
        raise FileNotFoundError(f"missing in-tree WBench runtime: {root / 'main.py'}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = resolve_model_id(args)
    model = runtime_model_name(model_id)
    args.model_name = model_id
    dataset_root = resolve_dataset_root(args, wbench_root=root)
    work_dir = resolve_isolated_work_dir(args, output_dir=output_dir, wbench_root=root)
    videos_dir = work_dir / model / "videos"
    generated_dir = args.generated_artifact_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    staging_manifest_path = output_dir / "wbench_staging.json"
    if generated_dir is not None:
        generated_dir = generated_dir.expanduser().resolve()
        args.generated_artifact_dir = generated_dir
        staged_rows = stage_generated_videos(
            generated_artifact_dir=generated_dir,
            videos_dir=videos_dir,
            dataset_root=dataset_root,
            staging_manifest_path=staging_manifest_path,
        )
    else:
        staged_rows = []
        if not any(videos_dir.glob("case_*_combined.mp4")):
            raise ValueError(
                "--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required "
                "unless the isolated work directory already contains WBench videos"
            )
    gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES")
    command = [
        sys.executable,
        str(root / "main.py"),
        "--model",
        model,
        "--phase",
        args.phase,
        "--work_dir",
        str(work_dir),
    ]
    if args.metrics:
        command.extend(["--metrics", args.metrics])
    if gpus:
        command.extend(["--gpus", gpus])
    if args.skip_sam2:
        command.append("--skip_sam2")
    if args.skip_da3:
        command.append("--skip_da3")
    if args.skip_megasam:
        command.append("--skip_megasam")
    if args.enable_megasam:
        command.append("--enable_megasam")
    if args.vlm_workers is not None:
        command.extend(["--vlm_workers", str(args.vlm_workers)])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(REPO_ROOT), str(root), env.get("PYTHONPATH", "")) if path
    )
    env["WBENCH_DATA_DIR"] = str(dataset_root)
    if args.weights_dir is not None:
        env["WBENCH_WEIGHTS_DIR"] = str(args.weights_dir.expanduser().resolve())
    started = utc_now_iso()
    proc = subprocess.run(command, cwd=str(root), env=env, text=True, capture_output=True, check=False)
    log_path = output_dir / "wbench_official_runtime.log"
    log_path.write_text((proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    eval_dir = Path(work_dir).expanduser().resolve() / model / "evaluation"
    report_path = eval_dir / "report.json"
    args.official_results_path = report_path if report_path.is_file() else eval_dir
    runtime_summary = {
        "command": command,
        "returncode": proc.returncode,
        "started_at": started,
        "log_path": str(log_path.resolve()),
        "model_id": model_id,
        "runtime_model_name": model,
        "phase": args.phase,
        "metrics": args.metrics,
        "dataset_root": str(dataset_root),
        "work_dir": str(work_dir),
        "staged_video_count": len(staged_rows),
        "evaluated_video_count": len(tuple(videos_dir.glob("case_*_combined.mp4"))),
        "expected_video_count": expected_video_count(dataset_root, args.submission_protocol),
        "staging_manifest": str(staging_manifest_path) if staged_rows else None,
    }
    if proc.returncode != 0:
        raise RuntimeError(f"WBench official runtime failed with code {proc.returncode}; see {log_path}")
    return normalize_wbench_results(args, official_runtime_executed=True, runtime_summary=runtime_summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize WBench official outputs.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "wbench"))
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"), required=False)
    parser.add_argument("--generated-artifact-dir", "--generated-video-dir", dest="generated_artifact_dir", type=Path)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--wbench-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--submission-protocol",
        choices=SUBMISSION_PROTOCOLS,
        default=os.environ.get("WORLDFOUNDRY_WBENCH_SUBMISSION_PROTOCOL", "unverified"),
        help=(
            "Generation protocol used to create the combined clips. The default is deliberately "
            "non-comparable until a WBench multi-turn protocol is declared."
        ),
    )
    parser.add_argument("--phase", choices=("all", "precompute", "gpu", "vlm", "report"), default="all")
    parser.add_argument("--metrics", help="Comma-separated WBench metric or dimension filter.")
    parser.add_argument("--gpus", help="Comma-separated GPU ids passed to the in-tree runtime.")
    parser.add_argument("--vlm-workers", type=int)
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--skip-da3", action="store_true")
    parser.add_argument("--skip-megasam", action="store_true")
    parser.add_argument("--enable-megasam", action="store_true", help="Backward-compatible no-op; MegaSAM runs unless --skip-megasam is set.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    try:
        scorecard = run_official_wbench(args) if args.run_official else normalize_wbench_results(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "leaderboard_blockers": [str(exc)],
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "benchmark_zoo_wbench_official_runner",
                "error": str(exc),
            },
            "benchmark": {"benchmark_id": args.benchmark_id, "name": "WBench"},
            "model": {
                "model_id": args.model_name or os.environ.get("WORLDFOUNDRY_MODEL_ID"),
                "submission_protocol": args.submission_protocol,
            },
            "metrics": {
                "leaderboard": {},
                "per_metric": {},
                "summary": {
                    "sample_count": 0,
                    "metric_count": len(METRIC_ORDER),
                    "available_metrics": 0,
                    "failed_metrics": len(METRIC_ORDER),
                },
            },
            "artifacts": {"scorecard": str((args.output_dir / "scorecard.json").resolve())},
        }
        write_json(args.output_dir / "scorecard.json", failure)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "scorecard": failure}, indent=2, ensure_ascii=False))
        else:
            print(f"wbench: failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": scorecard.get("normalization_ok") is True, "scorecard": scorecard}, indent=2, ensure_ascii=False))
    else:
        print(f"wbench: normalized {scorecard['evaluation']['available_metric_count']} metrics")
    return 0 if scorecard.get("normalization_ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
