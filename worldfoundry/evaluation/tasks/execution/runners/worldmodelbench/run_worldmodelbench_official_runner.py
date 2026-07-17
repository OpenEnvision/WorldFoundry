#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.utils import REPO_ROOT

from worldfoundry.runtime.env import resolve_hf_cache_dir  # type: ignore[reportMissingImports]  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, load_json, utc_now_iso, write_json, write_jsonl

DEFAULT_WORLDMODELBENCH_ROOT = Path(__file__).resolve().parent / "runtime"
WORLDMODELBENCH_HF_CACHE_DIR = "datasets--Efficient-Large-Model--worldmodelbench"
WORLDMODELBENCH_MANIFEST_ASSET = bundled_benchmark_asset("worldmodelbench", "worldmodelbench.json")
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})

METRIC_ALIASES = {
    "instruction": "instruction_following",
    "instruction_following": "instruction_following",
    "Instruction Following": "instruction_following",
    "common_sense": "common_sense",
    "Common Sense": "common_sense",
    "physical_laws": "physical_adherence",
    "physical_adherence": "physical_adherence",
    "Physical Adherence": "physical_adherence",
    "world_model_average": "world_model_average",
    "World Model Average": "world_model_average",
    "total_score": "world_model_average",
    "Total Score": "world_model_average",
}
METRIC_RAW_MAX = {
    "instruction_following": 3.0,
    "common_sense": 2.0,
    "physical_adherence": 5.0,
    "world_model_average": 10.0,
}
OFFICIAL_CATEGORY_TO_METRIC = {
    "instruction": "instruction_following",
    "common_sense": "common_sense",
    "physical_laws": "physical_adherence",
}
OFFICIAL_SUBCATEGORIES = {
    "common_sense": ("framewise", "temporal"),
    "physical_laws": ("newton", "mass", "fluid", "penetration", "gravity"),
}
METRIC_ORDER = ("instruction_following", "common_sense", "physical_adherence", "world_model_average")


def list_video_files(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [str(path)] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    return [str(item) for item in sorted(path.rglob("*")) if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS]


def latest_hf_dataset_snapshot(cache_dir: Path) -> Path | None:
    dataset_dir = cache_dir / WORLDMODELBENCH_HF_CACHE_DIR
    snapshots_dir = dataset_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return snapshots[-1] if snapshots else None


def resolve_worldmodelbench_data_root(args: argparse.Namespace) -> Path:
    explicit_root = args.data_root
    root_manifest = resolve_worldmodelbench_manifest_path(args.worldmodelbench_root)
    if explicit_root is not None and resolve_worldmodelbench_manifest_path(explicit_root) is not None:
        return explicit_root
    hf_snapshot = latest_hf_dataset_snapshot(args.hf_cache_dir)
    if hf_snapshot is not None:
        return hf_snapshot
    if root_manifest is not None:
        return args.worldmodelbench_root
    if WORLDMODELBENCH_MANIFEST_ASSET.is_file():
        return WORLDMODELBENCH_MANIFEST_ASSET.parent
    if explicit_root is not None:
        return explicit_root
    return args.worldmodelbench_root


def resolve_worldmodelbench_runtime_root(args: argparse.Namespace, data_root: Path) -> Path:
    if data_root.is_dir() and (data_root / "evaluation.py").is_file():
        return data_root
    if args.worldmodelbench_root.is_dir() and (args.worldmodelbench_root / "evaluation.py").is_file():
        return args.worldmodelbench_root
    return data_root


def resolve_worldmodelbench_manifest_path(data_root: Path) -> Path | None:
    direct_candidates = (
        data_root / "worldmodelbench.json",
        data_root / "data" / "worldmodelbench.json",
        data_root / "WorldModelBench" / "worldmodelbench.json",
    )
    for path in direct_candidates:
        if path.is_file():
            return path
    if data_root.is_file() and data_root.name == "worldmodelbench.json":
        return data_root
    if data_root.is_dir():
        matches = sorted(path for path in data_root.rglob("worldmodelbench.json") if path.is_file())
        if matches:
            return matches[0]
    return None


def manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "test", "samples", "examples", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def load_official_manifest(data_root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_worldmodelbench_manifest_path(data_root)
    if manifest_path is None:
        return []
    payload = load_json(manifest_path)
    return manifest_rows(payload)


def manifest_item_stem(item: dict[str, Any]) -> str | None:
    for key in ("first_frame", "image", "image_path", "video", "video_path", "video_name", "sample_id", "id", "prompt_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).stem
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def expected_video_stems(data_root: Path) -> list[str]:
    stems: list[str] = []
    for item in load_official_manifest(data_root):
        stem = manifest_item_stem(item)
        if stem:
            stems.append(stem)
    return sorted(set(stems))


def build_manifest_coverage(data_root: Path, video_dir: Path | None) -> dict[str, Any]:
    manifest_path = resolve_worldmodelbench_manifest_path(data_root)
    expected_stems = expected_video_stems(data_root)
    video_files = list_video_files(video_dir)
    generated_stems = sorted({Path(path).stem for path in video_files})
    expected_set = set(expected_stems)
    generated_set = set(generated_stems)
    missing = sorted(expected_set - generated_set)
    unexpected = sorted(generated_set - expected_set)
    matched = sorted(expected_set & generated_set)
    return {
        "data_root": str(data_root),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_available": manifest_path is not None,
        "expected_file_count": len(expected_stems),
        "generated_file_count": len(video_files),
        "matched_file_count": len(matched),
        "missing_file_count": len(missing),
        "unexpected_file_count": len(unexpected),
        "coverage_complete": bool(expected_stems) and not missing and not unexpected,
        "missing_video_names": [f"{stem}.mp4" for stem in missing[:20]],
        "unexpected_video_names": [f"{stem}.mp4" for stem in unexpected[:20]],
    }


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, list) or isinstance(value, tuple):
        numeric = [scalar(item) for item in value]
        numeric = [item for item in numeric if item is not None]
        if numeric:
            return sum(numeric) / len(numeric)
    if isinstance(value, dict):
        for key in ("score", "raw_score", "normalized_score", "value", "mean", "average", "avg", "overall"):
            if key in value:
                number = scalar(value[key])
                if number is not None:
                    return number
    return None


def normalized_only_score(value: Any) -> float | None:
    if not isinstance(value, dict) or "normalized_score" not in value:
        return None
    raw_keys = ("score", "raw_score", "value", "mean", "average", "avg", "overall")
    if any(key in value for key in raw_keys):
        return None
    return scalar(value["normalized_score"])


def normalize_metric_score(metric_id: str, raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    max_score = METRIC_RAW_MAX.get(metric_id)
    if max_score is None or max_score <= 0:
        return raw_score
    return raw_score / max_score


def expected_acc_length(category: str, sample_count: int) -> int | None:
    if category == "instruction":
        return sample_count
    subcategories = OFFICIAL_SUBCATEGORIES.get(category)
    if subcategories:
        return sample_count * len(subcategories)
    return None


def validate_official_acc_shape(raw_results: dict[str, Any]) -> dict[str, Any]:
    accs = raw_results.get("accs")
    preds = raw_results.get("preds")
    checked = isinstance(accs, dict)
    sample_count = len(preds) if isinstance(preds, dict) else 0
    issues: list[dict[str, Any]] = []
    category_lengths: dict[str, Any] = {}

    if not checked:
        return {
            "checked": False,
            "ok": True,
            "sample_count": sample_count,
            "category_lengths": category_lengths,
            "issues": issues,
        }
    if sample_count <= 0:
        issues.append({"reason": "missing_or_empty_preds", "sample_count": sample_count})

    for category in OFFICIAL_CATEGORY_TO_METRIC:
        values = accs.get(category)
        actual_length = len(values) if isinstance(values, list) else None
        expected_length = expected_acc_length(category, sample_count) if sample_count > 0 else None
        category_lengths[category] = {
            "actual": actual_length,
            "expected": expected_length,
        }
        if values is None:
            issues.append({"category": category, "reason": "missing_acc_category"})
            continue
        if not isinstance(values, list):
            issues.append({"category": category, "reason": "acc_category_not_list"})
            continue
        if expected_length is not None and actual_length != expected_length:
            issues.append(
                {
                    "category": category,
                    "reason": "invalid_acc_shape",
                    "actual": actual_length,
                    "expected": expected_length,
                }
            )

    return {
        "checked": True,
        "ok": not issues,
        "sample_count": sample_count,
        "category_lengths": category_lengths,
        "issues": issues,
    }


def compute_official_category_score(category: str, values: Any, sample_count: int) -> tuple[float | None, dict[str, float]]:
    if not isinstance(values, list) or not values:
        return None, {}
    numeric_values = [scalar(value) for value in values]
    numeric_values = [value for value in numeric_values if value is not None]
    if not numeric_values:
        return None, {}

    expected_length = expected_acc_length(category, sample_count) if sample_count > 0 else None
    if expected_length is None or len(numeric_values) != expected_length:
        return None, {}

    subcategories = OFFICIAL_SUBCATEGORIES.get(category)
    if not subcategories:
        return sum(numeric_values) / len(numeric_values), {}

    num_subcategories = len(numeric_values) // sample_count
    if num_subcategories != len(subcategories):
        return sum(numeric_values) / len(numeric_values), {}

    sub_scores = {
        subcategory: sum(numeric_values[index::num_subcategories]) / len(numeric_values[index::num_subcategories])
        for index, subcategory in enumerate(subcategories)
    }
    return sum(sub_scores.values()), sub_scores


def extract_scores_from_official_accs(raw_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    accs = raw_results.get("accs")
    if not isinstance(accs, dict):
        return {}
    preds = raw_results.get("preds")
    sample_count = len(preds) if isinstance(preds, dict) else 0

    extracted: dict[str, dict[str, Any]] = {}
    for category, values in accs.items():
        metric_id = OFFICIAL_CATEGORY_TO_METRIC.get(str(category))
        if not metric_id:
            continue
        raw_score, sub_scores = compute_official_category_score(str(category), values, sample_count)
        extracted[metric_id] = {
            "raw_score": raw_score,
            "sub_scores": sub_scores,
            "source": f"accs.{category}",
        }

    component_scores = [
        item["raw_score"]
        for metric_id, item in extracted.items()
        if metric_id in {"instruction_following", "common_sense", "physical_adherence"} and item["raw_score"] is not None
    ]
    if len(component_scores) == 3:
        extracted["world_model_average"] = {
            "raw_score": sum(component_scores),
            "sub_scores": {},
            "source": "computed_from_official_accs",
        }
    return extracted


def extract_scores_from_metric_maps(raw_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    candidate_maps: list[tuple[str, dict[str, Any]]] = []
    for key in ("scores", "metrics", "leaderboard", "leaderboard_metrics"):
        value = raw_results.get(key)
        if isinstance(value, dict):
            candidate_maps.append((key, value))
    candidate_maps.append(("root", raw_results))

    for source_name, metric_map in candidate_maps:
        for raw_key, raw_value in metric_map.items():
            metric_id = METRIC_ALIASES.get(str(raw_key))
            if not metric_id or metric_id in extracted:
                continue
            normalized_score = normalized_only_score(raw_value)
            raw_score = None if normalized_score is not None else scalar(raw_value)
            extracted[metric_id] = {
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "sub_scores": {},
                "source": f"{source_name}.{raw_key}",
                "score_scale": "normalized" if normalized_score is not None else "raw",
            }
    return extracted


def extract_scores(raw_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    official_scores = extract_scores_from_official_accs(raw_results)
    map_scores = extract_scores_from_metric_maps(raw_results)
    return {**map_scores, **official_scores}


def iter_judge_rows(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
    preds = raw_results.get("preds")
    if not isinstance(preds, dict):
        return []
    rows: list[dict[str, Any]] = []
    for video_name, per_video in sorted(preds.items()):
        if not isinstance(per_video, dict):
            rows.append({"video_name": video_name, "raw_response": per_video})
            continue
        for evaluation_type, responses in sorted(per_video.items()):
            response_list = responses if isinstance(responses, list) else [responses]
            for index, response in enumerate(response_list):
                rows.append(
                    {
                        "video_name": video_name,
                        "evaluation_type": evaluation_type,
                        "response_index": index,
                        "judge_response": response,
                    }
                )
    return rows


def normalize_worldmodelbench_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    output_dir: Path,
    upstream_results_path: Path,
    data_root: Path,
    video_dir: Path | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    judge_responses_path = output_dir / "judge_responses.jsonl"
    video_files = list_video_files(video_dir)
    manifest_coverage = build_manifest_coverage(data_root, video_dir)
    shape_validation = validate_official_acc_shape(raw_results)
    extracted_scores = extract_scores(raw_results)

    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in METRIC_ORDER:
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        normalized_score = item.get("normalized_score")
        if normalized_score is None:
            normalized_score = normalize_metric_score(metric_id, raw_score)
        available = raw_score is not None or normalized_score is not None
        row = {
            "metric_id": metric_id,
            "available": available,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "raw_max_score": METRIC_RAW_MAX[metric_id],
            "source": item.get("source"),
            "score_scale": item.get("score_scale", "raw"),
            "sub_scores": item.get("sub_scores") or {},
        }
        if not available:
            row["reason"] = "score_not_found_in_worldmodelbench_results"
        elif raw_score is not None:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    available_count = sum(1 for row in metric_rows if row["available"])
    shape_ok = bool(shape_validation["ok"])
    judge_rows = iter_judge_rows(raw_results)
    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(judge_responses_path, judge_rows)

    normalization_ok = returncode == 0 and available_count > 0 and shape_ok
    normalizer_only = command is None
    official_verified = command is not None and normalization_ok
    integration_evidence = official_verified and manifest_coverage["coverage_complete"]
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": "official_verified" if returncode == 0 and available_count and shape_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_worldmodelbench_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "WorldModelBench",
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_judge_model": True,
        },
        "dataset": {
            "data_root": str(data_root),
            "generated_artifact_dir": None if video_dir is None else str(video_dir),
            "generated_file_count": len(video_files),
            "official_manifest_coverage": manifest_coverage,
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "official WorldModelBench runtime validation; full leaderboard evidence still requires the upstream submission protocol",
            ],
        },
        "generation": {
            "successful": len(video_files),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "worldmodelbench_public_metrics": [row["metric_id"] for row in metric_rows if row["available"]],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(raw_results.get("preds", {})) if isinstance(raw_results.get("preds"), dict) else len(video_files),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "official_worldmodelbench",
            "upstream_results": str(upstream_results_path),
            "num_results": len(judge_rows),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_results_imported": normalizer_only and normalization_ok,
            "official_result_shape": shape_validation,
            "official_manifest_coverage": manifest_coverage,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "judge_responses": str(judge_responses_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def upstream_results_path(save_name: Path, cot: bool) -> Path:
    suffix = "_cot.json" if cot else ".json"
    return Path(f"{save_name}{suffix}")


def build_official_command(args: argparse.Namespace, save_name: Path, runtime_root: Path) -> list[str]:
    command = [
        args.python,
        str(runtime_root / "evaluation.py"),
        "--model_name",
        args.model_name,
        "--video_dir",
        str(args.video_dir),
        "--judge",
        str(args.judge),
        "--save_name",
        str(save_name),
    ]
    if args.cot:
        command.append("--cot")
    return command


def run_official_worldmodelbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    upstream_output_dir = output_dir / "upstream"
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_worldmodelbench_data_root(args)
    runtime_root = resolve_worldmodelbench_runtime_root(args, data_root)

    if args.from_upstream_results:
        raw_results = load_json(args.from_upstream_results)
        if not isinstance(raw_results, dict):
            raise ValueError(f"WorldModelBench result JSON must be an object: {args.from_upstream_results}")
        return normalize_worldmodelbench_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=args.from_upstream_results,
            data_root=data_root,
            video_dir=args.video_dir,
            command=None,
            duration_seconds=None,
            returncode=0,
        )

    if args.video_dir is None:
        raise ValueError("--video-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required unless --official-results-path is used")
    if args.judge is None:
        raise ValueError("--judge or WORLDFOUNDRY_WORLDMODELBENCH_JUDGE is required unless --official-results-path is used")
    if not (runtime_root / "evaluation.py").is_file():
        raise FileNotFoundError(
            "WorldModelBench evaluation.py not found under "
            f"data_root={data_root} or worldmodelbench_root={args.worldmodelbench_root}"
        )

    save_name = upstream_output_dir / "worldmodelbench_results"
    command = build_official_command(args, save_name, runtime_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{runtime_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    manifest_path = resolve_worldmodelbench_manifest_path(data_root)
    if manifest_path is not None:
        env["WORLDFOUNDRY_WORLDMODELBENCH_MANIFEST"] = str(manifest_path)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=runtime_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    results_path = upstream_results_path(save_name, args.cot)
    if results_path.is_file():
        raw_results = load_json(results_path)
        if not isinstance(raw_results, dict):
            raise ValueError(f"WorldModelBench result JSON must be an object: {results_path}")
    else:
        raw_results = {"error": "missing upstream WorldModelBench result JSON"}
        write_json(results_path, raw_results)

    return normalize_worldmodelbench_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        upstream_results_path=results_path,
        data_root=data_root,
        video_dir=args.video_dir,
        command=command,
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official WorldModelBench and normalize its output to a WorldFoundry scorecard."
    )
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "worldmodelbench"))
    parser.add_argument(
        "--worldmodelbench-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDMODELBENCH_ROOT", default=DEFAULT_WORLDMODELBENCH_ROOT),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDMODELBENCH_DATA_ROOT", "WORLDFOUNDRY_BENCHMARK_DATA_ROOT"),
        help="WorldModelBench data root containing worldmodelbench.json; defaults to HF cache snapshot or official repo root.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=resolve_hf_cache_dir())
    parser.add_argument("--video-dir", type=Path, default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"))
    parser.add_argument("--judge", type=Path, default=env_path("WORLDFOUNDRY_WORLDMODELBENCH_JUDGE"))
    parser.add_argument("--model-name", default=os.environ.get("WORLDFOUNDRY_MODEL_NAME", "worldfoundry-validation"))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_WORLDMODELBENCH_TIMEOUT", "3600")))
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--official-results-path", dest="from_upstream_results", type=Path)
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_official_worldmodelbench(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "full_official_ok": scorecard["official_benchmark_verified"] and scorecard["integration_evidence"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "judge_responses": scorecard["artifacts"]["judge_responses"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official WorldModelBench validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
