#!/usr/bin/env python3
"""Run or normalize the in-tree LARYBench runtime without an external checkout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import project_root
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json, write_jsonl
from worldfoundry.runtime.jobs import run_bounded_command

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
RUNNER_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = RUNNER_ROOT
REPO_ROOT = project_root(__file__)
CONFIG_ROOT = bundled_benchmark_asset("larybench", "configs")
STAGES = ("extract", "classify", "regress")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or normalize the in-tree LARYBench latent-action benchmark.")
    parser.add_argument("--benchmark-id", default="larybench")
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--latent-action-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--split", default="all")
    parser.add_argument("--input", type=Path, help="Explicit extraction metadata CSV.")
    parser.add_argument("--config", type=Path, help="Classification config YAML.")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--loader-timeout", type=int, default=0)

    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--classes", type=int)
    parser.add_argument("--model-type", choices=("mlp", "dit"), default="mlp")
    parser.add_argument("--action-mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--action-data-root", type=Path)
    parser.add_argument("--global-stats-json", type=Path)
    parser.add_argument("--val-unseen-csv", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--timeout", type=int, default=24 * 60 * 60)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _resolved_layout(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    return {
        "output_dir": output_dir,
        "metadata_root": (args.metadata_root or _env_path("LARY_METADATA_DIR") or output_dir / "metadata")
        .expanduser()
        .resolve(),
        "latent_action_root": (args.latent_action_root or _env_path("LARY_LA_DIR") or output_dir / "latent_actions")
        .expanduser()
        .resolve(),
        "model_root": (args.model_root or _env_path("MODEL_DIR") or output_dir / "models").expanduser().resolve(),
        "log_root": output_dir / "larybench_runtime",
    }


def _runtime_env(args: argparse.Namespace, layout: Mapping[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LARY_CONFIG_ROOT": str(CONFIG_ROOT),
            "LARY_METADATA_DIR": str(layout["metadata_root"]),
            "LARY_LA_DIR": str(layout["latent_action_root"]),
            "LARY_LOG_DIR": str(layout["log_root"]),
            "MODEL_DIR": str(layout["model_root"]),
            "CUDA_VISIBLE_DEVICES": args.gpus,
        }
    )
    dataset_root = args.dataset_root or _env_path("DATA_DIR")
    if dataset_root is not None:
        env["DATA_DIR"] = str(dataset_root.expanduser().resolve())
    return env


def _require_run_args(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError(f"--run-official --stage {args.stage} requires {', '.join(missing)}")


def _build_runtime_command(args: argparse.Namespace, layout: Mapping[str, Path]) -> list[str]:
    if args.stage not in STAGES:
        raise ValueError(f"--run-official requires --stage {{{','.join(STAGES)}}}")
    _require_run_args(args, "model", "dataset")
    command = [
        sys.executable,
        "-m",
        "worldfoundry.evaluation.tasks.execution.runners.larybench.cli",
        args.stage,
        "--model",
        args.model,
        "--dataset",
        args.dataset,
    ]

    batch_size = args.batch_size
    if args.stage == "extract":
        command.extend(
            [
                "--output",
                str(layout["latent_action_root"]),
                "--split",
                args.split,
                "--batch-size",
                str(batch_size or 16),
                "--num-workers",
                str(args.num_workers),
                "--gpus",
                args.gpus,
                "--mode",
                args.mode,
                "--stride",
                str(args.stride),
            ]
        )
        if args.input:
            command.extend(["--input", str(args.input.expanduser().resolve())])
    elif args.stage == "classify":
        _require_run_args(args, "dim", "classes")
        command.extend(
            [
                "--dim",
                str(args.dim),
                "--classes",
                str(args.classes),
                "--batch-size",
                str(batch_size or 256),
                "--num-workers",
                str(args.num_workers),
                "--loader-timeout",
                str(args.loader_timeout),
                "--gpus",
                args.gpus,
            ]
        )
        if args.config:
            command.extend(["--config", str(args.config.expanduser().resolve())])
    else:
        command.extend(
            [
                "--stride",
                str(args.stride),
                "--model-type",
                args.model_type,
                "--action-mode",
                args.action_mode,
                "--batch-size",
                str(batch_size or 256),
                "--num-workers",
                str(args.num_workers),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--mixed-precision",
                args.mixed_precision,
            ]
        )
        for flag, value in (
            ("--action-data-root", args.action_data_root),
            ("--global-stats-json", args.global_stats_json),
            ("--val-unseen-csv", args.val_unseen_csv),
        ):
            if value:
                command.extend([flag, str(value.expanduser().resolve())])
    return command


def _expected_runtime_result(args: argparse.Namespace, layout: Mapping[str, Path]) -> Path:
    assert args.stage and args.model and args.dataset
    if args.stage == "extract":
        suffix = f"_{args.stride}" if args.mode == "image" else ""
        return layout["metadata_root"] / f"{args.split}_la_{args.dataset}{suffix}_{args.model}.csv"
    if args.stage == "classify":
        return layout["log_root"] / "classification" / args.dataset / args.model / "classification_summary.json"
    run_name = f"{args.dataset}_{args.stride}_{args.model}_{args.model_type}_{args.action_mode}"
    return layout["log_root"] / "regression" / "logs" / run_name / "best_result.json"


def _run_official(args: argparse.Namespace, layout: Mapping[str, Path]) -> tuple[Path, dict[str, Any]]:
    if not (RUNTIME_ROOT / "cli.py").is_file():
        raise FileNotFoundError(f"missing in-tree LARYBench runtime: {RUNTIME_ROOT}")
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    command = _build_runtime_command(args, layout)
    result = run_bounded_command(
        command,
        cwd=REPO_ROOT,
        env=_runtime_env(args, layout),
        timeout=args.timeout,
    )
    log_path = layout["output_dir"] / "larybench_official_runtime.log"
    log_path.write_text(
        f"$ {' '.join(command)}\n\n[stdout]\n{result['stdout']}\n\n[stderr]\n{result['stderr']}\n",
        encoding="utf-8",
    )
    summary = {**result, "runtime_root": str(RUNTIME_ROOT), "log_path": str(log_path)}
    write_json(layout["output_dir"] / "runtime_summary.json", summary)
    if result["returncode"] != 0:
        raise RuntimeError(f"LARYBench {args.stage} failed with exit code {result['returncode']}; see {log_path}")
    expected = _expected_runtime_result(args, layout)
    if not expected.is_file():
        raise FileNotFoundError(f"LARYBench runtime completed but did not write expected result: {expected}")
    return expected, summary


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_first_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle), {})


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confusion_summary(payload: Mapping[str, Any]) -> dict[str, float]:
    matrix = payload.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return {}
    rows = [[float(value) for value in row] for row in matrix]
    total = sum(sum(row) for row in rows)
    if not total:
        return {}
    diagonal = [rows[i][i] if i < len(rows[i]) else 0.0 for i in range(len(rows))]
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    support = [sum(row) for row in rows]
    for index, true_positive in enumerate(diagonal):
        predicted = sum(row[index] for row in rows if index < len(row))
        p = true_positive / predicted if predicted else 0.0
        r = true_positive / support[index] if support[index] else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "top1_accuracy": sum(diagonal) / total,
        "macro_precision": sum(precision) / len(precision),
        "macro_recall": sum(recall) / len(recall),
        "macro_f1": sum(f1) / len(f1),
        "weighted_f1": sum(value * count for value, count in zip(f1, support)) / total,
        "sample_count": total,
    }


def _classification_metrics(path: Path) -> tuple[dict[str, float], Path]:
    root = path if path.is_dir() else path.parent
    candidates = ([path] if path.is_file() else []) + [
        root / "classification_summary.json",
        root / "confusion_matrix.json",
        root / "classification_stats.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = _json(candidate)
        if candidate.name == "confusion_matrix.json" or (
            isinstance(payload, Mapping) and "confusion_matrix" in payload
        ):
            return _confusion_summary(payload), candidate
        if isinstance(payload, Mapping):
            aliases = {"accuracy": "top1_accuracy", "val_accuracy": "top1_accuracy"}
            metrics = {
                aliases.get(str(key), str(key)): number
                for key, value in payload.items()
                if (number := _numeric(value)) is not None
            }
            if metrics:
                return metrics, candidate
        if isinstance(payload, list) and payload:
            fields = ("precision", "recall", "f1")
            metrics = {}
            for field in fields:
                values = [_numeric(item.get(field)) for item in payload if isinstance(item, Mapping)]
                clean = [value for value in values if value is not None]
                if clean:
                    metrics[f"macro_{field}"] = sum(clean) / len(clean)
            if metrics:
                return metrics, candidate
    recursive = sorted(root.rglob("classification_summary.json")) + sorted(root.rglob("confusion_matrix.json"))
    if recursive:
        return _classification_metrics(recursive[-1])
    raise FileNotFoundError(f"no LARYBench classification result found under {path}")


def _regression_metrics(path: Path) -> tuple[dict[str, float], Path]:
    root = path if path.is_dir() else path.parent
    candidates = ([path] if path.is_file() else []) + [root / "best_result.json", root / "best_result.csv"]
    candidates += sorted(root.rglob("best_result.json")) + sorted(root.rglob("best_result.csv"))
    for candidate in reversed(candidates):
        if not candidate.is_file():
            continue
        payload = _json(candidate) if candidate.suffix.lower() == ".json" else _csv_first_row(candidate)
        if not isinstance(payload, Mapping):
            continue
        metrics = {
            str(key): number
            for key, value in payload.items()
            if str(key) != "best_epoch" and (number := _numeric(value)) is not None
        }
        if metrics:
            return metrics, candidate
    raise FileNotFoundError(f"no LARYBench regression result found under {path}")


def _extraction_metrics(path: Path) -> tuple[dict[str, float], Path]:
    candidates = [path] if path.is_file() else sorted(path.rglob("*_la_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no LARYBench extracted latent-action CSV found under {path}")
    candidate = candidates[-1]
    total = populated = 0
    with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            value = str(row.get("la_path") or "").strip()
            populated += int(bool(value) and value.lower() != "nan")
    return {
        "extraction_coverage": populated / total if total else 0.0,
        "extracted_samples": float(populated),
        "input_samples": float(total),
    }, candidate


def _infer_stage(path: Path) -> str:
    text = path.name.lower()
    if "classification" in text or "confusion" in text:
        return "classify"
    if "best_result" in text or "regression" in text:
        return "regress"
    if path.suffix.lower() == ".csv" and "_la_" in text:
        return "extract"
    if path.is_dir():
        if next(path.rglob("classification_summary.json"), None) or next(path.rglob("confusion_matrix.json"), None):
            return "classify"
        if next(path.rglob("best_result.json"), None) or next(path.rglob("best_result.csv"), None):
            return "regress"
        if next(path.rglob("*_la_*.csv"), None):
            return "extract"
    raise ValueError("cannot infer LARYBench stage; pass --stage explicitly")


def _metric_rows(
    metrics: Mapping[str, float],
    *,
    stage: str,
    source: Path,
    runtime_executed: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_id, score in metrics.items():
        lower_is_better = "mse" in metric_id or "loss" in metric_id
        unit_score = metric_id in {
            "top1_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "extraction_coverage",
        }
        rows.append(
            {
                "metric_id": metric_id,
                "name": metric_id.replace("_", " ").title(),
                "stage": stage,
                "available": True,
                "raw_score": score,
                "normalized_score": max(0.0, min(1.0, score)) if unit_score else score,
                "score": score,
                "higher_is_better": not lower_is_better,
                "source": "larybench_in_tree_runtime" if runtime_executed else "larybench_official_results",
                "source_path": str(source.resolve()),
            }
        )
    return rows


def normalize_results(
    args: argparse.Namespace,
    result_path: Path,
    *,
    runtime_executed: bool,
    runtime_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_path.expanduser().resolve()
    stage = args.stage or _infer_stage(result_path)
    loaders = {
        "extract": _extraction_metrics,
        "classify": _classification_metrics,
        "regress": _regression_metrics,
    }
    metrics, source_path = loaders[stage](result_path)
    rows = _metric_rows(
        metrics,
        stage=stage,
        source=source_path,
        runtime_executed=runtime_executed,
    )
    primary_ids = {
        "extract": ("extraction_coverage",),
        "classify": ("top1_accuracy", "macro_f1"),
        "regress": ("val_seen_mse", "val_seen_loss"),
    }[stage]
    primary = next((metric_id for metric_id in primary_ids if metric_id in metrics), None)
    normalization_ok = primary is not None
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": runtime_executed and normalization_ok,
        "leaderboard_valid": False,
        "normalizer_only": not runtime_executed,
        "normalization_ok": normalization_ok,
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_larybench_official_runner",
            "returncode": 0 if normalization_ok else 1,
            "official_runtime_executed": runtime_executed,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": "LARYBench", "stage": stage},
        "dataset": {
            "dataset": args.dataset,
            "dataset_root": None if args.dataset_root is None else str(args.dataset_root.expanduser().resolve()),
            "model": args.model,
            "split": args.split,
        },
        "metrics": {
            "leaderboard": {row["metric_id"]: row["normalized_score"] for row in rows},
            "per_metric": {row["metric_id"]: row for row in rows},
            "summary": {
                "available_metric_count": len(rows),
                "primary_metric_id": primary,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "larybench_official_in_tree" if runtime_executed else "larybench_result_normalizer",
        },
        "artifacts": {
            "scorecard": str(output_dir / "scorecard.json"),
            "raw_metric_table": str(output_dir / "raw_metric_table.jsonl"),
            "official_results_path": str(source_path),
        },
        "provenance": {
            "runtime_root": str(RUNTIME_ROOT),
            "upstream_repository": "https://github.com/meituan-longcat/LARYBench.git",
            "upstream_revision": "e7feaf1b72921ee2c34e489adb0f45faf356ecee",
            "external_repository_checkout_required": False,
        },
    }
    if args.strict and not normalization_ok:
        scorecard["run"]["status"] = "failed"
        scorecard["run"]["returncode"] = 1
    write_jsonl(output_dir / "raw_metric_table.jsonl", rows)
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "stage": stage,
            "metric_ids": [row["metric_id"] for row in rows],
            "runtime_root": str(RUNTIME_ROOT),
            "external_repository_checkout_required": False,
        },
    )
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = _resolved_layout(args)
    runtime_summary: Mapping[str, Any] | None = None
    if args.run_official:
        result_path, runtime_summary = _run_official(args, layout)
    else:
        result_path = args.official_results_path or _env_path("WORLDFOUNDRY_LARYBENCH_RESULTS_PATH")
        if result_path is None:
            raise ValueError(
                "--official-results-path, WORLDFOUNDRY_LARYBENCH_RESULTS_PATH, or --run-official is required"
            )
    scorecard = normalize_results(
        args,
        result_path,
        runtime_executed=args.run_official,
        runtime_summary=runtime_summary,
    )
    if args.json:
        print(json.dumps(scorecard, ensure_ascii=False))
    else:
        print(args.output_dir.expanduser().resolve() / "scorecard.json")
    return int(scorecard["run"]["returncode"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
