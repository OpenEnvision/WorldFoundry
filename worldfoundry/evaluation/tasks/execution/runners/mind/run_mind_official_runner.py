#!/usr/bin/env python3
"""Run or normalize the in-tree MIND runtime without an external checkout.

``--run-official`` executes the vendored official entry point
(``runtime/mind/src/process.py``) and then normalizes its result JSON.
Without ``--run-official`` the runner only normalizes an existing MIND result
document supplied through ``--official-results-path``.

The vendored runtime is treated as read-only: every mutable artifact (work
directory, upstream cache, staged videos, logs, result JSON) is written under
``--output-dir``.
"""

from __future__ import annotations


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.core.io.paths import checkpoint_root_path  # noqa: E402
from worldfoundry.core.observability.time import utc_now_iso  # noqa: E402
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (  # noqa: E402
    bundled_benchmark_asset,
)
from worldfoundry.evaluation.tasks.execution.framework.io import (  # noqa: E402
    env_path,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.execution.runners.mind.mind_metrics import (  # noqa: E402
    COMPONENT_METRIC_IDS,
    METRIC_ORDER,
    MindResultError,
    PRIMARY_METRIC_ID,
    aggregate_metrics,
    discover_result_files,
    load_result_payloads,
    metric_rows,
    missing_component_metrics,
    sample_records,
    sample_summary,
)
from worldfoundry.runtime.jobs import run_bounded_command  # noqa: E402

BENCHMARK_NAME = "MIND"
RUNNER_NAME = "benchmark_zoo_mind_official_runner"
RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_MIND_ROOT = RUNNER_ROOT / "runtime" / "mind"
FIXTURE_PATH = bundled_benchmark_asset("mind", "fixtures", "mind_result_fixture.json")

UPSTREAM_REPOSITORY = "https://github.com/CSU-JPG/MIND"
UPSTREAM_REVISION = "219f458bbfbc3204e848bb6dd1f45363d4e34730"

#: Official MIND test split: 100 + 100 shared-action-space clips and 25 + 25
#: varied-action-space clips (README "Abstract" and "Dataset Overview").
EXPECTED_OFFICIAL_SAMPLE_COUNT = 250
GT_TEST_TYPES = ("mem_test", "action_space_test")

RUNTIME_METRIC_CHOICES = ("lcm", "visual", "dino", "action", "gsc")
DEFAULT_RUNTIME_METRICS = "lcm,visual,dino,action,gsc"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or normalize MIND official outputs.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "mind"))
    parser.add_argument("--official-results-path", dest="official_results_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"),
        help="Required. May also be supplied through WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--generated-artifact-dir",
        "--generated-video-dir",
        dest="generated_artifact_dir",
        type=Path,
        help="Model output root laid out as <model>/{1st,3rd}_data/<test_type>/<name>/video.mp4.",
    )
    parser.add_argument("--run-official", action="store_true", help="Execute the vendored MIND runtime.")
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Normalize the bundled synthetic fixture instead of a real MIND result file.",
    )
    parser.add_argument("--mind-root", type=Path, help="Override the checked-in MIND runtime root.")
    parser.add_argument("--gt-root", type=Path, help="MIND-Data ground-truth root passed to --gt_root.")
    parser.add_argument("--weights-dir", type=Path, help="Root holding MIND metric weights (never in-tree).")
    parser.add_argument("--dino-path", type=Path, help="DINOv3 weight directory passed to --dino_path.")
    parser.add_argument(
        "--vipe-repo",
        type=Path,
        help="ViPE checkout used by the action metric; linked into the work dir as ./vipe.",
    )
    parser.add_argument("--work-dir", type=Path, help="Mutable work directory; must sit under --output-dir.")
    parser.add_argument(
        "--metrics",
        default=os.environ.get("WORLDFOUNDRY_MIND_METRICS", DEFAULT_RUNTIME_METRICS),
        help=f"Comma-separated MIND metric selection from {','.join(RUNTIME_METRIC_CHOICES)}.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--video-max-time", type=int)
    parser.add_argument("--python-executable", help="Interpreter used for the vendored runtime.")
    parser.add_argument("--timeout", type=int, default=24 * 60 * 60)
    parser.add_argument("--json", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Path resolution and isolation guards
# ---------------------------------------------------------------------------


def resolve_mind_root(args: argparse.Namespace) -> Path:
    explicit = args.mind_root or env_path("WORLDFOUNDRY_MIND_ROOT")
    return (explicit or DEFAULT_MIND_ROOT).expanduser().resolve()


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_output_dir(args: argparse.Namespace, *, mind_root: Path) -> Path:
    if args.output_dir is None:
        raise ValueError("--output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if _is_inside(output_dir, mind_root):
        raise ValueError(f"--output-dir must not write into the in-tree MIND runtime: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_weights_dir(args: argparse.Namespace, *, mind_root: Path) -> Path:
    explicit = args.weights_dir or env_path("WORLDFOUNDRY_MIND_WEIGHTS_DIR")
    weights_dir = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else checkpoint_root_path("mind", specific_env="WORLDFOUNDRY_MIND_WEIGHTS_DIR").expanduser().resolve()
    )
    if _is_inside(weights_dir, mind_root):
        raise ValueError(f"MIND weights must not resolve inside the in-tree runtime: {weights_dir}")
    return weights_dir


def resolve_work_dir(args: argparse.Namespace, *, output_dir: Path, mind_root: Path) -> Path:
    work_dir = (args.work_dir or (output_dir / "mind_work")).expanduser().resolve()
    if work_dir != output_dir and not _is_inside(work_dir, output_dir):
        raise ValueError(f"--work-dir must be inside --output-dir: {work_dir}")
    if _is_inside(work_dir, mind_root):
        raise ValueError(f"MIND work directory must not modify the in-tree runtime: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def resolve_results_path(args: argparse.Namespace) -> Path:
    if args.run_fixture:
        return FIXTURE_PATH
    candidates = (
        args.official_results_path,
        env_path("WORLDFOUNDRY_MIND_RESULTS_PATH"),
    )
    for candidate in candidates:
        if candidate is not None:
            return Path(candidate).expanduser()
    raise ValueError(
        "--official-results-path, WORLDFOUNDRY_MIND_RESULTS_PATH, --run-fixture, or --run-official is required"
    )


def resolve_metric_selection(value: str | None) -> tuple[str, ...]:
    requested = tuple(item.strip().lower() for item in str(value or "").split(",") if item.strip())
    unknown = [item for item in requested if item not in RUNTIME_METRIC_CHOICES]
    if unknown:
        raise ValueError(f"unknown MIND metrics {unknown}; choose from {list(RUNTIME_METRIC_CHOICES)}")
    return requested or RUNTIME_METRIC_CHOICES


# ---------------------------------------------------------------------------
# Official runtime execution
# ---------------------------------------------------------------------------


def build_runtime_command(
    args: argparse.Namespace,
    *,
    mind_root: Path,
    gt_root: Path,
    test_root: Path,
    dino_path: Path,
    result_path: Path,
    metrics: Sequence[str],
) -> list[str]:
    command = [
        args.python_executable or sys.executable,
        str(mind_root / "src" / "process.py"),
        "--gt_root",
        str(gt_root),
        "--test_root",
        str(test_root),
        "--dino_path",
        str(dino_path),
        "--num_gpus",
        str(max(1, int(args.num_gpus))),
        "--metrics",
        ",".join(metrics),
        "--output",
        str(result_path),
    ]
    if args.video_max_time is not None:
        command.extend(["--video_max_time", str(args.video_max_time)])
    return command


def _link_vipe_repo(work_dir: Path, vipe_repo: Path | None) -> str | None:
    """Expose a caller-supplied ViPE checkout as ``./vipe`` inside the work dir.

    ``runtime/mind/src/utils/vipe_utils.py`` calls ``vipe_to_colmap(out_dir,
    Path("vipe"))``, which resolves ``vipe`` against the process working
    directory. Linking instead of patching keeps the vendored tree untouched.
    """
    if vipe_repo is None:
        return None
    source = Path(vipe_repo).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"--vipe-repo is not a directory: {source}")
    target = work_dir / "vipe"
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)
    return str(target)


def run_official_mind(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Path, Path]:
    mind_root = resolve_mind_root(args)
    entrypoint = mind_root / "src" / "process.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"missing in-tree MIND runtime: {entrypoint}")
    output_dir = resolve_output_dir(args, mind_root=mind_root)
    work_dir = resolve_work_dir(args, output_dir=output_dir, mind_root=mind_root)
    weights_dir = resolve_weights_dir(args, mind_root=mind_root)
    metrics = resolve_metric_selection(args.metrics)

    gt_root = args.gt_root or env_path("WORLDFOUNDRY_MIND_GT_ROOT")
    if gt_root is None:
        raise ValueError("--gt-root or WORLDFOUNDRY_MIND_GT_ROOT is required for --run-official")
    gt_root = Path(gt_root).expanduser().resolve()
    if not gt_root.is_dir():
        raise FileNotFoundError(f"MIND ground-truth root is missing: {gt_root}")

    test_root = args.generated_artifact_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if test_root is None:
        raise ValueError(
            "--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official"
        )
    test_root = Path(test_root).expanduser().resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"generated artifact directory is missing: {test_root}")
    args.generated_artifact_dir = test_root

    dino_path = (args.dino_path or env_path("WORLDFOUNDRY_MIND_DINO_PATH") or weights_dir / "dinov3_vitb16")
    dino_path = Path(dino_path).expanduser().resolve()
    if _is_inside(dino_path, mind_root):
        raise ValueError(f"DINOv3 weights must not resolve inside the in-tree runtime: {dino_path}")

    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "mind_result.json"
    vipe_link = _link_vipe_repo(work_dir, args.vipe_repo or env_path("WORLDFOUNDRY_MIND_VIPE_REPO"))

    command = build_runtime_command(
        args,
        mind_root=mind_root,
        gt_root=gt_root,
        test_root=test_root,
        dino_path=dino_path,
        result_path=result_path,
        metrics=metrics,
    )
    env = {
        "MIND_CACHE_DIR": str(cache_dir),
        "PYTHONPATH": os.pathsep.join(
            path for path in (str(mind_root / "src"), str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if path
        ),
    }
    started_at = utc_now_iso()
    result = run_bounded_command(command, cwd=work_dir, env=env, timeout=int(args.timeout))
    log_path = output_dir / "official_runtime.log"
    log_path.write_text(
        "$ {command}\n\n[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n".format(
            command=" ".join(command),
            stdout=result.get("stdout") or "",
            stderr=result.get("stderr") or "",
        ),
        encoding="utf-8",
    )
    runtime_summary = {
        "command": command,
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out")),
        "started_at": started_at,
        "log_path": str(log_path),
        "runtime_root": str(mind_root),
        "work_dir": str(work_dir),
        "cache_dir": str(cache_dir),
        "gt_root": str(gt_root),
        "test_root": str(test_root),
        "dino_path": str(dino_path),
        "weights_dir": str(weights_dir),
        "vipe_repo_link": vipe_link,
        "requested_metrics": list(metrics),
        "result_path": str(result_path),
    }
    write_json(output_dir / "runtime_summary.json", runtime_summary)
    if result.get("returncode") != 0:
        raise RuntimeError(
            f"MIND official runtime failed with exit code {result.get('returncode')}; see {log_path}"
        )
    if not result_path.is_file():
        raise FileNotFoundError(f"MIND runtime completed but wrote no result document: {result_path}")
    return result_path, runtime_summary, output_dir, mind_root


# ---------------------------------------------------------------------------
# Scorecard construction
# ---------------------------------------------------------------------------


def build_scorecard(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mind_root: Path,
    results_path: Path,
    official_runtime_executed: bool,
    runtime_summary: Mapping[str, Any] | None = None,
    fixture_only: bool = False,
) -> dict[str, Any]:
    result_files = discover_result_files(results_path)
    payloads = load_result_payloads(result_files)
    records = sample_records(payloads)
    metrics = aggregate_metrics(records)
    summary = sample_summary(records)
    source = "mind_official_runtime" if official_runtime_executed else "mind_results_file"
    rows = metric_rows(metrics, source=source, source_path=results_path)
    available = [row for row in rows if row["available"]]

    missing_components = missing_component_metrics(metrics)
    gt_sample_count = sum(
        1 for record in records if str(record.get("test_type") or "") in GT_TEST_TYPES
    )
    comparability_blockers: list[str] = []
    if fixture_only:
        comparability_blockers.append(
            "scores come from the bundled synthetic MIND fixture, not from a MIND evaluation"
        )
    if not official_runtime_executed:
        comparability_blockers.append("official MIND runtime was not executed by this run")
    if missing_components:
        comparability_blockers.append(
            f"official metric coverage is partial ({len(missing_components)} component metrics missing)"
        )
    if gt_sample_count < EXPECTED_OFFICIAL_SAMPLE_COUNT:
        comparability_blockers.append(
            "official test coverage is partial "
            f"({gt_sample_count}/{EXPECTED_OFFICIAL_SAMPLE_COUNT} mem_test + action_space_test clips)"
        )
    if summary["failed_sample_count"]:
        comparability_blockers.append(
            f"{summary['failed_sample_count']} MIND samples reported an upstream error"
        )

    official_verified = official_runtime_executed and not comparability_blockers and bool(available)
    leaderboard_blockers = list(comparability_blockers)
    leaderboard_blockers.append(
        "MIND publishes no official composite score or leaderboard at the pinned revision; "
        "mind_average is a WorldFoundry-derived aggregate"
    )
    leaderboard_blockers.append(
        "official submission packaging and dataset provenance were not validated by this run"
    )

    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_path = output_dir / "per_sample_scores.jsonl"
    contract_path = output_dir / "benchmark_contract.json"

    write_jsonl(raw_metric_table_path, rows)
    write_jsonl(per_sample_path, records)
    write_json(
        contract_path,
        {
            "benchmark_id": args.benchmark_id,
            "name": BENCHMARK_NAME,
            "metric_ids": list(METRIC_ORDER),
            "component_metric_ids": list(COMPONENT_METRIC_IDS),
            "primary_metric": PRIMARY_METRIC_ID,
            "runtime_root": str(mind_root),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "external_repository_checkout_required": False,
        },
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
            "runner": RUNNER_NAME,
            "returncode": 0 if available else 1,
            "runtime_summary": dict(runtime_summary or {}),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": BENCHMARK_NAME},
        "metrics": {
            "leaderboard": {
                row["metric_id"]: row["normalized_score"] for row in rows if row["available"]
            },
            "per_metric": {row["metric_id"]: row for row in rows},
            "summary": {
                "sample_count": summary["sample_count"],
                "metric_count": len(METRIC_ORDER),
                "available_metrics": len(available),
                "failed_metrics": len(METRIC_ORDER) - len(available),
                "primary_metric": PRIMARY_METRIC_ID,
                "primary_score": metrics.get(PRIMARY_METRIC_ID, {}).get("normalized_score"),
                **summary,
            },
        },
        "evaluation": {
            "kind": "mind_official_in_tree" if official_runtime_executed else "mind_result_normalizer",
            "available_metric_count": len(available),
            "declared_metric_count": len(METRIC_ORDER),
            "official_runtime_executed": official_runtime_executed,
            "benchmark_comparable": official_verified,
            "comparability_blockers": comparability_blockers,
            "missing_component_metrics": missing_components,
            "fixture_only": fixture_only,
        },
        "dataset": {
            "generated_artifact_dir": (
                None
                if args.generated_artifact_dir is None
                else str(Path(args.generated_artifact_dir).expanduser().resolve())
            ),
            "gt_root": None if args.gt_root is None else str(Path(args.gt_root).expanduser().resolve()),
            "results_path": str(results_path),
            "result_files": [str(path) for path in result_files],
        },
        "provenance": {
            "runtime_root": str(mind_root),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "external_repository_checkout_required": False,
        },
        "artifacts": {
            "scorecard": str(scorecard_path),
            "raw_metric_table": str(raw_metric_table_path),
            "per_sample_scores": str(per_sample_path),
            "benchmark_contract": str(contract_path),
            "official_results_path": str(results_path),
        },
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def normalize_mind_results(
    args: argparse.Namespace,
    *,
    official_runtime_executed: bool = False,
    runtime_summary: Mapping[str, Any] | None = None,
    results_path: Path | None = None,
    output_dir: Path | None = None,
    mind_root: Path | None = None,
) -> dict[str, Any]:
    mind_root = mind_root or resolve_mind_root(args)
    output_dir = output_dir or resolve_output_dir(args, mind_root=mind_root)
    resolved_results = (results_path or resolve_results_path(args)).expanduser().resolve()
    return build_scorecard(
        args,
        output_dir=output_dir,
        mind_root=mind_root,
        results_path=resolved_results,
        official_runtime_executed=official_runtime_executed,
        runtime_summary=runtime_summary,
        fixture_only=bool(args.run_fixture) and not official_runtime_executed,
    )


def failure_scorecard(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "leaderboard_blockers": [str(error)],
        "normalizer_only": not bool(args.run_official),
        "normalization_ok": False,
        "run": {
            "status": "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 1,
            "error": f"{type(error).__name__}: {error}",
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": BENCHMARK_NAME},
        "metrics": {
            "leaderboard": {},
            "per_metric": {},
            "summary": {
                "sample_count": 0,
                "metric_count": len(METRIC_ORDER),
                "available_metrics": 0,
                "failed_metrics": len(METRIC_ORDER),
                "primary_metric": PRIMARY_METRIC_ID,
                "primary_score": None,
            },
        },
        "evaluation": {
            "kind": "mind_official_in_tree" if args.run_official else "mind_result_normalizer",
            "available_metric_count": 0,
            "declared_metric_count": len(METRIC_ORDER),
            "official_runtime_executed": False,
            "benchmark_comparable": False,
            "comparability_blockers": [str(error)],
        },
        "artifacts": {},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.run_official:
            results_path, runtime_summary, output_dir, mind_root = run_official_mind(args)
            scorecard = normalize_mind_results(
                args,
                official_runtime_executed=True,
                runtime_summary=runtime_summary,
                results_path=results_path,
                output_dir=output_dir,
                mind_root=mind_root,
            )
        else:
            scorecard = normalize_mind_results(args)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - always emit a scorecard
        scorecard = failure_scorecard(args, exc)
        if args.output_dir is not None:
            output_dir = Path(args.output_dir).expanduser().resolve()
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                scorecard_path = output_dir / "scorecard.json"
                scorecard["artifacts"] = {"scorecard": str(scorecard_path)}
                write_json(scorecard_path, scorecard)
            except OSError as write_error:
                print(f"mind: could not write failure scorecard: {write_error}", file=sys.stderr)
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "scorecard": scorecard}, indent=2, ensure_ascii=False))
        else:
            print(f"mind: failed: {exc}", file=sys.stderr)
        return 1
    ok = scorecard.get("normalization_ok") is True
    if args.json:
        print(json.dumps({"ok": ok, "scorecard": scorecard}, indent=2, ensure_ascii=False))
    else:
        print(f"mind: normalized {scorecard['evaluation']['available_metric_count']} metrics")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
