#!/usr/bin/env python3
"""Run the in-tree WorldOlympiad runtime or normalize its official judge outputs.

WorldOlympiad scores video world models on three tracks — physical faithfulness,
geometric consistency, and interaction fidelity — and writes one judge JSON per
case and pipeline. This runner either drives the official batch evaluator that is
checked in under ``runtime/worldolympiad`` (``--run-official``) or normalizes
judge outputs that already exist (``--official-results-path``).

Weights, the Depth Anything 3 source tree, and the batch working state all live
outside the checked-in runtime, so the vendored source stays read-only.
"""

from __future__ import annotations


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.core.io.paths import checkpoint_root_path  # noqa: E402
from worldfoundry.core.observability.time import utc_now_iso  # noqa: E402
from worldfoundry.evaluation.reporting.scorecard import SCORECARD_SCHEMA_VERSION  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, write_json, write_jsonl  # noqa: E402
from worldfoundry.evaluation.tasks.execution.runners.worldolympiad.base_model_resolver import (  # noqa: E402
    ResolvedBaseModels,
    resolve_all,
)
from worldfoundry.evaluation.tasks.execution.runners.worldolympiad.worldolympiad_metrics import (  # noqa: E402
    METRIC_ORDER,
    PRIMARY_METRIC,
    TRACK_METRICS,
    TRACKS,
    iter_case_metric_rows,
    metric_rows,
    normalize_results,
    track_summary,
)
from worldfoundry.runtime.jobs import run_bounded_command  # noqa: E402

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = RUNNER_ROOT / "runtime" / "worldolympiad"
RUNTIME_ENTRYPOINT = Path("batch_test") / "evaluate_pipelines.py"
FIXTURE_ROOT = REPO_ROOT / "worldfoundry/data/benchmarks/assets/worldolympiad"
DEFAULT_DOMAINS = ("general", "gaming", "embodied")
SCORING_TRACKS = ("physical", "geometry", "interaction")
# Headline score of each track plus the aggregate; everything else is diagnostic.
REQUIRED_METRICS = ("combined_score", "physical_score", "three_d_score", "interaction_score")
RUNNER_NAME = "benchmark_zoo_worldolympiad_official_runner"

# Upstream resolves these relative to its own root, which would write weights into
# the checked-in runtime. Each value is sourced from ``worldfoundry.base_models``
# when the registered asset is staged, and falls back to ``--weights-dir`` otherwise,
# so the vendored runtime stays byte-identical and an operator need not stage a
# separate weights directory or an external Depth-Anything-3 checkout.
WEIGHT_ARGS: tuple[tuple[str, str], ...] = (
    ("--vlm-model", "QwenVL"),
    ("--interaction-vlm-model", "QwenVL"),
    ("--three-d-scoring-model", "QwenVL"),
    ("--three-d-model-name", "da3"),
    ("--sam3-model", "sam3/sam3.pt"),
    ("--clip-download-root", "clip"),
)


def _resolve_weight_value(
    relative: str, *, resolved: ResolvedBaseModels, weights_dir: Path
) -> str:
    """Pick each weight path from base_models when staged, else --weights-dir.

    Precedence per model: an explicit per-model env var (``QWENVL_MODEL_PATH``,
    ``SAM3_MODEL``, ``CLIP_DOWNLOAD_ROOT``) wins via the capability registry's
    ``check``; then the staged base_models asset; then the bulk ``--weights-dir``
    fallback. CLIP has no staged weight asset, so its managed cache (or
    ``CLIP_DOWNLOAD_ROOT``) is always used and ViT-B/32 auto-downloads on first use.
    """

    if relative == "QwenVL" and resolved.qwenvl_dir is not None:
        return str(resolved.qwenvl_dir)
    if relative == "da3" and resolved.da3_weights_dir is not None:
        return str(resolved.da3_weights_dir)
    if relative == "sam3/sam3.pt" and resolved.sam3_path is not None:
        return str(resolved.sam3_path)
    if relative == "clip":
        return str(resolved.clip_cache_dir)
    return str(weights_dir / relative)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "worldolympiad"))
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument(
        "--generated-artifact-dir",
        "--generated-video-dir",
        dest="generated_artifact_dir",
        type=Path,
        help="Case root laid out as <domain>/<case_id>/{prompt.json,ref_*.mp4,<prefix>_gen_<case_id>.mp4}.",
    )
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--run-fixture", action="store_true")
    parser.add_argument(
        "--worldolympiad-root",
        type=Path,
        help="Override the checked-in runtime with an external WorldOlympiad checkout.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Directory holding QwenVL/, da3/, sam3/sam3.pt, and clip/. Never inside the runtime tree.",
    )
    parser.add_argument(
        "--da3-src",
        type=Path,
        help="Depth Anything 3 'src' directory added to PYTHONPATH for the geometry track.",
    )
    parser.add_argument("--domains", nargs="*", help="Domain directories under the case root.")
    parser.add_argument("--domain-name", help="Domain label when the case root is a single custom directory.")
    parser.add_argument("--pipelines", nargs="*", help="Upstream pipeline aliases whose outputs are scored.")
    parser.add_argument("--limit", type=int, help="Maximum pending cases per domain and pipeline.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-slots", default=os.environ.get("WORLDFOUNDRY_WORLDOLYMPIAD_GPU_SLOTS", "0"))
    parser.add_argument("--qwen-server-urls", default=os.environ.get("WORLDFOUNDRY_WORLDOLYMPIAD_QWEN_URLS", ""))
    parser.add_argument("--sam3-server-urls", default=os.environ.get("WORLDFOUNDRY_WORLDOLYMPIAD_SAM3_URLS", ""))
    parser.add_argument(
        "--reward-3d-server-urls", default=os.environ.get("WORLDFOUNDRY_WORLDOLYMPIAD_REWARD_3D_URLS", "")
    )
    parser.add_argument("--skip-physical", action="store_true")
    parser.add_argument("--skip-interaction", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--no-clip-interaction", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute cases whose judge JSON already exists.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the official batch without scoring; validates the case layout with no GPU or weights.",
    )
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all three triathlon tracks to be present for a successful run.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def resolve_runtime_root(args: argparse.Namespace) -> Path:
    """Resolve the official evaluator tree, defaulting to the checked-in runtime."""

    override = args.worldolympiad_root or env_path("WORLDFOUNDRY_WORLDOLYMPIAD_ROOT")
    root = (override or DEFAULT_RUNTIME_ROOT).expanduser().resolve()
    if not (root / RUNTIME_ENTRYPOINT).is_file():
        raise FileNotFoundError(f"WorldOlympiad runtime is missing {RUNTIME_ENTRYPOINT}: {root}")
    return root


def resolve_weights_dir(args: argparse.Namespace, *, runtime_root: Path) -> Path:
    """Resolve the weights root, refusing any location inside the runtime tree."""

    weights_dir = (
        args.weights_dir
        or env_path("WORLDFOUNDRY_WORLDOLYMPIAD_WEIGHTS_DIR")
        or checkpoint_root_path("worldolympiad", specific_env="WORLDFOUNDRY_WORLDOLYMPIAD_WEIGHTS_DIR")
    )
    weights_dir = Path(weights_dir).expanduser().resolve()
    if weights_dir == runtime_root or runtime_root in weights_dir.parents:
        raise ValueError(f"--weights-dir must not write into the checked-in runtime: {weights_dir}")
    return weights_dir


def resolve_da3_src(args: argparse.Namespace) -> Path | None:
    da3_src = args.da3_src or env_path("WORLDFOUNDRY_WORLDOLYMPIAD_DA3_SRC", "WORLDFOUNDRY_DA3_SRC")
    return None if da3_src is None else da3_src.expanduser().resolve()


def resolve_isolated_output_dir(output_dir: Path, *, runtime_root: Path) -> Path:
    """Keep every mutable batch artifact outside the checked-in runtime."""

    if output_dir == runtime_root or runtime_root in output_dir.parents:
        raise ValueError(f"--output-dir must not write into the checked-in runtime: {output_dir}")
    return output_dir


def resolve_case_root(args: argparse.Namespace) -> Path:
    case_root = args.generated_artifact_dir or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    if case_root is None:
        raise ValueError(
            "--generated-artifact-dir or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required for --run-official"
        )
    case_root = case_root.expanduser().resolve()
    if not case_root.is_dir():
        raise FileNotFoundError(f"WorldOlympiad case root does not exist: {case_root}")
    return case_root


def _domain_layout(case_root: Path, args: argparse.Namespace) -> tuple[list[str], bool]:
    """Return the domain names to evaluate and whether the case root is custom.

    A custom root holds case directories directly; the default layout groups them
    under one directory per domain.
    """

    if args.domain_name:
        return [args.domain_name], True
    requested = [name for value in (args.domains or ()) for name in str(value).split(",") if name]
    if requested:
        return requested, False
    present = [domain for domain in DEFAULT_DOMAINS if (case_root / domain).is_dir()]
    if present:
        return present, False
    return ["custom"], True


def build_official_command(
    args: argparse.Namespace,
    *,
    root: Path,
    case_root: Path,
    output_dir: Path,
    weights_dir: Path,
    resolved: ResolvedBaseModels,
) -> list[str]:
    domains, custom_root = _domain_layout(case_root, args)
    python = os.environ.get("WORLDFOUNDRY_WORLDOLYMPIAD_PYTHON") or os.environ.get(
        "WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable
    )
    command = [
        python,
        str(root / RUNTIME_ENTRYPOINT),
        "--worldeval-root",
        str(root),
        "--project-root",
        str(root),
        "--python-bin",
        python,
        # Keep every mutable batch artifact inside the requested output directory.
        "--manifest-dir",
        str(output_dir / "batch_manifests"),
        "--log-root",
        str(output_dir / "batch_logs"),
        "--gpu-slots",
        str(args.gpu_slots),
        "--workers",
        str(args.workers),
    ]
    for flag, relative in WEIGHT_ARGS:
        command.extend(
            [flag, _resolve_weight_value(relative, resolved=resolved, weights_dir=weights_dir)]
        )
    if custom_root:
        command.extend(["--root", str(case_root), "--domain-name", domains[0]])
    else:
        command.extend(["--output-root", str(case_root), "--domains", *domains])
    if args.pipelines:
        command.extend(["--pipelines", *[str(value) for value in args.pipelines]])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    for flag, value in (
        ("--qwen-server-urls", args.qwen_server_urls),
        ("--sam3-server-urls", args.sam3_server_urls),
        ("--reward-3d-server-urls", args.reward_3d_server_urls),
    ):
        if value:
            command.extend([flag, str(value)])
    if args.no_clip_interaction:
        command.append("--no-run-clip-interaction")
    else:
        command.append("--run-clip-interaction")
    for flag, enabled in (
        ("--skip-physical", args.skip_physical),
        ("--skip-interaction", args.skip_interaction),
        ("--skip-3d", args.skip_3d),
        ("--force", args.force),
        ("--dry-run", args.dry_run),
    ):
        if enabled:
            command.append(flag)
    return command


def runtime_env(
    *, root: Path, da3_src: Path | None, resolved: ResolvedBaseModels | None = None
) -> dict[str, str]:
    """Expose the runtime tree plus base_models code on PYTHONPATH.

    The vendored modules insert their own root into ``sys.path``. The base_models
    ``openai_clip_runtime`` (so ``import clip`` resolves there) and the DA3 name
    shim (so ``import depth_anything_3`` resolves into the vendored
    ``depth_anything_v3`` package) are prepended; an explicit ``--da3-src`` always
    wins. Model-weight env vars resolved from base_models are exported for the
    subprocess unless the operator already set them.
    """

    entries: list[str] = []
    if da3_src is not None:
        entries.append(str(da3_src))
    if resolved is not None:
        entries.extend(str(path) for path in resolved.pythonpath_entries())
    entries.extend([str(REPO_ROOT), str(root), os.environ.get("PYTHONPATH", "")])
    # Dedup while preserving order; the DA3 shim can arrive both as effective_da3_src
    # and as an entry of resolved.pythonpath_entries().
    seen: set[str] = set()
    unique = [entry for entry in entries if entry and not (entry in seen or seen.add(entry))]
    env: dict[str, str] = {"PYTHONPATH": os.pathsep.join(unique)}
    if resolved is not None:
        for name, value in resolved.env_overrides().items():
            if name not in os.environ:
                env[name] = value
    return env


def run_official(args: argparse.Namespace, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    root = resolve_runtime_root(args)
    resolve_isolated_output_dir(output_dir, runtime_root=root)
    case_root = resolve_case_root(args)
    weights_dir = resolve_weights_dir(args, runtime_root=root)
    da3_src = resolve_da3_src(args)
    resolved = resolve_all(shim_parent=output_dir)
    effective_da3_src = da3_src or resolved.da3_code_shim_dir
    command = build_official_command(
        args,
        root=root,
        case_root=case_root,
        output_dir=output_dir,
        weights_dir=weights_dir,
        resolved=resolved,
    )
    started = utc_now_iso()
    execution = run_bounded_command(
        command,
        cwd=root,
        env=runtime_env(root=root, da3_src=effective_da3_src, resolved=resolved),
        timeout=args.timeout,
    )
    log_path = output_dir / "official_runtime.log"
    log_path.write_text(
        f"STDOUT\n{execution.get('stdout', '')}\n\nSTDERR\n{execution.get('stderr', '')}", encoding="utf-8"
    )
    runtime = {
        "command": command,
        "returncode": execution.get("returncode"),
        "started_at": started,
        "timed_out": execution.get("timed_out", False),
        "log_path": str(log_path.resolve()),
        "runtime_root": str(root),
        "in_tree_runtime": root == DEFAULT_RUNTIME_ROOT,
        "weights_dir": str(weights_dir),
        "da3_src": None if effective_da3_src is None else str(effective_da3_src),
        "base_models_resolved": resolved.as_provenance(),
        "case_root": str(case_root),
        "skipped_tracks": [
            track
            for track, skipped in (
                ("physical", args.skip_physical),
                ("interaction", args.skip_interaction),
                ("geometry", args.skip_3d),
            )
            if skipped
        ],
    }
    if execution.get("returncode") != 0:
        raise RuntimeError(
            f"WorldOlympiad official runtime exited with {execution.get('returncode')}; see {log_path}"
        )
    return case_root, runtime


def resolve_results_path(args: argparse.Namespace) -> Path:
    if args.run_fixture:
        if not FIXTURE_ROOT.is_dir():
            raise FileNotFoundError(f"WorldOlympiad fixture is missing: {FIXTURE_ROOT}")
        return FIXTURE_ROOT
    candidate = (
        args.official_results_path
        or env_path("WORLDFOUNDRY_WORLDOLYMPIAD_RESULTS_PATH")
        or args.generated_artifact_dir
        or env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR")
    )
    if candidate is None:
        raise ValueError(
            "--official-results-path or WORLDFOUNDRY_WORLDOLYMPIAD_RESULTS_PATH is required; "
            "pass the case root, a judge JSON, a scheduler summary.jsonl, or a summarize_scores aggregate"
        )
    return candidate.expanduser().resolve()


def build_scorecard(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    results_path: Path,
    normalized: Mapping[str, Any],
    runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scores = normalized["scores"]
    official_runtime_executed = runtime is not None
    source = "worldolympiad_official_runtime" if official_runtime_executed else "worldolympiad_judge_results"
    rows = metric_rows(scores, source=source, source_path=str(results_path))
    available = [row for row in rows if row["available"]]
    tracks = track_summary(scores)
    covered_tracks = [track for track in SCORING_TRACKS if tracks[track]["available"]]
    full_triathlon = len(covered_tracks) == len(SCORING_TRACKS)

    blockers: list[str] = []
    if not official_runtime_executed:
        blockers.append("the official WorldOlympiad runtime was not executed by this run")
    missing_tracks = [track for track in SCORING_TRACKS if track not in covered_tracks]
    if missing_tracks:
        blockers.append(f"triathlon coverage is partial; missing tracks: {', '.join(missing_tracks)}")
    # Per-dimension physical metrics are legitimately absent when a case asks no
    # question of that dimension, so only the headline scores gate comparability.
    missing_required = [metric for metric in REQUIRED_METRICS if scores.get(metric) is None]
    if missing_required and not missing_tracks:
        blockers.append(f"required metrics are missing: {', '.join(missing_required)}")
    if normalized["kind"] == "summarize_scores_aggregate":
        blockers.append("scores were imported from a summarize_scores aggregate without per-case evidence")
    missing_metrics = [row["metric_id"] for row in rows if not row["available"]]

    strict_ok = not args.strict or full_triathlon
    normalization_ok = bool(available) and strict_ok
    official_verified = official_runtime_executed and bool(available) and not blockers
    case_records = list(normalized["case_records"])

    scorecard_path = output_dir / "scorecard.json"
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": official_verified,
        "integration_evidence": official_runtime_executed and bool(available),
        "leaderboard_valid": False,
        "leaderboard_blockers": [
            *blockers,
            "leaderboard parity requires the full official 1,000-video split and the official judge configuration",
        ],
        "normalizer_only": not official_runtime_executed,
        "normalization_ok": normalization_ok,
        "run": {
            "status": "succeeded" if normalization_ok else "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 0 if normalization_ok else 1,
            "official_runtime": dict(runtime or {}),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": "WorldOlympiad"},
        "metrics": {
            "leaderboard": {
                row["metric_id"]: row["score"] for row in available if row["score"] is not None
            },
            "per_metric": {row["metric_id"]: row for row in rows},
            "tracks": {track: list(TRACK_METRICS[track]) for track in TRACKS},
            "primary_metric": PRIMARY_METRIC,
            "summary": {
                "sample_count": normalized["case_count"],
                "metric_count": len(METRIC_ORDER),
                "available_metrics": len(available),
                "failed_metrics": len(METRIC_ORDER) - len(available),
            },
        },
        "evaluation": {
            "kind": "worldolympiad_official_in_tree"
            if official_runtime_executed
            else "worldolympiad_result_normalizer",
            "available_metric_count": len(available),
            "declared_metric_count": len(METRIC_ORDER),
            "official_runtime_executed": official_runtime_executed,
            "benchmark_comparable": official_verified,
            "result_kind": normalized["kind"],
            "tracks": tracks,
            "covered_tracks": covered_tracks,
            "full_triathlon": full_triathlon,
            "missing_metrics": missing_metrics,
            "comparability_blockers": blockers,
        },
        "dataset": {
            "case_count": normalized["case_count"],
            "results_path": str(results_path),
            "generated_artifact_dir": None
            if args.generated_artifact_dir is None
            else str(Path(args.generated_artifact_dir).expanduser().resolve()),
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str((output_dir / "raw_metric_table.jsonl").resolve()),
            "per_case_metrics": str((output_dir / "per_case_metrics.jsonl").resolve()),
            "official_results_path": str(results_path.resolve()),
        },
        "notes": [
            "The official physical, geometry, and interaction evaluator is checked in under runtime/worldolympiad.",
            "Model weights, the Depth Anything 3 source tree, and the official videos are resolved outside that tree.",
        ],
    }
    write_jsonl(output_dir / "raw_metric_table.jsonl", rows)
    write_jsonl(output_dir / "per_case_metrics.jsonl", list(iter_case_metric_rows(case_records)))
    write_json(
        output_dir / "benchmark_contract.json",
        {
            "benchmark_id": args.benchmark_id,
            "metric_ids": list(METRIC_ORDER),
            "primary_metric": PRIMARY_METRIC,
            "tracks": {track: list(TRACK_METRICS[track]) for track in TRACKS},
        },
    )
    write_json(scorecard_path, scorecard)
    return scorecard


def failure_scorecard(args: argparse.Namespace, output_dir: Path, error: Exception) -> dict[str, Any]:
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "leaderboard_blockers": [str(error)],
        "normalizer_only": not args.run_official,
        "normalization_ok": False,
        "run": {
            "status": "failed",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 1,
            "error": f"{type(error).__name__}: {error}",
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": "WorldOlympiad"},
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
        "artifacts": {"scorecard": str((output_dir / "scorecard.json").resolve())},
    }
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def dry_run_scorecard(args: argparse.Namespace, output_dir: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Report a planning-only run, which validates the case layout but scores nothing."""

    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "leaderboard_valid": False,
        "leaderboard_blockers": ["--dry-run plans the official batch without scoring any case"],
        "normalizer_only": False,
        "normalization_ok": False,
        "run": {
            "status": "succeeded",
            "started_at": utc_now_iso(),
            "runner": RUNNER_NAME,
            "returncode": 0,
            "official_runtime": dict(runtime),
        },
        "benchmark": {"benchmark_id": args.benchmark_id, "name": "WorldOlympiad"},
        "metrics": {"leaderboard": {}, "per_metric": {}, "summary": {"sample_count": 0}},
        "evaluation": {
            "kind": "worldolympiad_official_in_tree_dry_run",
            "official_runtime_executed": True,
            "scored": False,
        },
        "artifacts": {"scorecard": str((output_dir / "scorecard.json").resolve())},
    }
    write_json(output_dir / "scorecard.json", scorecard)
    return scorecard


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_run = args.dry_run and args.run_official
    try:
        runtime: dict[str, Any] | None = None
        if args.run_official:
            results_path, runtime = run_official(args, output_dir)
            if dry_run:
                scorecard = dry_run_scorecard(args, output_dir, runtime)
                print(f"worldolympiad: planned the official batch (dry run); see {runtime['log_path']}")
                return 0
        else:
            results_path = resolve_results_path(args)
        normalized = normalize_results(results_path)
        scorecard = build_scorecard(
            args, output_dir=output_dir, results_path=results_path, normalized=normalized, runtime=runtime
        )
    except Exception as exc:  # noqa: BLE001 - failures are reported as a scorecard, not a traceback
        scorecard = failure_scorecard(args, output_dir, exc)
    ok = scorecard["normalization_ok"] is True
    if args.json:
        print(json.dumps({"ok": ok, "output_dir": str(output_dir), "scorecard": scorecard}, ensure_ascii=False, indent=2))
    elif ok:
        evaluation = scorecard["evaluation"]
        print(
            f"worldolympiad: {evaluation['kind']} "
            f"({evaluation['available_metric_count']}/{evaluation['declared_metric_count']} metrics, "
            f"{scorecard['metrics']['summary']['sample_count']} cases, "
            f"tracks: {', '.join(evaluation['covered_tracks']) or 'none'})"
        )
    else:
        print(f"worldolympiad: failed ({scorecard['run'].get('error', 'incomplete results')})", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
