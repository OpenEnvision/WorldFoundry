#!/usr/bin/env python3
"""Official runner for DEVIL Dynamics."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors
from worldfoundry.evaluation.tasks.execution.framework.io import normalize_unit_score, scalar_number



CONFIG = ors.BenchRunnerConfig(
    benchmark_id='devil-dynamics',
    display_name='DEVIL Dynamics',
    root_env='WORLDFOUNDRY_DEVIL_DYNAMICS_RUNTIME_ROOT',
    results_path_env='WORLDFOUNDRY_DEVIL_DYNAMICS_RESULTS_PATH',
    default_repo_subdir='worldfoundry/evaluation/tasks/execution/runners/devil_dynamics/runtime/official',
    metric_order=('dynamics_range', 'dynamics_controllability', 'dynamics_quality', 'devil_dynamics_average'),
    metric_specs={'dynamics_range': {'name': 'Dynamics Range', 'group': 'dynamics', 'higher_is_better': True}, 'dynamics_controllability': {'name': 'Dynamics Controllability', 'group': 'dynamics', 'higher_is_better': True}, 'dynamics_quality': {'name': 'Dynamics Quality', 'group': 'dynamics', 'higher_is_better': True}, 'devil_dynamics_average': {'name': 'DEVIL Dynamics Average', 'group': 'aggregate', 'higher_is_better': True}},
    # Only official DEVIL result keys belong here (devil_dynamics_results.json emits the
    # canonical ids; the quality xlsx generic row keys are mapped by DEVIL_KEY_TO_METRIC).
    # Cross-benchmark generic names (subject_consistency, background_consistency,
    # motion_smoothness, naturalness) were removed: aliasing them to dynamics metrics
    # mis-scores payloads that carry same-named metrics from other benchmarks (ET-13).
    metric_aliases={'dynamics_range': 'dynamics_range', 'dynamics_controllability': 'dynamics_controllability', 'dynamics_quality': 'dynamics_quality', 'dynamics_based_quality': 'dynamics_quality', 'overall': 'devil_dynamics_average', 'devil_dynamics_average': 'devil_dynamics_average'},
    average_metric_id='devil_dynamics_average',
    official_entry='worldfoundry.evaluation.tasks.execution.runners.devil_dynamics.runtime.official.devil_official_runtime',
    official_output_globs=(
        'upstream/devil_dynamics_results.json',
        'upstream/dynamics_quality_results.xlsx',
        'devil_dynamics_results.json',
        'Devil-eval-results-*/dynamics_quality_results.xlsx',
        'dynamics_quality_results.xlsx',
    ),
    requires_api_env=(),
    usage_epilog='Examples:\n  python3 run_devil_dynamics_official_runner.py \\\n    --run-official --generated-video-dir /path/to/devil/videos \\\n    --output-dir /tmp/devil_out --json\n\n  python3 run_devil_dynamics_official_runner.py \\\n    --official-results-path Devil-eval-results-20250101/dynamics_quality_results.xlsx \\\n    --output-dir /tmp/devil_out --json',
)

DEVIL_KEY_TO_METRIC = {
    "motion_smoothness": "dynamics_controllability",
    "naturalness": "dynamics_quality",
    "subject_consistency": "dynamics_range",
    "background_consistency": "dynamics_range",
}

def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    search_roots = [output_dir]
    if repo_root is not None:
        search_roots.extend([repo_root, repo_root / "Devil-eval-results"])
    return ors.discover_by_globs(search_roots, CONFIG.official_output_globs)

def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    extracted = ors.generic_extract_metrics(payload, CONFIG, str(results_path))
    if extracted or results_path.suffix.lower() not in {".xlsx", ".xls"}:
        return extracted
    rows = payload if isinstance(payload, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("Key") or row.get("key") or "").strip()
        metric_id = DEVIL_KEY_TO_METRIC.get(key) or ors.metric_id_from_key(key, CONFIG)
        if metric_id is None:
            continue
        raw_score = scalar_number(row.get("Overall Mean") if row.get("Overall Mean") is not None else row.get("overall_mean"))
        if raw_score is None or metric_id in extracted:
            continue
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_score,
            "normalized_score": normalize_unit_score(raw_score),
            "source": "devil_dynamics_quality_xlsx",
            "sample_count": None,
        }
    return extracted

def build_official_command(*, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any) -> list[str] | None:
    command = [
        args.python,
        "-m",
        str(config.official_entry),
        "--video-dir",
        str(generated_video_dir),
        "--output-dir",
        str(output_dir / "upstream"),
        "--num-gpus",
        str(args.num_gpus),
        "--python",
        args.python,
    ]
    optional_args = {
        "--naturalness-path": args.naturalness_path,
        "--model-weights-dir": args.model_weights_dir,
        "--regression-ckpt": args.regression_ckpt,
        "--raft-ckpt": args.raft_ckpt,
        "--clip-vit-l14": args.clip_vit_l14,
        "--clip-vit-b32": args.clip_vit_b32,
        "--viclip-ckpt": args.viclip_ckpt,
        "--dinov2-vitl14-ckpt": args.dinov2_vitl14_ckpt,
        "--timm-dino-ckpt": args.timm_dino_ckpt,
        "--amt-config": args.amt_config,
        "--amt-ckpt": args.amt_ckpt,
        "--dino-source-dir": args.dino_source_dir,
        "--dino-vitb16-ckpt": args.dino_vitb16_ckpt,
    }
    for flag, value in optional_args.items():
        if value is not None:
            command.extend([flag, str(value)])
    return command


def build_official_environment(
    *, config, repo_root: Path, generated_video_dir: Path, output_dir: Path, args: Any
) -> dict[str, str]:
    """Pass the Gemini credential without exposing it in argv or scorecards."""
    del config, repo_root, generated_video_dir, output_dir
    if not args.gemini_api_key:
        return {}
    return {"WORLDFOUNDRY_DEVIL_GEMINI_API_KEY": str(args.gemini_api_key)}

def extend_parser(parser) -> None:
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("WORLDFOUNDRY_DEVIL_NUM_GPUS", "1")),
        help="Number of GPUs for DEVIL distributed metric computation.",
    )
    parser.add_argument("--gemini-api-key", default=os.environ.get("WORLDFOUNDRY_DEVIL_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--naturalness-path", type=Path, help="Precomputed DEVIL naturalness xlsx; skips Gemini calls.")
    parser.add_argument("--model-weights-dir", type=Path, default=ors.env_path("WORLDFOUNDRY_DEVIL_MODEL_WEIGHTS_DIR"))
    parser.add_argument("--regression-ckpt", type=Path)
    parser.add_argument("--raft-ckpt", type=Path)
    parser.add_argument("--clip-vit-l14", type=Path)
    parser.add_argument("--clip-vit-b32", type=Path)
    parser.add_argument("--viclip-ckpt", type=Path)
    parser.add_argument("--dinov2-vitl14-ckpt", type=Path)
    parser.add_argument("--timm-dino-ckpt", type=Path)
    parser.add_argument("--amt-config", type=Path)
    parser.add_argument("--amt-ckpt", type=Path)
    parser.add_argument("--dino-source-dir", type=Path)
    parser.add_argument(
        "--dino-vitb16-ckpt",
        type=Path,
        help="DINO ViT-B/16 checkpoint for DEVIL quality scoring.",
    )

def main(argv: list[str] | None = None) -> int:
    return ors.run_main(
        CONFIG,
        ors.RunnerHooks(
            build_official_command=build_official_command,
            build_official_environment=build_official_environment,
            discover_official_results=discover_official_results,
            extract_metrics=extract_metrics,
            extend_parser=extend_parser,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
