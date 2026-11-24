#!/usr/bin/env python3
"""WorldFoundry in-tree launcher for the DEVIL official evaluator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
from worldfoundry.base_models.perception_core.frame_interpolation.amt import (
    checkpoint_path as amt_checkpoint_path,
)
from worldfoundry.base_models.perception_core.frame_interpolation.amt import (
    config_path as amt_config_path,
)
from worldfoundry.base_models.perception_core.optical_flow.raft import checkpoint_path as raft_checkpoint_path

RUNTIME_ROOT = Path(__file__).resolve().parent


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _maybe_path(value: str | os.PathLike | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _model_weights_candidate(model_weights_dir: Path | None, filename: str) -> Path | None:
    if model_weights_dir is None:
        return None
    candidate = model_weights_dir / filename
    return candidate if candidate.exists() else None


def _capability_asset_path(capability_id: str, asset_id: str) -> Path | None:
    capability = BASE_MODEL_CAPABILITIES.get(capability_id)
    if capability is None:
        return None
    for asset in capability.assets:
        if asset.id != asset_id:
            continue
        status = asset.check()
        candidate = status.get("matched_path") or status.get("local_path")
        return Path(candidate).expanduser().resolve() if candidate else None
    return None


def _callable_path(func: Callable[[], Path]) -> Path | None:
    try:
        return func().expanduser().resolve()
    except Exception:
        return None


def _resolve_required_file(
    *,
    label: str,
    explicit: str | os.PathLike | None = None,
    env_names: tuple[str, ...] = (),
    candidates: tuple[Path | None, ...] = (),
) -> Path:
    ordered: list[Path | None] = [_maybe_path(explicit), _maybe_path(_first_env(*env_names))]
    ordered.extend(candidates)
    for candidate in ordered:
        if candidate is not None and candidate.is_file():
            return candidate
    hints = list(env_names)
    if explicit:
        hints.insert(0, str(explicit))
    raise FileNotFoundError(f"missing DEVIL {label}; set one of: {', '.join(hints) or label}")


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _prompt_dynamic_grades(video_names: pd.Series) -> np.ndarray:
    keywords = ["static", "low", "medium", "very_high", "high"]
    values = [0, 1, 2, 4, 3]
    grades = np.zeros(len(video_names), dtype=int)
    for index, name in enumerate(video_names):
        normalized = str(name).lower()
        for keyword, value in zip(keywords, values, strict=False):
            if keyword in normalized:
                grades[index] = value
                break
    return grades


def _controllability_winning_rate(prompt_dynamics: np.ndarray, video_dynamics: np.ndarray) -> float:
    x_degree_other: dict[int, np.ndarray] = {}
    y_degree_other: dict[int, np.ndarray] = {}
    for degree in np.unique(prompt_dynamics):
        mask = prompt_dynamics == degree
        x_degree_other[int(degree)] = video_dynamics[~mask]
        y_degree_other[int(degree)] = prompt_dynamics[~mask]

    winning_rates = []
    for video_score, prompt_score in zip(video_dynamics, prompt_dynamics, strict=False):
        x_other = x_degree_other[int(prompt_score)]
        y_other = y_degree_other[int(prompt_score)]
        if len(x_other) == 0:
            continue
        winning_rates.append(((video_score - x_other) * (prompt_score - y_other) > 0).mean())
    return float(sum(winning_rates) / len(winning_rates)) if winning_rates else 0.0


def _summarize_results(save_dir: Path, dynamic_score_name: str) -> Path:
    dynamic_path = save_dir / dynamic_score_name
    quality_path = save_dir / "dynamics_quality_results.xlsx"
    dynamics = pd.read_excel(dynamic_path)
    if "Overall_dynamics_scores" not in dynamics.columns:
        raise ValueError(f"{dynamic_path} does not contain Overall_dynamics_scores")
    overall_dynamic = dynamics["Overall_dynamics_scores"].to_numpy(dtype=float)
    dynamics_range = float(np.percentile(overall_dynamic, 99) - np.percentile(overall_dynamic, 1))
    dynamics_controllability = _controllability_winning_rate(
        _prompt_dynamic_grades(dynamics["video_name"]),
        overall_dynamic,
    )

    quality = pd.read_excel(quality_path)
    average_rows = quality[quality["Key"].astype(str).str.lower() == "average"]
    if average_rows.empty:
        dynamics_quality = float(quality["Overall Mean"].astype(float).mean())
    else:
        dynamics_quality = float(average_rows.iloc[0]["Overall Mean"])
    summary = {
        "dynamics_range": dynamics_range,
        "dynamics_controllability": dynamics_controllability,
        "dynamics_quality": dynamics_quality,
        "devil_dynamics_average": float(np.mean([dynamics_range, dynamics_controllability, dynamics_quality])),
    }
    output_path = save_dir / "devil_dynamics_results.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the in-tree DEVIL official evaluator.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("WORLDFOUNDRY_DEVIL_NUM_GPUS", "1")))
    parser.add_argument("--gemini-api-key", default=_first_env("WORLDFOUNDRY_DEVIL_GEMINI_API_KEY", "GEMINI_API_KEY"))
    parser.add_argument("--naturalness-path", type=Path)
    parser.add_argument(
        "--model-weights-dir", type=Path, default=_maybe_path(_first_env("WORLDFOUNDRY_DEVIL_MODEL_WEIGHTS_DIR"))
    )
    parser.add_argument("--regression-ckpt")
    parser.add_argument("--raft-ckpt")
    parser.add_argument("--clip-vit-l14")
    parser.add_argument("--clip-vit-b32")
    parser.add_argument("--viclip-ckpt")
    parser.add_argument("--dinov2-vitl14-ckpt")
    parser.add_argument("--timm-dino-ckpt")
    parser.add_argument("--amt-config")
    parser.add_argument("--amt-ckpt")
    parser.add_argument("--dino-source-dir")
    parser.add_argument("--dino-vitb16-ckpt")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    save_dir = args.output_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_weights_dir = args.model_weights_dir.expanduser().resolve() if args.model_weights_dir else None

    regression_ckpt = _resolve_required_file(
        label="linear regression checkpoint",
        explicit=args.regression_ckpt,
        env_names=("WORLDFOUNDRY_DEVIL_REGRESSION_CKPT", "WORLDFOUNDRY_DEVIL_LINEAR_REGRESSION_CKPT"),
        candidates=(_model_weights_candidate(model_weights_dir, "linear_regress_model.pth"),),
    )
    raft_ckpt = _resolve_required_file(
        label="RAFT checkpoint",
        explicit=args.raft_ckpt,
        env_names=("WORLDFOUNDRY_DEVIL_RAFT_CKPT", "WORLDFOUNDRY_RAFT_THINGS_CKPT"),
        candidates=(
            _model_weights_candidate(model_weights_dir, "raft-things.pth"),
            _callable_path(raft_checkpoint_path),
        ),
    )
    clip_vit_l14 = _resolve_required_file(
        label="CLIP ViT-L/14 checkpoint",
        explicit=args.clip_vit_l14,
        env_names=("WORLDFOUNDRY_DEVIL_CLIP_VIT_L14_PT", "WORLDFOUNDRY_VBENCH_CLIP_VIT_L14_PT"),
        candidates=(
            _model_weights_candidate(model_weights_dir, "ViT-L-14.pt"),
            _capability_asset_path("vbench_metric_checkpoint_assets", "vbench_clip_vit_l14_checkpoint"),
        ),
    )
    clip_vit_b32 = _resolve_required_file(
        label="CLIP ViT-B/32 checkpoint",
        explicit=args.clip_vit_b32,
        env_names=("WORLDFOUNDRY_DEVIL_CLIP_VIT_B32_PT", "WORLDFOUNDRY_VBENCH_CLIP_VIT_B32_PT"),
        candidates=(
            _model_weights_candidate(model_weights_dir, "ViT-B-32.pt"),
            _capability_asset_path("vbench_metric_checkpoint_assets", "vbench_clip_vit_b32_checkpoint"),
        ),
    )
    viclip_ckpt = _resolve_required_file(
        label="ViCLIP checkpoint",
        explicit=args.viclip_ckpt,
        env_names=("WORLDFOUNDRY_DEVIL_VICLIP_CKPT", "WORLDFOUNDRY_VBENCH_VICLIP_CKPT"),
        candidates=(
            _model_weights_candidate(model_weights_dir, "ViClip-InternVid-10M-FLT.pth"),
            _capability_asset_path("vbench_metric_checkpoint_assets", "vbench_viclip_checkpoint"),
        ),
    )
    dinov2_ckpt = _resolve_required_file(
        label="DINOv2 ViT-L/14 checkpoint",
        explicit=args.dinov2_vitl14_ckpt,
        env_names=("WORLDFOUNDRY_DEVIL_DINOV2_VITL14_CKPT", "WORLDFOUNDRY_WBENCH_MEGASAM_DINOV2_CKPT"),
        candidates=(
            _model_weights_candidate(model_weights_dir, "dinov2_vitl14_pretrain.pth"),
            _capability_asset_path("wbench_megasam", "wbench_megasam_dinov2_checkpoint"),
        ),
    )
    amt_config = _resolve_required_file(
        label="AMT config",
        explicit=args.amt_config,
        env_names=("WORLDFOUNDRY_DEVIL_AMT_CONFIG", "WORLDFOUNDRY_VBENCH_AMT_S_CONFIG"),
        candidates=(_model_weights_candidate(model_weights_dir, "AMT-S.yaml"), _callable_path(amt_config_path)),
    )
    amt_ckpt = _resolve_required_file(
        label="AMT checkpoint",
        explicit=args.amt_ckpt,
        env_names=("WORLDFOUNDRY_DEVIL_AMT_CKPT", "WORLDFOUNDRY_VBENCH_AMT_S_CKPT"),
        candidates=(_model_weights_candidate(model_weights_dir, "amt-s.pth"), _callable_path(amt_checkpoint_path)),
    )
    dino_source = _maybe_path(args.dino_source_dir) or _capability_asset_path(
        "vbench_metric_checkpoint_assets", "vbench_dino_source"
    )
    dino_vitb16_ckpt = _resolve_required_file(
        label="DINO ViT-B/16 checkpoint",
        explicit=args.dino_vitb16_ckpt,
        env_names=("WORLDFOUNDRY_DEVIL_DINO_VITB16_CKPT", "WORLDFOUNDRY_VBENCH_DINO_VITB16_CKPT"),
        candidates=(
            _model_weights_candidate(model_weights_dir, "dino_vitbase16_pretrain.pth"),
            _capability_asset_path("vbench_metric_checkpoint_assets", "vbench_dino_vitb16_checkpoint"),
        ),
    )
    if dino_source is None or not (dino_source / "hubconf.py").exists():
        raise FileNotFoundError("missing in-tree DINO source shim for DEVIL quality scoring")
    if args.naturalness_path is None and not args.gemini_api_key:
        raise ValueError("DEVIL naturalness requires --gemini-api-key/GEMINI_API_KEY or --naturalness-path")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(RUNTIME_ROOT), env.get("PYTHONPATH", "")])
    dynamic_score_name = "dynamics_results.xlsx"
    quality_score_name = "quality_results.xlsx"

    dynamic_cmd = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.num_gpus),
        "tools.evaluate_dynamic_ddp_metrics",
        "--video_dir",
        str(args.video_dir),
        "--save_dir",
        str(save_dir),
        "--dynamic_score_save_name",
        dynamic_score_name,
        "--regress_model_weight_path",
        str(regression_ckpt),
        "--raft_model_path",
        str(raft_ckpt),
        "--clip_model_path",
        str(clip_vit_l14),
        "--viclip_model_path",
        str(viclip_ckpt),
        "--dinov2_model_path",
        str(dinov2_ckpt),
    ]
    if args.timm_dino_ckpt:
        dynamic_cmd.extend(["--timm_dino_model_path", str(Path(args.timm_dino_ckpt).expanduser().resolve())])
    _run(dynamic_cmd, cwd=RUNTIME_ROOT, env=env)

    quality_cmd = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.num_gpus),
        "tools.evaluate_quality_dist",
        "--video_dir",
        str(args.video_dir),
        "--save_dir",
        str(save_dir),
        "--quality_save_name",
        quality_score_name,
        "--clip_path",
        str(clip_vit_b32),
        "--amt_config_path",
        str(amt_config),
        "--amt_ckpt_path",
        str(amt_ckpt),
        "--dino_source_dir",
        str(dino_source),
        "--dino_checkpoint_path",
        str(dino_vitb16_ckpt),
    ]
    if args.naturalness_path:
        quality_cmd.extend(["--naturalness_path", str(args.naturalness_path.expanduser().resolve())])
    else:
        quality_cmd.extend(["--gemini_api_key", str(args.gemini_api_key)])
    _run(quality_cmd, cwd=RUNTIME_ROOT, env=env)

    final_cmd = [
        args.python,
        "-m",
        "tools.calculate_metrics",
        "--video_dir",
        str(args.video_dir),
        "--save_dir",
        str(save_dir),
        "--regress_model_weight_path",
        str(regression_ckpt),
        "--dynamic_score_save_name",
        dynamic_score_name,
        "--quality_save_name",
        quality_score_name,
        "--print_detail_quality_results",
    ]
    _run(final_cmd, cwd=RUNTIME_ROOT, env=env)
    summary_path = _summarize_results(save_dir, dynamic_score_name)
    print(f"DEVIL summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
