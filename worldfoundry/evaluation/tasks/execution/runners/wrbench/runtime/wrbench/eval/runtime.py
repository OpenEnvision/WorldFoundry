"""Eval runtime configuration and orchestration helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wrbench.runtime import (
    RuntimeConfigError,
    _load_runtime_payload,
    _optional_mapping,
    _require_int,
    _resolve_runtime_path,
)


@dataclass(frozen=True)
class EvalScorerRuntime:
    """Paths to external model scorers (host-only, not pip deps)."""

    gpu_id: int
    vggt_python_bin: str | None = None
    vggt_repo: str | None = None
    vggt_checkpoint: str | None = None
    dinov2_python_bin: str | None = None
    dinov2_model_path: str | None = None
    qwen_scorer_python: str | None = None
    qwen35_model: str | None = None
    qwen3vl_model: str | None = None
    extra_paths: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRuntimeConfig:
    schema_version: int
    scorers: EvalScorerRuntime


def _parse_eval_node(payload: dict[str, Any], *, context: str) -> EvalScorerRuntime:
    eval_block = payload.get("eval")
    if isinstance(eval_block, dict) and isinstance(eval_block.get("scorers"), dict):
        node = eval_block["scorers"]
    else:
        raise RuntimeConfigError(f"{context}: required object field 'eval.scorers' is missing or invalid")
    return EvalScorerRuntime(
        vggt_python_bin=node.get("vggt_python_bin"),
        vggt_repo=node.get("vggt_repo"),
        vggt_checkpoint=node.get("vggt_checkpoint"),
        dinov2_python_bin=node.get("dinov2_python_bin"),
        dinov2_model_path=node.get("dinov2_model_path"),
        qwen_scorer_python=node.get("qwen_scorer_python"),
        qwen35_model=node.get("qwen35_model"),
        qwen3vl_model=node.get("qwen3vl_model"),
        gpu_id=_require_int(node, "gpu_id", context=f"{context}: eval.scorers"),
        extra_paths={str(k): str(v) for k, v in _optional_mapping(node, "extra_paths", context=f"{context}: eval.scorers").items()},
        env={str(k): str(v) for k, v in _optional_mapping(node, "env", context=f"{context}: eval.scorers").items()},
    )


def load_eval_runtime(path: Path | None = None) -> EvalRuntimeConfig | None:
    resolved = _resolve_runtime_path(path)
    if resolved is None:
        return None
    payload = _load_runtime_payload(resolved)
    if "eval" not in payload and "scorers" not in payload:
        return None
    return EvalRuntimeConfig(
        schema_version=int(payload["schema_version"]),
        scorers=_parse_eval_node(payload, context=str(resolved)),
    )


def _require_extra(scorers: EvalScorerRuntime, field: str) -> str:
    value = scorers.extra_paths.get(field)
    if value is None or not str(value).strip():
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} is required")
    return str(value).strip()


def _require_extra_int(scorers: EvalScorerRuntime, field: str) -> int:
    value = _require_extra(scorers, field)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} must be an integer") from exc
    return parsed


def _require_extra_float(scorers: EvalScorerRuntime, field: str) -> float:
    value = _require_extra(scorers, field)
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} must be a number") from exc


def require_eval_runtime(path: Path | None = None) -> EvalRuntimeConfig:
    cfg = load_eval_runtime(path)
    if cfg is None:
        raise RuntimeError(
            "eval scorers not configured: add an 'eval.scorers' section to wrbench.runtime.json "
            "(see wrbench.runtime.example.json)"
        )
    return cfg


def contract_path() -> Path:
    return Path(__file__).resolve().parent / "contract" / "latest_d1_d6_metric_contract.json"


def run_module_main(module: str, argv: list[str], *, env: dict[str, str] | None = None) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    cmd = [sys.executable, "-m", module, *argv]
    proc = subprocess.run(cmd, env=merged)
    return int(proc.returncode)


def run_scorer_python(
    python_bin: str,
    module: str,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    cmd = [python_bin, "-m", module, *argv]
    proc = subprocess.run(cmd, env=merged)
    return int(proc.returncode)


def run_shell_script(script: Path, stage: str, *, env: dict[str, str]) -> int:
    merged = os.environ.copy()
    merged.update(env)
    proc = subprocess.run(["bash", str(script), stage], env=merged)
    return int(proc.returncode)


def d1_score(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    summary_csv: Path,
    pose_cache_root: Path,
    pose_backend: str,
    poses_file: str,
    default_frames: int,
    sidecar_profile_gate: str,
    predicted_pose_type: str,
    predicted_camera_convention: str,
    target_camera_convention: str,
    rot_scale_deg: float,
    trans_scale: float,
    yaw_weak_threshold_deg: float,
    pan_weak_threshold: float,
    static_rot_threshold_deg: float,
    static_trans_threshold: float,
) -> int:
    from wrbench.eval.d1.d1_camera import main as d1_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-jsonl",
        str(output_jsonl),
        "--summary-csv",
        str(summary_csv),
        "--megasam-cache-root",
        str(pose_cache_root),
        "--pose-backend",
        pose_backend,
        "--poses-file",
        poses_file,
        "--default-frames",
        str(default_frames),
        "--sidecar-profile-gate",
        sidecar_profile_gate,
        "--predicted-pose-type",
        predicted_pose_type,
        "--predicted-camera-convention",
        predicted_camera_convention,
        "--target-camera-convention",
        target_camera_convention,
        "--rot-scale-deg",
        str(rot_scale_deg),
        "--trans-scale",
        str(trans_scale),
        "--yaw-weak-threshold-deg",
        str(yaw_weak_threshold_deg),
        "--pan-weak-threshold",
        str(pan_weak_threshold),
        "--static-rot-threshold-deg",
        str(static_rot_threshold_deg),
        "--static-trans-threshold",
        str(static_trans_threshold),
    ]
    return d1_main(argv)


def d1_camalign_score(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    pose_cache_root: Path,
    poses_file: str,
) -> int:
    from wrbench.eval.d1.d1_camalign import main as camalign_main

    return camalign_main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--pose-cache-root",
            str(pose_cache_root),
            "--poses-file",
            poses_file,
        ]
    )


def d1_vggt_batch(
    *,
    eval_runtime: EvalRuntimeConfig,
    input_jsonl: Path,
    output_root: Path,
    cache_root: Path,
    execution_mode: str,
) -> int:
    scorers = eval_runtime.scorers
    for field_name, value in (
        ("vggt_python_bin", scorers.vggt_python_bin),
        ("vggt_repo", scorers.vggt_repo),
        ("vggt_checkpoint", scorers.vggt_checkpoint),
    ):
        if not value or not Path(str(value)).exists():
            raise FileNotFoundError(f"eval.scorers.{field_name} missing or not found: {value!r}")
    from wrbench.eval.d1.run_d1_vggt_omega_batch import main as batch_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-root",
        str(output_root),
        "--vggt-repo",
        str(scorers.vggt_repo),
        "--checkpoint",
        str(scorers.vggt_checkpoint),
        "--gpu-id",
        str(scorers.gpu_id),
        "--cache-root",
        str(cache_root),
        "--python",
        str(scorers.vggt_python_bin),
        "--execution-mode",
        execution_mode,
        "--poses-file",
        _require_extra(scorers, "d1_poses_file"),
        "--image-resolution",
        str(_require_extra_int(scorers, "d1_vggt_image_resolution")),
        "--preprocess-mode",
        _require_extra(scorers, "d1_vggt_preprocess_mode"),
        "--cwd",
        _require_extra(scorers, "d1_vggt_cwd"),
        "--shard-index",
        _require_extra(scorers, "d1_vggt_shard_index"),
        "--num-shards",
        _require_extra(scorers, "d1_vggt_num_shards"),
    ]
    return batch_main(argv)


def d2_extract(
    *,
    eval_runtime: EvalRuntimeConfig,
    videos_manifest: Path,
    out_jsonl: Path,
    model_dir: Path | None = None,
) -> int:
    scorers = eval_runtime.scorers
    if not scorers.dinov2_python_bin or not Path(str(scorers.dinov2_python_bin)).exists():
        raise FileNotFoundError(f"eval.scorers.dinov2_python_bin missing or not found: {scorers.dinov2_python_bin!r}")
    if model_dir is None and (not scorers.dinov2_model_path or not Path(str(scorers.dinov2_model_path)).exists()):
        raise FileNotFoundError(f"eval.scorers.dinov2_model_path missing or not found: {scorers.dinov2_model_path!r}")
    python_bin = scorers.dinov2_python_bin
    model = model_dir if model_dir is not None else Path(str(scorers.dinov2_model_path))
    argv = [
        "--videos",
        str(videos_manifest),
        "--out-jsonl",
        str(out_jsonl),
        "--device",
        _require_extra(scorers, "d2_device"),
        "--sample-policy",
        _require_extra(scorers, "d2_sample_policy"),
        "--sample-fps",
        str(_require_extra_float(scorers, "d2_sample_fps")),
        "--min-frames",
        str(_require_extra_int(scorers, "d2_min_frames")),
        "--max-frames",
        str(_require_extra_int(scorers, "d2_max_frames")),
        "--batch-size",
        str(_require_extra_int(scorers, "d2_batch_size")),
    ]
    argv.extend(["--model-dir", str(model)])
    return run_scorer_python(
        python_bin,
        "wrbench.eval.d2.extract_d2_dinov2_local_global_candidate",
        argv,
        env={"CUDA_VISIBLE_DEVICES": str(scorers.gpu_id), **scorers.env},
    )


def wrbench_repo_root() -> Path:
    """Return the WRBench repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[3]


def d3d6_env(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scorer_profile: str,
) -> dict[str, str]:
    profile = str(scorer_profile).strip()
    if not profile:
        raise ValueError("scorer_profile is required")
    scorers = eval_runtime.scorers
    for field_name, value in (
        ("qwen_scorer_python", scorers.qwen_scorer_python),
        ("qwen35_model", scorers.qwen35_model),
        ("qwen3vl_model", scorers.qwen3vl_model),
    ):
        if not value or not Path(str(value)).exists():
            raise FileNotFoundError(f"eval.scorers.{field_name} missing or not found: {value!r}")
    required_env = (
        "FORCE_QWENVL_VIDEO_READER",
        "WORLD_STATE_VIDEO_BACKEND",
        "RUN_TAG",
        "PROMPT_MODE",
        "TASK_CONTEXT_MODE",
        "NUM_SHARDS",
        "SHARD_IDS",
        "CUDA_DEVICES_CSV",
        "FPS",
        "PROGRESS_EVERY",
        "SKIP_EXISTING",
        "QWEN35_VLM_NAME",
        "QWEN35_LOADER_FAMILY",
        "QWEN35_DTYPE",
        "QWEN35_ATTN_IMPLEMENTATION",
        "QWEN35_NUM_SAMPLES",
        "QWEN35_MAX_VIDEOS",
        "QWEN35_EVIDENCE_CONTEXT_MODE",
        "QWEN3VL_DTYPE",
        "QWEN3VL_ATTN_IMPLEMENTATION",
        "QWEN3VL_MAX_NEW_TOKENS",
        "QWEN3VL_MAX_VIDEOS",
        "QWEN3VL_BINARY_PROMPT_SCHEMA",
        "QWEN3VL_RESCUE_PROMPT_SCHEMA",
    )
    missing_env = [name for name in required_env if not str(scorers.env.get(name, "")).strip()]
    if missing_env:
        raise RuntimeConfigError(
            "eval.scorers.env missing required D3-D6 field(s): " + ", ".join(missing_env)
        )
    repo_root = wrbench_repo_root()
    shell = repo_root / "scripts" / "eval" / "score_runtime_v2_d3d6.sh"
    env = {
        "MANIFEST": str(manifest.resolve()),
        "OUT_DIR": str(out_dir.resolve()),
        "SCORER_PROFILE": profile,
        "PY_SCORER": str(scorers.qwen_scorer_python),
        "PY_HELPER": str(scorers.qwen_scorer_python),
        "QWEN35_MODEL": str(scorers.qwen35_model),
        "QWEN3VL_MODEL": str(scorers.qwen3vl_model),
        "CUDA_VISIBLE_DEVICES": str(scorers.gpu_id),
        "PYTHONPATH": str(repo_root),
        **scorers.env,
    }
    if not shell.is_file():
        raise FileNotFoundError(f"D3-D6 shell orchestrator missing: {shell}")
    env["WRBENCH_D3D6_SCRIPT"] = str(shell)
    return env


def d3d6_score(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    stage: str,
    scorer_profile: str,
) -> int:
    env = d3d6_env(
        eval_runtime=eval_runtime,
        manifest=manifest,
        out_dir=out_dir,
        scorer_profile=scorer_profile,
    )
    script = Path(env["WRBENCH_D3D6_SCRIPT"])
    return run_shell_script(script, stage, env=env)


def eval_run(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scorer_profile: str,
    sidecar_profile_gate: str,
) -> int:
    """Run the full WRBench eval pipeline: D1 pose -> D1 score -> D2 -> D3-D6 -> table."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest.resolve()
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list")

    d1_input = out_dir / "d1_input.jsonl"
    d1_input.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if isinstance(row, dict)),
        encoding="utf-8",
    )
    pose_root = out_dir / "d1_vggt"
    cache_root = pose_root / "cache"
    d1_scored = out_dir / "d1_scored.jsonl"
    d1_camalign_scored = out_dir / "d1_camalign_scored.jsonl"
    d1_summary = out_dir / "d1_summary.csv"
    d2_out = out_dir / "d2_features.jsonl"
    d3d6_out = out_dir / "d3d6"
    runtime_scores = d3d6_out / "final_exports" / "scores_v7_runtime_v2_evidence_first_gate_masked_export.json"

    steps: list[tuple[str, int]] = []
    rc = d1_vggt_batch(
        eval_runtime=eval_runtime,
        input_jsonl=d1_input,
        output_root=pose_root,
        cache_root=cache_root,
        execution_mode=_require_extra(eval_runtime.scorers, "d1_vggt_execution_mode"),
    )
    steps.append(("d1-vggt", rc))
    if rc != 0:
        return rc

    rc = d1_score(
        input_jsonl=d1_input,
        output_jsonl=d1_scored,
        summary_csv=d1_summary,
        pose_cache_root=cache_root,
        pose_backend=_require_extra(eval_runtime.scorers, "d1_pose_backend"),
        poses_file=_require_extra(eval_runtime.scorers, "d1_poses_file"),
        default_frames=_require_extra_int(eval_runtime.scorers, "d1_default_frames"),
        sidecar_profile_gate=sidecar_profile_gate,
        predicted_pose_type=_require_extra(eval_runtime.scorers, "d1_predicted_pose_type"),
        predicted_camera_convention=_require_extra(eval_runtime.scorers, "d1_predicted_camera_convention"),
        target_camera_convention=_require_extra(eval_runtime.scorers, "d1_target_camera_convention"),
        rot_scale_deg=_require_extra_float(eval_runtime.scorers, "d1_rot_scale_deg"),
        trans_scale=_require_extra_float(eval_runtime.scorers, "d1_trans_scale"),
        yaw_weak_threshold_deg=_require_extra_float(eval_runtime.scorers, "d1_yaw_weak_threshold_deg"),
        pan_weak_threshold=_require_extra_float(eval_runtime.scorers, "d1_pan_weak_threshold"),
        static_rot_threshold_deg=_require_extra_float(eval_runtime.scorers, "d1_static_rot_threshold_deg"),
        static_trans_threshold=_require_extra_float(eval_runtime.scorers, "d1_static_trans_threshold"),
    )
    steps.append(("d1", rc))
    if rc != 0:
        return rc

    rc = d1_camalign_score(
        input_jsonl=d1_input,
        output_jsonl=d1_camalign_scored,
        pose_cache_root=cache_root,
        poses_file=_require_extra(eval_runtime.scorers, "d1_poses_file"),
    )
    steps.append(("d1-camalign", rc))
    if rc != 0:
        return rc

    rc = d2_extract(
        eval_runtime=eval_runtime,
        videos_manifest=manifest,
        out_jsonl=d2_out,
    )
    steps.append(("d2", rc))
    if rc != 0:
        return rc

    rc = d3d6_score(
        eval_runtime=eval_runtime,
        manifest=manifest,
        out_dir=d3d6_out,
        stage="all",
        scorer_profile=scorer_profile,
    )
    steps.append(("d3d6", rc))
    if rc != 0:
        return rc

    if not runtime_scores.is_file():
        raise FileNotFoundError(f"D3-D6 export missing: {runtime_scores}")

    rc = build_table(
        [
            "--runtime-scores",
            str(runtime_scores),
            "--d1-scores",
            str(d1_scored),
            "--d1-camalign-scores",
            str(d1_camalign_scored),
            "--d2-scores",
            str(d2_out),
            "--out-csv",
            str(out_dir / "main_table.csv"),
            "--out-md",
            str(out_dir / "main_table.md"),
            "--out-summary",
            str(out_dir / "main_table_summary.json"),
        ]
    )
    steps.append(("table", rc))
    return rc


def build_table(argv: list[str] | None = None) -> int:
    from wrbench.eval.aggregate.build_wrbench_vnext_main_table import main as table_main

    return table_main(argv)
