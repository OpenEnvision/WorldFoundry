"""VideoPhy in-tree VideoCon-Physics judge runtime."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.base_models.capabilities import get_base_model_capability
from worldfoundry.base_models.llm_mllm_core.mllm.videocon_physics import runtime_root as videocon_runtime_root
from worldfoundry.base_models.llm_mllm_core.mllm.videocon_physics.constants import (
    PROMPT_PHYSICS,
    PROMPT_VTA,
)
from worldfoundry.evaluation.tasks.execution.framework.official_runner import default_benchmark_timeout
from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path, VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.videophy.videophy_prompts import (
    load_prompt_records,
    official_video_filename_for_record,
    resolve_prompts_json_path,
    resolve_videophy_root,
    unique_generation_records,
)



@dataclass(frozen=True)
class VideoPhyJudgeConfig:
    backend: str
    runtime_root: Path
    checkpoint: Path | None
    prompts_json_path: Path | None
    sa_threshold: float = 0.5
    pc_threshold: float = 0.5
    batch_size: int = 16
    num_frames: int = 32
    python_executable: str = sys.executable
    strict: bool = False




def judge_config_from_env(
    *,
    checkpoint: Path | None = None,
    batch_size: int | None = None,
    num_frames: int | None = None,
    python_executable: str | None = None,
    sa_threshold: float | None = None,
    pc_threshold: float | None = None,
) -> VideoPhyJudgeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_VIDEOPHY_JUDGE_BACKEND")
        or os.environ.get("WORLDFOUNDRY_VIDEOPHY_SCORER_BACKEND")
        or "videocon_physics"
    ).strip().lower()
    if backend in {"official", "real", "autoeval", "videocon", "videocon-physics"}:
        backend = "videocon_physics"
    repo_root = resolve_videophy_root()
    prompts_json_path = None
    try:
        prompts_json_path = resolve_prompts_json_path(repo_root=repo_root)
    except FileNotFoundError:
        prompts_json_path = None
    strict = os.environ.get("WORLDFOUNDRY_VIDEOPHY_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    resolved_python = (
        python_executable
        or os.environ.get("WORLDFOUNDRY_VIDEOPHY_VIDEOCON_PYTHON")
        or os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON")
        or sys.executable
    )
    return VideoPhyJudgeConfig(
        backend=backend,
        runtime_root=videocon_runtime_root(),
        checkpoint=checkpoint or resolve_env_path("WORLDFOUNDRY_VIDEOPHY_VIDEOCON_CKPT"),
        prompts_json_path=prompts_json_path,
        sa_threshold=(
            float(sa_threshold)
            if sa_threshold is not None
            else float(os.environ.get("WORLDFOUNDRY_VIDEOPHY_SA_THRESHOLD", "0.5"))
        ),
        pc_threshold=(
            float(pc_threshold)
            if pc_threshold is not None
            else float(os.environ.get("WORLDFOUNDRY_VIDEOPHY_PC_THRESHOLD", "0.5"))
        ),
        batch_size=batch_size or int(os.environ.get("WORLDFOUNDRY_VIDEOPHY_BATCH_SIZE", "16")),
        num_frames=num_frames or int(os.environ.get("WORLDFOUNDRY_VIDEOPHY_NUM_FRAMES", "32")),
        python_executable=resolved_python,
        strict=strict,
    )


def _video_stems(generated_artifact_dir: Path) -> set[str]:
    if not generated_artifact_dir.is_dir():
        return set()
    return {
        path.stem
        for path in generated_artifact_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }


def _base_model_asset_path(capability_id: str, asset_id: str) -> Path:
    capability = get_base_model_capability(capability_id)
    for asset in capability.assets:
        if asset.id != asset_id:
            continue
        status = asset.check()
        matched_path = status.get("matched_path")
        if status.get("ready") and matched_path:
            return Path(str(matched_path)).expanduser().resolve()
        candidates = "\n".join(f"  - {path}" for path in status.get("candidate_paths", ()))
        exports = "\n".join(f"  {command}" for command in status.get("export_commands", ()))
        raise FileNotFoundError(
            f"Required VideoPhy VideoCon-Physics asset {asset_id!r} is not staged.\n"
            f"Candidate paths:\n{candidates or '  <none>'}\n"
            f"Environment override:\n{exports or '  <none>'}"
        )
    raise KeyError(f"unknown asset {asset_id!r} for capability {capability_id!r}")


def _resolve_checkpoint(config: VideoPhyJudgeConfig) -> Path:
    if config.checkpoint is not None:
        checkpoint = config.checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"VideoPhy VideoCon-Physics checkpoint not found: {checkpoint}")
        return checkpoint
    return _base_model_asset_path("videocon_physics_model", "videocon_physics_checkpoint_dir")


def _video_path_for_record(generated_artifact_dir: Path, record: Mapping[str, Any]) -> Path | None:
    official_name = official_video_filename_for_record(dict(record))
    prompt_id = str(record.get("prompt_id") or "")
    candidates = [generated_artifact_dir / official_name]
    if prompt_id:
        candidates.extend(generated_artifact_dir / f"{prompt_id}{suffix}" for suffix in sorted(VIDEO_SUFFIXES))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _records_with_videos(
    *,
    generated_artifact_dir: Path,
    prompt_records: list[Mapping[str, Any]],
    strict: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in prompt_records:
        prompt_id = str(record.get("prompt_id") or "")
        video_path = _video_path_for_record(generated_artifact_dir, record)
        if video_path is None:
            missing.append(prompt_id)
            continue
        row = dict(record)
        row["videopath"] = str(video_path)
        rows.append(row)
    if strict and missing:
        preview = ", ".join(missing[:20])
        raise FileNotFoundError(f"VideoPhy generated videos are missing for prompt ids: {preview}")
    if not rows:
        raise FileNotFoundError(f"No VideoPhy generated videos found in {generated_artifact_dir}")
    return rows, missing


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_official_inputs(output_dir: Path, prompt_records: list[Mapping[str, Any]]) -> dict[str, Path]:
    input_dir = output_dir / "videocon_physics_inputs"
    source_rows = [
        {
            "prompt_id": str(record.get("prompt_id")),
            "videopath": str(record.get("videopath")),
            "caption": str(record.get("caption") or record.get("prompt") or ""),
        }
        for record in prompt_records
    ]
    sa_rows = [
        {
            "videopath": row["videopath"],
            "caption": PROMPT_VTA.format(caption=row["caption"]),
        }
        for row in source_rows
    ]
    pc_rows = [{"videopath": row["videopath"], "caption": PROMPT_PHYSICS} for row in source_rows]
    source_csv = input_dir / "examples.csv"
    sa_csv = input_dir / "sa_testing.csv"
    pc_csv = input_dir / "physics_testing.csv"
    _write_csv(source_csv, source_rows, ["prompt_id", "videopath", "caption"])
    _write_csv(sa_csv, sa_rows, ["videopath", "caption"])
    _write_csv(pc_csv, pc_rows, ["videopath", "caption"])
    return {"source": source_csv, "sa": sa_csv, "pc": pc_csv}


def _run_videocon_entailment(
    *,
    config: VideoPhyJudgeConfig,
    checkpoint: Path,
    input_csv: Path,
    output_csv: Path,
    log_path: Path,
) -> None:
    command = [
        str(config.python_executable),
        "-m",
        "worldfoundry.base_models.llm_mllm_core.mllm.videocon_physics.entailment_inference",
        "--input_csv",
        str(input_csv),
        "--output_csv",
        str(output_csv),
        "--checkpoint",
        str(checkpoint),
        "--batch_size",
        str(config.batch_size),
        "--num_frames",
        str(config.num_frames),
    ]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[6]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            cwd=str(config.runtime_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=default_benchmark_timeout(),
        )
    if process.returncode != 0:
        raise RuntimeError(f"VideoPhy VideoCon-Physics inference failed with code {process.returncode}; log: {log_path}")


def _read_probability_by_video(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                scores[Path(row[0]).stem] = float(row[2])
            except (TypeError, ValueError):
                continue
    return scores


def _merge_videocon_outputs(
    *,
    prompt_records: list[Mapping[str, Any]],
    sa_csv: Path,
    pc_csv: Path,
    sa_threshold: float,
    pc_threshold: float,
) -> list[dict[str, Any]]:
    sa_scores = _read_probability_by_video(sa_csv)
    pc_scores = _read_probability_by_video(pc_csv)
    rows: list[dict[str, Any]] = []
    for record in prompt_records:
        prompt_id = str(record.get("prompt_id"))
        video_stem = Path(str(record.get("videopath") or prompt_id)).stem
        sa_score = sa_scores.get(video_stem)
        pc_score = pc_scores.get(video_stem)
        sa = None if sa_score is None else int(sa_score >= sa_threshold)
        pc = None if pc_score is None else int(pc_score >= pc_threshold)
        rows.append(
            {
                "prompt_id": prompt_id,
                "caption": record.get("caption") or record.get("prompt"),
                "sa": sa,
                "pc": pc,
                "sa_score": sa_score,
                "pc_score": pc_score,
                "joint": None if sa is None or pc is None else int(sa == 1 and pc == 1),
                "videopath": str(record.get("videopath") or ""),
            }
        )
    return rows


def run_videophy_judge(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: VideoPhyJudgeConfig,
    prompts_json_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if config.backend != "videocon_physics":
        raise ValueError(f"Unsupported VideoPhy judge backend {config.backend!r}; only 'videocon_physics' is in-tree.")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "videophy_results.json"
    prompts_path = prompts_json_path or config.prompts_json_path
    if prompts_path is None:
        prompts_path = resolve_prompts_json_path(repo_root=resolve_videophy_root())
    prompt_records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_records, missing_prompt_ids = _records_with_videos(
        generated_artifact_dir=generated_artifact_dir,
        prompt_records=prompt_records,
        strict=config.strict,
    )
    checkpoint = _resolve_checkpoint(config)
    inputs = _write_official_inputs(output_dir, prompt_records)
    output_csv_dir = output_dir / "videocon_physics_outputs"
    log_dir = output_dir / "logs"
    sa_csv = output_csv_dir / "sa.csv"
    pc_csv = output_csv_dir / "pc.csv"
    _run_videocon_entailment(
        config=config,
        checkpoint=checkpoint,
        input_csv=inputs["sa"],
        output_csv=sa_csv,
        log_path=log_dir / "videocon_physics_sa.log",
    )
    _run_videocon_entailment(
        config=config,
        checkpoint=checkpoint,
        input_csv=inputs["pc"],
        output_csv=pc_csv,
        log_path=log_dir / "videocon_physics_pc.log",
    )
    payload = _merge_videocon_outputs(
        prompt_records=prompt_records,
        sa_csv=sa_csv,
        pc_csv=pc_csv,
        sa_threshold=config.sa_threshold,
        pc_threshold=config.pc_threshold,
    )
    results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "backend": "videocon_physics",
        "results_json": str(results_json.resolve()),
        "video_count": len(_video_stems(generated_artifact_dir)),
        "evaluated_prompt_count": len(prompt_records),
        "missing_prompt_count": len(missing_prompt_ids),
        "missing_prompt_ids": missing_prompt_ids[:50],
        "prompts_json_path": str(prompts_path.resolve()),
        "runtime_root": str(config.runtime_root),
        "checkpoint": str(checkpoint),
        "batch_size": config.batch_size,
        "num_frames": config.num_frames,
        "sa_threshold": config.sa_threshold,
        "pc_threshold": config.pc_threshold,
    }
