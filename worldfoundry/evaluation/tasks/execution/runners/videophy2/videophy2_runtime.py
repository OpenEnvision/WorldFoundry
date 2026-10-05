"""VideoPhy2 in-tree AutoEval judge runtime."""

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
from worldfoundry.base_models.llm_mllm_core.mllm.videophy2_autoeval import runtime_root as autoeval_runtime_root
from worldfoundry.evaluation.tasks.execution.framework.official_runner import default_benchmark_timeout
from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path, VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.videophy2.videophy2_prompts import (
    load_prompt_records,
    official_video_filename_for_record,
    resolve_prompts_json_path,
    resolve_videophy2_root,
    unique_generation_records,
)

JOINT_SA_MIN = 4
JOINT_PC_MIN = 4


@dataclass(frozen=True)
class VideoPhy2JudgeConfig:
    backend: str
    runtime_root: Path
    checkpoint: Path | None
    lora_checkpoint: Path | None
    prompts_json_path: Path | None
    batch_size: int = 1
    num_frames: int = 32
    python_executable: str = sys.executable
    strict: bool = False




def judge_config_from_env(
    *,
    checkpoint: Path | None = None,
    lora_checkpoint: Path | None = None,
    batch_size: int | None = None,
    num_frames: int | None = None,
    python_executable: str | None = None,
) -> VideoPhy2JudgeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_VIDEOPHY2_JUDGE_BACKEND")
        or os.environ.get("WORLDFOUNDRY_VIDEOPHY2_SCORER_BACKEND")
        or "autoeval"
    ).strip().lower()
    if backend in {"official", "real", "video_phy2_auto", "videophy2_auto"}:
        backend = "autoeval"
    repo_root = resolve_videophy2_root()
    prompts_json_path = None
    try:
        prompts_json_path = resolve_prompts_json_path(repo_root=repo_root)
    except FileNotFoundError:
        prompts_json_path = None
    strict = os.environ.get("WORLDFOUNDRY_VIDEOPHY2_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    resolved_batch_size = batch_size or int(os.environ.get("WORLDFOUNDRY_VIDEOPHY2_BATCH_SIZE", "1"))
    resolved_num_frames = num_frames or int(os.environ.get("WORLDFOUNDRY_VIDEOPHY2_NUM_FRAMES", "32"))
    resolved_python = (
        python_executable
        or os.environ.get("WORLDFOUNDRY_VIDEOPHY2_AUTOEVAL_PYTHON")
        or os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON")
        or sys.executable
    )
    return VideoPhy2JudgeConfig(
        backend=backend,
        runtime_root=autoeval_runtime_root(),
        checkpoint=checkpoint or resolve_env_path("WORLDFOUNDRY_VIDEOPHY2_AUTOEVAL_CKPT"),
        lora_checkpoint=lora_checkpoint or resolve_env_path("WORLDFOUNDRY_VIDEOPHY2_AUTOEVAL_LORA_CKPT"),
        prompts_json_path=prompts_json_path,
        batch_size=resolved_batch_size,
        num_frames=resolved_num_frames,
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
            f"Required VideoPhy2 AutoEval asset {asset_id!r} is not staged.\n"
            f"Candidate paths:\n{candidates or '  <none>'}\n"
            f"Environment override:\n{exports or '  <none>'}"
        )
    raise KeyError(f"unknown asset {asset_id!r} for capability {capability_id!r}")


def _resolve_checkpoint(config: VideoPhy2JudgeConfig) -> Path:
    if config.checkpoint is not None:
        checkpoint = config.checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"VideoPhy2 AutoEval checkpoint not found: {checkpoint}")
        return checkpoint
    return _base_model_asset_path("videophy2_autoeval_model", "videophy2_autoeval_checkpoint_dir")


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
        video_path = _video_path_for_record(generated_artifact_dir, record)
        prompt_id = str(record.get("prompt_id") or "")
        if video_path is None:
            missing.append(prompt_id)
            continue
        row = dict(record)
        row["videopath"] = str(video_path)
        rows.append(row)
    if strict and missing:
        preview = ", ".join(missing[:20])
        raise FileNotFoundError(f"VideoPhy2 generated videos are missing for prompt ids: {preview}")
    if not rows:
        raise FileNotFoundError(f"No VideoPhy2 generated videos found in {generated_artifact_dir}")
    return rows, missing


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_autoeval_inputs(output_dir: Path, prompt_records: list[Mapping[str, Any]]) -> dict[str, Path | None]:
    input_dir = output_dir / "videophy2_autoeval_inputs"
    sa_pc_rows = [
        {
            "prompt_id": str(record.get("prompt_id")),
            "videopath": str(record.get("videopath")),
            "caption": str(record.get("caption") or record.get("prompt") or ""),
        }
        for record in prompt_records
    ]
    rule_rows: list[dict[str, Any]] = []
    for record in prompt_records:
        rules = record.get("physics_rules") if isinstance(record.get("physics_rules"), list) else []
        for rule_index, rule in enumerate(rules):
            rule_rows.append(
                {
                    "prompt_id": str(record.get("prompt_id")),
                    "videopath": str(record.get("videopath")),
                    "rule_index": rule_index,
                    "rule": str(rule),
                }
            )
    sa_pc_csv = input_dir / "sa_pc.csv"
    rule_csv = input_dir / "rule.csv"
    _write_csv(sa_pc_csv, sa_pc_rows, ["prompt_id", "videopath", "caption"])
    if rule_rows:
        _write_csv(rule_csv, rule_rows, ["prompt_id", "videopath", "rule_index", "rule"])
        return {"sa_pc": sa_pc_csv, "rule": rule_csv}
    return {"sa_pc": sa_pc_csv, "rule": None}


def _run_autoeval_task(
    *,
    config: VideoPhy2JudgeConfig,
    checkpoint: Path,
    task: str,
    input_csv: Path,
    output_csv: Path,
    log_path: Path,
) -> None:
    command = [
        str(config.python_executable),
        str(config.runtime_root / "inference.py"),
        "--input_csv",
        str(input_csv),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--batch_size",
        str(config.batch_size),
        "--num_frames",
        str(config.num_frames),
        "--output_csv",
        str(output_csv),
    ]
    if config.lora_checkpoint is not None:
        command.extend(["--lora_checkpoint", str(config.lora_checkpoint)])
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(config.runtime_root) if not existing_pythonpath else f"{config.runtime_root}{os.pathsep}{existing_pythonpath}"
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
        raise RuntimeError(f"VideoPhy2 AutoEval task {task!r} failed with code {process.returncode}; log: {log_path}")


def _score_value(row: Mapping[str, Any]) -> int | None:
    value = row.get("score")
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _scores_by_prompt(path: Path) -> dict[str, int]:
    scores: dict[str, int] = {}
    for row in _read_csv_rows(path):
        prompt_id = str(row.get("prompt_id") or "").strip()
        score = _score_value(row)
        if prompt_id and score is not None:
            scores[prompt_id] = score
    return scores


def _rule_lists_by_prompt(path: Path) -> dict[str, tuple[list[str], list[str], list[str]]]:
    grouped: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for row in _read_csv_rows(path):
        prompt_id = str(row.get("prompt_id") or "").strip()
        rule = str(row.get("rule") or "").strip()
        score = _score_value(row)
        if not prompt_id or not rule or score is None:
            continue
        followed, unfollowed, indeterminate = grouped.setdefault(prompt_id, ([], [], []))
        if score == 1:
            followed.append(rule)
        elif score == 0:
            unfollowed.append(rule)
        else:
            indeterminate.append(rule)
    return grouped


def _merge_autoeval_outputs(
    *,
    prompt_records: list[Mapping[str, Any]],
    sa_csv: Path,
    pc_csv: Path,
    rule_csv: Path | None,
) -> list[dict[str, Any]]:
    sa_scores = _scores_by_prompt(sa_csv)
    pc_scores = _scores_by_prompt(pc_csv)
    rule_lists = _rule_lists_by_prompt(rule_csv) if rule_csv is not None and rule_csv.is_file() else {}
    rows: list[dict[str, Any]] = []
    for record in prompt_records:
        prompt_id = str(record.get("prompt_id"))
        sa = sa_scores.get(prompt_id)
        pc = pc_scores.get(prompt_id)
        followed, unfollowed, indeterminate = rule_lists.get(prompt_id, ([], [], []))
        joint = None if sa is None or pc is None else int(sa >= JOINT_SA_MIN and pc >= JOINT_PC_MIN)
        rows.append(
            {
                "prompt_id": prompt_id,
                "caption": record.get("caption") or record.get("prompt"),
                "sa": sa,
                "pc": pc,
                "joint": joint,
                "physics_rules_followed": followed,
                "physics_rules_unfollowed": unfollowed,
                "physics_rules_cannot_be_determined": indeterminate,
                "videopath": str(record.get("videopath") or ""),
            }
        )
    return rows


def run_videophy2_judge(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: VideoPhy2JudgeConfig,
    prompts_json_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if config.backend != "autoeval":
        raise ValueError(f"Unsupported VideoPhy2 judge backend {config.backend!r}; only 'autoeval' is in-tree.")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "videophy2_results.json"
    prompts_path = prompts_json_path or config.prompts_json_path
    if prompts_path is None:
        prompts_path = resolve_prompts_json_path(repo_root=resolve_videophy2_root())
    prompt_records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_records, missing_prompt_ids = _records_with_videos(
        generated_artifact_dir=generated_artifact_dir,
        prompt_records=prompt_records,
        strict=config.strict,
    )
    checkpoint = _resolve_checkpoint(config)
    inputs = _write_autoeval_inputs(output_dir, prompt_records)
    output_csv_dir = output_dir / "videophy2_autoeval_outputs"
    log_dir = output_dir / "logs"
    sa_csv = output_csv_dir / "sa.csv"
    pc_csv = output_csv_dir / "pc.csv"
    rule_csv = output_csv_dir / "rule.csv"
    _run_autoeval_task(
        config=config,
        checkpoint=checkpoint,
        task="sa",
        input_csv=Path(inputs["sa_pc"]),
        output_csv=sa_csv,
        log_path=log_dir / "videophy2_autoeval_sa.log",
    )
    _run_autoeval_task(
        config=config,
        checkpoint=checkpoint,
        task="pc",
        input_csv=Path(inputs["sa_pc"]),
        output_csv=pc_csv,
        log_path=log_dir / "videophy2_autoeval_pc.log",
    )
    rule_input = inputs.get("rule")
    if rule_input is not None:
        _run_autoeval_task(
            config=config,
            checkpoint=checkpoint,
            task="rule",
            input_csv=Path(rule_input),
            output_csv=rule_csv,
            log_path=log_dir / "videophy2_autoeval_rule.log",
        )
    payload = _merge_autoeval_outputs(
        prompt_records=prompt_records,
        sa_csv=sa_csv,
        pc_csv=pc_csv,
        rule_csv=rule_csv if rule_input is not None else None,
    )
    results_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "backend": "autoeval",
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
    }
