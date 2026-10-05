"""MiraBench official scorer runtime (mock average_score or optional calculate_score.py dispatch)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.execution.process import read_text_tail, run_logged_subprocess
from worldfoundry.core.io import materialize_file
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_metrics import METRIC_ORDER, METRIC_SPECS
from worldfoundry.evaluation.tasks.execution.runners.mirabench.mirabench_prompts import (
    load_prompt_records,
    resolve_meta_csv_path,
    resolve_mirabench_root,
    unique_prompt_records,
    write_meta_csv,
)


def resolve_gt_meta_csv_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_file() else None
    env_gt = os.environ.get("WORLDFOUNDRY_MIRABENCH_GT_META_CSV")
    if env_gt:
        gt_path = Path(env_gt).expanduser().resolve()
        return gt_path if gt_path.is_file() else None
    bundled = bundled_benchmark_asset("mirabench", Path("data/evaluation_example/meta_gt.csv"))
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_mirabench_root()
    if root is None:
        return None
    candidate = root / "data/evaluation_example/meta_gt.csv"
    return candidate if candidate.is_file() else None


def resolve_calculate_score_script(*, repo_root: Path | None = None) -> Path | None:
    root = repo_root or resolve_mirabench_root()
    if root is None:
        return None
    candidate = root / "calculate_score.py"
    return candidate if candidate.is_file() else None


def resolve_ckpt_path(*, repo_root: Path | None = None) -> Path:
    env_ckpt = os.environ.get("WORLDFOUNDRY_MIRABENCH_CKPT_PATH")
    if env_ckpt:
        return Path(env_ckpt).expanduser().resolve()
    root = repo_root or resolve_mirabench_root()
    if root is None:
        return Path("data/ckpt")
    return (root / "data" / "ckpt").resolve()



@dataclass(frozen=True)
class MiraBenchScorerConfig:
    backend: str
    repo_root: Path | None
    meta_csv_path: Path | None
    gt_meta_csv_path: Path | None
    ckpt_path: Path
    device: str
    strict: bool = False


def scorer_config_from_env(*, repo_root: Path | None = None) -> MiraBenchScorerConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_MIRABENCH_SCORER_BACKEND")
        or os.environ.get("WORLDFOUNDRY_MIRABENCH_JUDGE_BACKEND")
        or "mock"
    ).strip().lower()
    repo_root = resolve_mirabench_root(repo_root)
    meta_csv_path = None
    gt_meta_csv_path = None
    try:
        meta_csv_path = resolve_meta_csv_path(repo_root=repo_root)
    except FileNotFoundError:
        meta_csv_path = None
    gt_meta_csv_path = resolve_gt_meta_csv_path(repo_root=repo_root)
    strict = os.environ.get("WORLDFOUNDRY_MIRABENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return MiraBenchScorerConfig(
        backend=backend,
        repo_root=repo_root,
        meta_csv_path=meta_csv_path,
        gt_meta_csv_path=gt_meta_csv_path,
        ckpt_path=resolve_ckpt_path(repo_root=repo_root),
        device=os.environ.get("WORLDFOUNDRY_MIRABENCH_DEVICE", "cuda"),
        strict=strict,
    )


def _deterministic_mock_score(*, seed: str, metric_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{metric_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 25
    higher_is_better = METRIC_SPECS[metric_id]["higher_is_better"]
    if higher_is_better:
        return round(0.55 + bucket / 100.0, 4)
    return round(0.05 + bucket / 500.0, 4)


def _matching_videos(*, generated_artifact_dir: Path, prompt_ids: set[str]) -> list[str]:
    if not generated_artifact_dir.is_dir():
        return []
    matched: list[str] = []
    for path in sorted(generated_artifact_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.stem in prompt_ids:
            matched.append(path.stem)
    return matched


def _write_mock_average_score(*, output_path: Path, seed: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        metric_id: _deterministic_mock_score(seed=seed, metric_id=metric_id)
        for metric_id in METRIC_ORDER
        if metric_id != "mirabench_average"
    }
    component_values = list(payload.values())
    payload["mirabench_average"] = round(sum(component_values) / len(component_values), 4)
    output_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    return output_path


def _run_upstream_calculate_score(
    *,
    repo_root: Path,
    generated_artifact_dir: Path,
    output_dir: Path,
    prompt_records: list[Mapping[str, Any]],
    config: MiraBenchScorerConfig,
    meta_csv_path: Path,
) -> Path:
    script = resolve_calculate_score_script(repo_root=repo_root)
    if script is None:
        raise FileNotFoundError(f"MiraBench calculate_score.py not found under {repo_root}")

    runtime_dir = output_dir / "upstream_scorer"
    frame_dir = runtime_dir / "frames_generated"
    gt_frame_dir = runtime_dir / "frames_gt"
    results_dir = runtime_dir / "results"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    materialized_meta = runtime_dir / "meta_generated.csv"
    write_meta_csv(
        records=[dict(record) for record in prompt_records],
        generated_artifact_dir=generated_artifact_dir,
        output_path=materialized_meta,
    )

    gt_meta = config.gt_meta_csv_path or resolve_gt_meta_csv_path(repo_root=repo_root)
    command = [
        os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable),
        str(script.resolve()),
        "--meta_file",
        str(materialized_meta.resolve()),
        "--frame_dir",
        str(frame_dir.resolve()),
        "--output_folder",
        str(results_dir.resolve()),
        "--ckpt_path",
        str(config.ckpt_path.resolve()),
        "--device",
        config.device,
    ]
    if gt_meta is not None:
        command.extend(
            [
                "--gt_meta_file",
                str(gt_meta.resolve()),
                "--gt_frame_dir",
                str(gt_frame_dir.resolve()),
            ]
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root.resolve()), env.get("PYTHONPATH")) if part
    )
    timeout_value = os.environ.get("WORLDFOUNDRY_MIRABENCH_TIMEOUT_SECONDS", "").strip()
    timeout_seconds = float(timeout_value) if timeout_value else None
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("WORLDFOUNDRY_MIRABENCH_TIMEOUT_SECONDS must be positive")
    stdout_path = runtime_dir / "upstream_stdout.log"
    stderr_path = runtime_dir / "upstream_stderr.log"
    completed = run_logged_subprocess(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=str(repo_root.resolve()),
        env=env,
        timeout=timeout_seconds,
        start_new_session=False,
    )
    if completed.returncode != 0:
        detail = read_text_tail(stderr_path) or read_text_tail(stdout_path)
        raise RuntimeError(
            "MiraBench upstream calculate_score.py failed "
            f"(exit={completed.returncode}): {detail}"
        )

    average_score = results_dir / "average_score.csv"
    if not average_score.is_file():
        raise FileNotFoundError(
            "MiraBench upstream calculate_score.py did not write average_score.csv under "
            f"{results_dir}"
        )
    return average_score


def run_mirabench_scorer(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: MiraBenchScorerConfig,
    meta_csv_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mirabench_average_score.json"
    meta_path = meta_csv_path or config.meta_csv_path
    if meta_path is None:
        meta_path = resolve_meta_csv_path(repo_root=config.repo_root)
    prompt_records = unique_prompt_records(load_prompt_records(meta_csv_path=meta_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_ids = {record["prompt_id"] for record in prompt_records}
    matched_videos = _matching_videos(generated_artifact_dir=generated_artifact_dir, prompt_ids=prompt_ids)

    if config.backend == "mock":
        seed = f"mirabench-mock:{len(matched_videos)}:{limit or 'all'}"
        _write_mock_average_score(output_path=results_path, seed=seed)
        return {
            "backend": "mock",
            "results_path": str(results_path.resolve()),
            "video_count": len(matched_videos),
            "prompt_count": len(prompt_records),
            "meta_csv_path": str(meta_path.resolve()),
        }

    repo_root = config.repo_root
    if repo_root is None:
        raise FileNotFoundError(
            "MiraBench in-tree runtime is missing. Expected runtime/mirabench next to this runner, "
            "or set WORLDFOUNDRY_MIRABENCH_ROOT to an equivalent local copy."
        )

    upstream_average = _run_upstream_calculate_score(
        repo_root=repo_root,
        generated_artifact_dir=generated_artifact_dir,
        output_dir=output_dir,
        prompt_records=prompt_records,
        config=config,
        meta_csv_path=meta_path,
    )
    materialize_file(upstream_average, results_path)
    return {
        "backend": "official",
        "results_path": str(results_path.resolve()),
        "upstream_average_score_path": str(upstream_average.resolve()),
        "repo_root": str(repo_root.resolve()),
        "video_count": len(matched_videos),
        "prompt_count": len(prompt_records),
        "meta_csv_path": str(meta_path.resolve()),
    }
