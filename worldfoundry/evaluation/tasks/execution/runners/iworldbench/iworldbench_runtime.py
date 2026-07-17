"""iWorld-Bench official evaluator runtime (mock reports or upstream dispatch)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_metrics import METRIC_ORDER
from worldfoundry.evaluation.tasks.execution.runners.iworldbench.iworldbench_prompts import (
    resolve_iworldbench_root,
)
from worldfoundry.evaluation.tasks.execution.runners.vbench.vbench_official_impl import IN_TREE_VBENCH_ROOT
from worldfoundry.evaluation.utils import REPO_ROOT

EVAL_SCRIPT_REL = Path("run_iworldbench_evaluation.py")
REPORTS_DIR_NAME = "reports"

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def resolve_evaluation_script(*, repo_root: Path | None = None) -> Path | None:
    root = repo_root or resolve_iworldbench_root()
    if root is None:
        return None
    candidate = root / EVAL_SCRIPT_REL
    return candidate if candidate.is_file() else None


def resolve_camera_txt_dir(*, repo_root: Path | None = None) -> Path | None:
    root = repo_root or resolve_iworldbench_root()
    if root is None:
        return None
    candidate = root / "camera_trajectories" / "inference_txt"
    return candidate if candidate.is_dir() else None


def resolve_reference_npz_dir(*, repo_root: Path | None = None, metric: str = "all") -> Path | None:
    root = repo_root or resolve_iworldbench_root()
    if root is None:
        return None
    relative = (
        "camera_trajectories/source_reference_npz"
        if metric == "camera_following"
        else "camera_trajectories/reference_npz"
    )
    candidate = root / relative
    return candidate if candidate.is_dir() else None


def discover_report_results(search_roots: list[Path]) -> Path | None:
    patterns = ("**/reports", "reports")
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}:
            return root
        if root.is_dir() and root.name == REPORTS_DIR_NAME:
            return root
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[-1]
    return None


@dataclass(frozen=True)
class IWorldBenchRuntimeConfig:
    backend: str
    repo_root: Path | None
    metric: str
    vbench_gpu: str
    timeout: int
    strict: bool = False


def runtime_config_from_env(
    *,
    metric: str | None = None,
    repo_root: Path | None = None,
) -> IWorldBenchRuntimeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_RUNTIME_BACKEND")
        or os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_SCORER_BACKEND")
        or "official"
    ).strip().lower()
    if backend not in {"official", "mock"}:
        raise ValueError(
            "WORLDFOUNDRY_IWORLD_BENCH_RUNTIME_BACKEND must be 'official' or the explicit CI-only 'mock'"
        )
    strict = os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return IWorldBenchRuntimeConfig(
        backend=backend,
        repo_root=resolve_iworldbench_root(repo_root),
        metric=metric or os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_METRIC", "memory"),
        vbench_gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        timeout=int(os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_TIMEOUT", "3600")),
        strict=strict,
    )


def _deterministic_mock_score(*, seed: str, metric_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{metric_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 25
    return round(0.55 + bucket / 100.0, 4)


def _matching_videos(*, generated_artifact_dir: Path) -> list[str]:
    if not generated_artifact_dir.is_dir():
        return []
    matched: list[str] = []
    for path in sorted(generated_artifact_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            matched.append(path.stem)
    return matched


def _write_mock_report_csv(*, output_dir: Path, seed: str) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "iworldbench_mock_summary.csv"
    rows = [
        {"metric_id": metric_id, "score": _deterministic_mock_score(seed=seed, metric_id=metric_id)}
        for metric_id in METRIC_ORDER
        if metric_id != "iworldbench_average"
    ]
    component_values = [float(row["score"]) for row in rows]
    rows.append(
        {
            "metric_id": "iworldbench_average",
            "score": round(sum(component_values) / len(component_values), 4),
        }
    )
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=["metric_id", "score"])
        writer.writeheader()
        writer.writerows(rows)
    return reports_dir


def _build_upstream_command(
    *,
    repo_root: Path,
    generated_artifact_dir: Path,
    upstream_output_dir: Path,
    config: IWorldBenchRuntimeConfig,
) -> list[str]:
    script = resolve_evaluation_script(repo_root=repo_root)
    if script is None:
        raise FileNotFoundError(f"missing iWorld-Bench runner: {repo_root / 'run_iworldbench_evaluation.py'}")
    command = [
        os.environ.get("WORLDFOUNDRY_IWORLD_BENCH_PYTHON")
        or os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable),
        str(script),
        str(generated_artifact_dir.resolve()),
        str(upstream_output_dir.resolve()),
        "--metric",
        config.metric,
        "--iworld-root",
        str(repo_root.resolve()),
        "--vbench-root",
        str(IN_TREE_VBENCH_ROOT.resolve()),
    ]
    if config.metric in {"all", "action_control"}:
        camera_txt = resolve_camera_txt_dir(repo_root=repo_root)
        reference_npz = resolve_reference_npz_dir(repo_root=repo_root, metric=config.metric)
        if camera_txt is not None:
            command.extend(["--camera-txt-dir", str(camera_txt)])
        if reference_npz is not None:
            command.extend(["--source-npz-dir", str(reference_npz)])
    if config.metric == "camera_following":
        reference_npz = resolve_reference_npz_dir(repo_root=repo_root, metric="camera_following")
        if reference_npz is not None:
            command.extend(["--source-npz-dir", str(reference_npz)])
    if config.vbench_gpu is not None:
        command.extend(["--vbench-gpu", str(config.vbench_gpu)])
    return command


def run_iworldbench_evaluator(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: IWorldBenchRuntimeConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_videos = _matching_videos(generated_artifact_dir=generated_artifact_dir)

    if config.backend == "mock":
        seed = f"iworldbench-mock:{config.metric}:{len(matched_videos)}"
        reports_dir = _write_mock_report_csv(output_dir=output_dir, seed=seed)
        return {
            "backend": "mock",
            "results_path": str(reports_dir.resolve()),
            "video_count": len(matched_videos),
            "metric": config.metric,
        }

    repo_root = config.repo_root
    if repo_root is None:
        raise FileNotFoundError(
            "iWorld-Bench in-tree runtime is missing. Expected runtime/iworldbench next to this runner, "
            "or set WORLDFOUNDRY_IWORLD_BENCH_ROOT to an equivalent local copy."
        )
    upstream_output_dir = output_dir / "upstream"
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    command = _build_upstream_command(
        repo_root=repo_root,
        generated_artifact_dir=generated_artifact_dir,
        upstream_output_dir=upstream_output_dir,
        config=config,
    )
    completed = subprocess.run(
        command,
        cwd=str(repo_root.resolve()),
        text=True,
        capture_output=True,
        check=False,
        timeout=config.timeout,
        env={
            **os.environ.copy(),
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (
                    str(REPO_ROOT),
                    str(repo_root.resolve()),
                    os.environ.get("PYTHONPATH"),
                )
                if part
            ),
        },
    )
    command_record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    (output_dir / "upstream_command.json").write_text(json.dumps(command_record, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "iWorld-Bench official command failed "
            f"(exit={completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
        )
    reports_dir = upstream_output_dir / "reports"
    discovered = discover_report_results([reports_dir, upstream_output_dir, output_dir])
    if discovered is None:
        raise FileNotFoundError("iWorld-Bench upstream evaluator did not write report files.")
    return {
        "backend": "official",
        "results_path": str(discovered.resolve()),
        "upstream_output_dir": str(upstream_output_dir.resolve()),
        "repo_root": str(repo_root.resolve()),
        "video_count": len(matched_videos),
        "metric": config.metric,
        "upstream_command": command,
    }
