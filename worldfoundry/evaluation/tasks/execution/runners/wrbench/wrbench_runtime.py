"""Dispatch the checked-in WRBench D1-D6 runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .wrbench_paths import resolve_wrbench_root


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def discover_video_manifest(generated_artifact_dir: Path | None, explicit: Path | None = None) -> Path:
    if explicit is not None and explicit.is_file():
        return explicit.expanduser().resolve()
    env_manifest = _env_path("WORLDFOUNDRY_WRBENCH_VIDEO_MANIFEST") or _env_path("WORLDFOUNDRY_PROMPT_MANIFEST")
    if env_manifest is not None and env_manifest.is_file():
        return env_manifest
    if generated_artifact_dir is not None:
        for name in ("videos.json", "manifest.json", "videos_manifest.json"):
            candidate = generated_artifact_dir / name
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        "WRBench video manifest is required; pass --video-manifest or set "
        "WORLDFOUNDRY_WRBENCH_VIDEO_MANIFEST"
    )


def run_wrbench_evaluator(
    *,
    output_dir: Path,
    generated_artifact_dir: Path | None,
    video_manifest: Path | None = None,
    runtime_config: Path | None = None,
    scorer_profile: str = "wrbench_default",
    sidecar_profile_gate: str = "main",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = resolve_wrbench_root(repo_root)
    manifest = discover_video_manifest(generated_artifact_dir, video_manifest)
    runtime_config = runtime_config or _env_path("WORLDFOUNDRY_WRBENCH_RUNTIME_CONFIG")
    if runtime_config is None or not runtime_config.is_file():
        raise FileNotFoundError(
            "WRBench scorer configuration is required; pass --runtime-config or set "
            "WORLDFOUNDRY_WRBENCH_RUNTIME_CONFIG"
        )
    eval_dir = output_dir / "wrbench_official"
    command = [
        sys.executable,
        "-m",
        "wrbench.cli",
        "eval",
        "--runtime-config",
        str(runtime_config.resolve()),
        "run",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(eval_dir),
        "--scorer-profile",
        scorer_profile,
        "--sidecar-profile-gate",
        sidecar_profile_gate,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(root), env.get("PYTHONPATH")) if part)
    completed = subprocess.run(command, cwd=str(root), env=env, text=True, capture_output=True, check=False)
    stdout_path = output_dir / "wrbench_runtime_stdout.log"
    stderr_path = output_dir / "wrbench_runtime_stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"WRBench D1-D6 runtime failed (exit={completed.returncode}); see {stderr_path}"
        )
    results_path = eval_dir / "main_table.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"WRBench runtime did not produce {results_path}")
    return {
        "backend": "in_tree_wrbench",
        "command": command,
        "returncode": completed.returncode,
        "manifest": str(manifest),
        "runtime_config": str(runtime_config.resolve()),
        "results_path": str(results_path.resolve()),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    }


__all__ = ["discover_video_manifest", "run_wrbench_evaluator"]
