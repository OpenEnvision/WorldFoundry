from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core import cuda_visible_devices_from_device, jsonable
from worldfoundry.runtime.assets import expand_worldfoundry_path


SRC_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_ROOT = SRC_ROOT / "worldfoundry" / "synthesis" / "visual_generation" / "longcat_video" / "longcat_video_runtime"
CHECKPOINT_SUBDIRS = ("tokenizer", "text_encoder", "vae", "scheduler", "dit")
REQUIRED_RUNTIME_FILES = (
    "run_inference_text_to_video.py",
    "run_inference_image_to_video.py",
    "run_inference_video_continuation.py",
    "run_inference_long_video.py",
    "longcat_video/pipeline_longcat_video.py",
)
TASK_SCRIPTS = {
    "t2v": "run_inference_text_to_video.py",
    "i2v": "run_inference_image_to_video.py",
    "vc": "run_inference_video_continuation.py",
    "refine": "run_inference_long_video.py",
}
DEFAULT_TASK = "t2v"
SUPPORTED_TASKS = frozenset(TASK_SCRIPTS)


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expandvars(str(value))
    return Path(expand_worldfoundry_path(expanded)).expanduser().resolve()


def _video_files_in(directory: Path) -> list[Path]:
    return sorted(
        {
            path
            for pattern in ("*.mp4", "*.webm", "*.mov", "*.gif")
            for path in directory.glob(pattern)
        }
    )


def _preferred_video_output(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    by_name = {path.name: path for path in paths}
    for name in (
        "output_t2v_refine.mp4",
        "output_i2v_refine.mp4",
        "output_vc_refine.mp4",
        "output_refine.mp4",
        "output_t2v.mp4",
        "output_t2v_distill.mp4",
        "output_i2v.mp4",
        "output_vc.mp4",
    ):
        if name in by_name:
            return by_name[name]
    return paths[0]


@dataclass(frozen=True)
class LongCatVideoRuntimePlan:
    """Plan payload for a locally executable LongCat-Video run."""

    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    checkpoint_dir: str
    output_dir: str
    output_path: str
    task_type: str
    script: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "env": dict(self.env),
            "workdir": self.workdir,
            "checkpoint_dir": self.checkpoint_dir,
            "output_dir": self.output_dir,
            "output_path": self.output_path,
            "task_type": self.task_type,
            "script": self.script,
        }


class LongCatVideoRuntime:
    """In-tree LongCat-Video runtime shim for locally executable runs."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        task_type: str = DEFAULT_TASK,
        device: str = "cuda",
        loaded: bool = False,
        python_executable: str | Path | None = None,
        runtime_root: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported LongCat-Video task_type: {task_type!r}. Supported tasks are {', '.join(sorted(SUPPORTED_TASKS))}."
            )
        self.checkpoint_dir = _expand_path(checkpoint_dir)
        self.task_type = task_type
        self.device = str(device)
        self.runtime_root = _expand_path(runtime_root) or RUNTIME_ROOT
        self.python_executable = (
            Path(python_executable).expanduser()
            if python_executable is not None
            else Path(os.environ.get("PYTHON") or sys.executable).expanduser()
        )
        self.loaded = bool(loaded)
        self.options = {
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir is not None else "",
            "task_type": self.task_type,
            "device": self.device,
        }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path | None,
        task_type: str = DEFAULT_TASK,
        device: str = "cuda",
        **options: Any,
    ) -> "LongCatVideoRuntime":
        return cls(checkpoint_dir=checkpoint_dir, task_type=task_type, device=device, **options)

    def missing_checkpoint_files(self) -> tuple[str, ...]:
        if self.checkpoint_dir is None:
            return tuple(
                str(Path(relative_path)) for relative_path in CHECKPOINT_SUBDIRS
            )
        return tuple(
            str(self.checkpoint_dir / relative_path)
            for relative_path in CHECKPOINT_SUBDIRS
            if not (self.checkpoint_dir / relative_path).is_dir()
        )

    def missing_runtime_files(self) -> tuple[str, ...]:
        return tuple(
            str(self.runtime_root / relative_path)
            for relative_path in REQUIRED_RUNTIME_FILES
            if not (self.runtime_root / relative_path).is_file()
            and not (self.runtime_root / relative_path).is_dir()
        )

    def preflight(self) -> dict[str, Any]:
        missing_checkpoint = self.missing_checkpoint_files()
        missing_runtime = self.missing_runtime_files()
        checkpoint_ready = len(missing_checkpoint) == 0 and self.checkpoint_dir is not None and self.checkpoint_dir.is_dir()
        runtime_ready = len(missing_runtime) == 0
        return {
            "status": "ready" if checkpoint_ready and runtime_ready else "blocked",
            "checkpoint_ready": checkpoint_ready,
            "runtime_ready": runtime_ready,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir is not None else "",
            "checkpoint_dir_exists": bool(self.checkpoint_dir and self.checkpoint_dir.is_dir()),
            "runtime_root": str(self.runtime_root),
            "runtime_root_exists": self.runtime_root.is_dir(),
            "runtime_scripts": {task: str(self.runtime_root / script) for task, script in TASK_SCRIPTS.items()},
            "missing_checkpoint_files": list(missing_checkpoint),
            "missing_runtime_files": list(missing_runtime),
            "expected_checkpoint_subdirs": list(CHECKPOINT_SUBDIRS),
            "required_runtime_files": list(REQUIRED_RUNTIME_FILES),
            "requires": [
                "Python dependency import surface for longcat_video (transformers, torch, diffusers, torchvision)",
                "checkpoint_dir/tokenizer",
                "checkpoint_dir/text_encoder",
                "checkpoint_dir/vae",
                "checkpoint_dir/scheduler",
                "checkpoint_dir/dit",
            ],
        }

    def build_plan(
        self,
        *,
        request: Mapping[str, Any],
        task_type: str,
        output_dir: str | Path,
        output_path: str | Path | None = None,
        context_parallel_size: int = 1,
        enable_compile: bool = False,
    ) -> LongCatVideoRuntimePlan:
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported LongCat-Video task_type: {task_type!r}. Supported tasks are {', '.join(sorted(SUPPORTED_TASKS))}."
            )
        if self.checkpoint_dir is None:
            raise ValueError("LongCat-Video runtime build_plan requires checkpoint_dir.")
        script_path = self.runtime_root / TASK_SCRIPTS[task_type]
        target_output_dir = Path(output_dir).expanduser().resolve()
        target_output_dir.mkdir(parents=True, exist_ok=True)
        output_path_obj = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else target_output_dir / "longcat_video_report.json"
        )
        context_parallel_size = max(1, int(context_parallel_size))
        script_args = (
            str(script_path),
            "--checkpoint_dir",
            str(self.checkpoint_dir),
            "--context_parallel_size",
            str(context_parallel_size),
        )
        if context_parallel_size > 1:
            command = (
                str(self.python_executable),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(context_parallel_size),
                *script_args,
            )
        else:
            command = (str(self.python_executable), *script_args)
        if enable_compile:
            command = (*command, "--enable_compile")
        env = {
            "PYTHONPATH": f"{self.runtime_root}:{SRC_ROOT}{os.pathsep + os.environ.get('PYTHONPATH', '') if os.environ.get('PYTHONPATH') else ''}",
            "WORLD_EVALS_LONGCAT_TASK": str(task_type),
            "WORLD_EVALS_LONGCAT_REQUEST_PATH": str(output_path_obj.with_suffix('.json')),
        }
        if self.device:
            env["WORLD_EVALS_LONGCAT_DEVICE"] = self.device
            visible_devices = cuda_visible_devices_from_device(self.device)
            if visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = visible_devices
        if context_parallel_size == 1:
            env.update(
                {
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(29500 + (os.getpid() % 1000)),
                }
            )
        output_request_path = (
            output_path_obj.with_suffix(".json")
            if output_path_obj.suffix.lower() in {".mp4", ".mov", ".webm", ".gif"}
            else output_path_obj
        )
        env["WORLD_EVALS_LONGCAT_REQUEST_PATH"] = str(output_request_path)
        output_request_path.parent.mkdir(parents=True, exist_ok=True)
        output_request_path.write_text(
            json.dumps(jsonable(request), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return LongCatVideoRuntimePlan(
            command=tuple(command),
            env=env,
            workdir=str(target_output_dir if task_type == "t2v" else self.runtime_root),
            checkpoint_dir=str(self.checkpoint_dir),
            output_dir=str(target_output_dir),
            output_path=str(output_path_obj.with_suffix(".mp4") if output_path_obj.suffix.lower() not in {".json", ".mp4", ".mov", ".webm", ".gif"} else output_path_obj),
            task_type=task_type,
            script=str(script_path),
        )

    def run_plan(
        self,
        plan: LongCatVideoRuntimePlan,
        *,
        timeout_seconds: int = 3600,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        target_log_dir = Path(log_dir or plan.output_dir).expanduser().resolve()
        target_log_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        stdout_path = target_log_dir / "longcat_video_stdout.log"
        stderr_path = target_log_dir / "longcat_video_stderr.log"
        returncode = -1
        generated_files: list[str] = []
        error: str | None = None
        command = list(plan.command)
        env = os.environ.copy()
        env.update(jsonable(plan.env))
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=plan.workdir,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
            )
            returncode = completed.returncode
            runtime_output_dir = Path(plan.workdir).expanduser().resolve()
            requested_output_dir = Path(plan.output_dir).expanduser().resolve()
            runtime_videos = _video_files_in(runtime_output_dir)
            requested_videos = _video_files_in(requested_output_dir)
            preferred = _preferred_video_output(runtime_videos or requested_videos)
            copied_videos: list[Path] = []
            if preferred is not None:
                primary_output_path = Path(plan.output_path).expanduser().resolve()
                if primary_output_path.suffix.lower() not in {".mp4", ".webm", ".mov", ".gif"}:
                    primary_output_path = requested_output_dir / "longcat_video.mp4"
                primary_output_path.parent.mkdir(parents=True, exist_ok=True)
                if preferred.resolve() != primary_output_path.resolve():
                    shutil.copy2(preferred, primary_output_path)
                copied_videos.append(primary_output_path)
                for source in runtime_videos:
                    target = requested_output_dir / source.name
                    if source.resolve() != target.resolve():
                        shutil.copy2(source, target)
                    if target.resolve() != primary_output_path.resolve():
                        copied_videos.append(target)
            materialized_videos = _video_files_in(requested_output_dir)
            generated_files = [str(path) for path in dict.fromkeys([*copied_videos, *requested_videos, *materialized_videos])]
            for source in runtime_videos:
                if source.parent == runtime_output_dir and source.parent != requested_output_dir:
                    source.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - environment-dependent subprocess path
            error = str(exc)
            generated_files = []
        return {
            "ok": returncode == 0 and bool(generated_files),
            "status": "success" if returncode == 0 and bool(generated_files) else "failed",
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_count": len(generated_files),
            "generated_files": generated_files,
            "error": error,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "runtime_plan": plan.to_dict(),
            "run_dir": str(plan.output_dir),
        }


__all__ = [
    "LongCatVideoRuntime",
    "LongCatVideoRuntimePlan",
]
