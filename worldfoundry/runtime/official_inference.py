"""Shared execution contract for explicitly selected official inference entrypoints.

Only argument vectors are executed. Each call owns a fresh artifact directory;
failed launches and stale videos cannot be reported as successful generation.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from worldfoundry.runtime.assets import expand_worldfoundry_path


def local_path(value):
    return Path(os.path.expandvars(str(expand_worldfoundry_path(str(value))))).expanduser().resolve()


def positive_int(value, name):
    if isinstance(value, bool) or int(value) != float(value) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class InferencePlan:
    model_id: str
    command: tuple[str, ...]
    workdir: str
    output_dir: str
    upstream_revision: str
    env: Mapping[str, str] = field(default_factory=dict)
    artifact_globs: tuple[str, ...] = ("**/*.mp4",)
    completion_globs: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)


class OfficialInferenceRuntime:
    MODEL_ID = ""
    REVISION = ""
    REQUIRED_SOURCE_FILES = ()

    def __init__(self, *, source_root, python_executable=None, device="cuda"):
        self.source_root = local_path(source_root)
        # Preserve virtualenv symlinks: resolving bin/python can select the wrong environment.
        executable = os.path.expandvars(str(python_executable or sys.executable))
        self.python_executable = str(Path(shutil.which(executable) or executable).expanduser().absolute())
        self.device = str(device)
        if self.device != "cuda" and not (
            self.device.startswith("cuda:") and all(v.isdigit() for v in self.device[5:].split(","))
        ):
            raise ValueError(f"{self.MODEL_ID} inference requires device='cuda' or explicit CUDA indices.")

    def required_assets(self):
        return ()

    def preflight(self):
        missing = [
            str(p)
            for p in (
                *[self.source_root / name for name in self.REQUIRED_SOURCE_FILES],
                Path(self.python_executable),
                *self.required_assets(),
            )
            if not p.exists()
        ]
        actual_revision = None
        if (self.source_root / ".git").exists():
            try:
                probe = subprocess.run(
                    ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                actual_revision = probe.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        return {
            "status": "ready" if not missing else "blocked",
            "model_id": self.MODEL_ID,
            "source_revision": actual_revision,
            "source_matches_pin": actual_revision == self.REVISION if actual_revision else None,
            "missing_paths": missing,
            "upstream_revision": self.REVISION,
            "backend": "official_external_runtime",
            "gpu_validation": "pending",
        }

    def environment(self, *python_paths):
        env = {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": os.pathsep.join([*(str(p) for p in python_paths), os.environ.get("PYTHONPATH", "")]),
        }
        if self.device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = self.device[5:]
        return env

    def run_plan(self, plan, *, timeout_seconds=7200):
        timeout_seconds = positive_int(timeout_seconds, "timeout_seconds")
        preflight = self.preflight()
        if preflight["status"] != "ready":
            raise FileNotFoundError(
                f"{self.MODEL_ID} preflight failed: {preflight['missing_paths'] + preflight.get('missing_options', [])}"
            )
        output = Path(plan.output_dir)
        # Some adapters prepare request files here before launch; pre-existing artifacts are rejected.
        if output.exists() and any(output.rglob("*.mp4")):
            raise FileExistsError(f"Inference output already contains videos: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "worldfoundry-plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
        stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
        env = os.environ.copy()
        env.update(plan.env)
        error = None
        code = None
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            try:
                process = subprocess.Popen(
                    plan.command, cwd=plan.workdir, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                )
                try:
                    code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = process.wait()
                    error = f"Inference exceeded {timeout_seconds} seconds"
            except OSError as exc:
                error = f"Inference launch failed: {exc}"
                stderr.write(error + "\n")
        artifacts = sorted(
            {
                str(p)
                for pattern in plan.artifact_globs
                for p in output.glob(pattern)
                if p.is_file() and p.stat().st_size > 0 and not p.name.endswith(".compare.mp4")
            }
        )

        def completed(path):
            try:
                marker = json.loads(path.read_text())
                return isinstance(marker, dict) and marker.get("status") not in {"failed", "error", "incomplete"}
            except (OSError, ValueError):
                return False

        complete = all(any(completed(p) for p in output.glob(pattern)) for pattern in plan.completion_globs)
        success = code == 0 and bool(artifacts) and complete
        result = {
            "status": "success" if success else "failed",
            "model_id": plan.model_id,
            "artifact_kind": "generated_world",
            "artifact_files": artifacts,
            "artifact_path": artifacts[0] if success else None,
            "returncode": code,
            "runtime": "official_external_runtime",
            "upstream_revision": plan.upstream_revision,
            "source_revision": preflight["source_revision"],
            "source_matches_pin": preflight["source_matches_pin"],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        if not success:
            result["error"] = (
                error or f"Inference incomplete (exit={code}, artifacts={len(artifacts)}, completion={complete})"
            )
        report = output / "worldfoundry-result.json"
        result["metadata_path"] = str(report)
        report.write_text(json.dumps(result, indent=2) + "\n")
        return result
