"""Small subprocess helpers for model-owned in-tree inference runtimes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldfoundry.core.io import file_sha256

from .jobs import _decode_process_text, _kill_process_group

MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg"})

# Opt-in global default for ``execute_in_tree(timeout=...)``; unset means no timeout.
IN_TREE_CLI_TIMEOUT_ENV = "WORLDFOUNDRY_IN_TREE_CLI_TIMEOUT_SECONDS"


def _default_cli_timeout() -> float | None:
    """Read the opt-in in-tree CLI timeout from the environment."""
    raw = os.environ.get(IN_TREE_CLI_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def require_path(value: Any, label: str, *, kind: str | None = None) -> Path:
    """Resolve an existing path with a model-specific diagnostic."""
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required runtime path: {label}")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")
    return path


def ensure_in_tree_runtime(path: Path, *, package_file: str | Path) -> Path:
    """Reject external source checkouts; model code must be owned by this tree."""
    package_root = Path(package_file).resolve().parent
    resolved = path.resolve()
    if resolved != package_root and package_root not in resolved.parents:
        raise ValueError(f"runtime source must be in-tree under {package_root}, got {resolved}")
    return resolved


def newest_media(
    roots: Iterable[str | Path],
    *,
    since: float,
    preferred_names: Sequence[str] = (),
) -> Path | None:
    """Return the newest fresh media artifact from model-owned output roots."""
    # Cache (mtime, size) at collection time: files can disappear between the
    # scan and the sort, and re-stating every file during sorting both raced
    # with cleanup (unhandled OSError) and doubled the stat traffic.
    candidates: list[tuple[float, int, Path]] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        if root.is_file():
            try:
                stat = root.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime, stat.st_size, root))
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime >= since - 1.0:
                candidates.append((stat.st_mtime, stat.st_size, path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered = [item[2] for item in candidates]
    for name in preferred_names:
        for path in ordered:
            if path.name == name:
                return path
    return ordered[0] if ordered else None


def execute_in_tree(
    command: Sequence[str | Path],
    *,
    cwd: str | Path,
    output_path: str | Path,
    search_roots: Sequence[str | Path] = (),
    env: Mapping[str, Any] | None = None,
    python_paths: Sequence[str | Path] = (),
    preferred_names: Sequence[str] = (),
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one model-owned CLI and normalize its generated artifact.

    ``timeout`` bounds the model CLI in seconds; on expiry the whole process
    group is killed and a structured ``failed`` result is returned. When it is
    omitted, the ``WORLDFOUNDRY_IN_TREE_CLI_TIMEOUT_SECONDS`` environment
    variable supplies an opt-in default; otherwise the CLI may run unbounded
    (legacy behaviour).
    """
    workdir = require_path(cwd, "in-tree runtime root", kind="dir")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_output = None
    if output.is_file():
        previous_stat = output.stat()
        previous_output = (previous_stat.st_mtime_ns, previous_stat.st_size)
    log_path = output.with_suffix(output.suffix + ".log")
    process_env = os.environ.copy()
    if python_paths:
        process_env["PYTHONPATH"] = os.pathsep.join(
            [*(str(Path(item).resolve()) for item in python_paths), process_env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    rendered = [str(item) for item in command]
    effective_timeout = timeout if timeout is not None else _default_cli_timeout()
    started = time.time()
    timed_out = False
    if effective_timeout is None:
        completed = subprocess.run(
            rendered,
            cwd=workdir,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        returncode = completed.returncode
    else:
        # Model CLIs are the subprocesses most likely to hang (CUDA deadlocks,
        # NCCL waits, stuck downloads). Run them in their own session so the
        # whole process group can be killed on timeout.
        process = subprocess.Popen(
            rendered,
            cwd=workdir,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=sys.platform != "win32",
        )
        try:
            stdout, stderr = process.communicate(timeout=effective_timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            _kill_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as kill_exc:
                stdout = _decode_process_text(kill_exc.stdout or exc.stdout)
                stderr = _decode_process_text(kill_exc.stderr or exc.stderr)
        stdout = _decode_process_text(stdout)
        stderr = _decode_process_text(stderr)
        returncode = 124 if timed_out else process.returncode
    log_path.write_text(stdout + "\n" + stderr, encoding="utf-8")
    if timed_out:
        return {
            "status": "failed",
            "error": (
                f"in-tree model CLI timed out after {effective_timeout}s and its "
                f"process group was killed; see {log_path}"
            ),
            "artifact_path": str(output),
            "metadata": {
                "command": rendered,
                "cwd": str(workdir),
                "log_path": str(log_path),
                "timed_out": True,
                "timeout_seconds": effective_timeout,
            },
        }
    if returncode != 0:
        return {
            "status": "failed",
            "error": f"in-tree model CLI exited with code {returncode}; see {log_path}",
            "artifact_path": str(output),
            "metadata": {"command": rendered, "cwd": str(workdir), "log_path": str(log_path)},
        }
    output_is_fresh = False
    if output.is_file():
        current_stat = output.stat()
        current_output = (current_stat.st_mtime_ns, current_stat.st_size)
        output_is_fresh = (
            current_stat.st_mtime >= started - 1.0
            and (previous_output is None or current_output != previous_output)
        )
    produced = output if output_is_fresh else newest_media(
        search_roots, since=started, preferred_names=preferred_names
    )
    if produced is None:
        return {
            "status": "failed",
            "error": f"in-tree model CLI completed but produced no media artifact; see {log_path}",
            "artifact_path": str(output),
            "metadata": {"command": rendered, "cwd": str(workdir), "log_path": str(log_path)},
        }
    if produced.resolve() != output:
        shutil.copy2(produced, output)
    return {
        "status": "succeeded",
        "video": str(output),
        "artifact_path": str(output),
        "artifact_sha256": file_sha256(output),
        "backend_quality": "in_tree_official_runtime",
        "metadata": {
            "command": rendered,
            "cwd": str(workdir),
            "source_artifact": str(produced),
            "log_path": str(log_path),
        },
    }


__all__ = [
    "ensure_in_tree_runtime",
    "execute_in_tree",
    "newest_media",
    "require_path",
]
