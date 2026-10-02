"""Import-light subprocess launch helpers shared by model runtimes.

Official recipes and eval harnesses spawn workers (single-node
``torchrun``, ffmpeg, vendor scripts) and must stay observable without
pulling a training stack into the parent. This module therefore depends
only on the stdlib plus an optional ``psutil`` for trees that were not
started in their own session.

Public surface:

- :func:`run_logged_subprocess` — bounded stdout/stderr files, JSONL
  lifecycle events, WorldFoundry log-context env, timeout → SIGTERM
  then SIGKILL of the process group (or a psutil tree fallback).
- :func:`terminate_process_group` / :func:`terminate_process_tree` —
  the two teardown strategies used by the launcher.
- :func:`read_text_tail` — last *N* lines of a potentially huge log
  without reading the whole file.
- :func:`torchrun_module_command` / :func:`run_torchrun_module` —
  build or run ``python -m torch.distributed.run --standalone``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Teardown — process-group first, then psutil tree fallback
# ──────────────────────────────────────────────────────────────────────────


def process_identity(pid: int) -> dict[str, str | int] | None:
    """Read a Linux process incarnation, including its boot and session identity."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start_ticks": fields[19],
            "pgid": int(fields[2]),
            "session_id": int(fields[3]),
        }
    except (OSError, IndexError, ValueError):
        return None


def process_group_alive(pgid: int) -> bool:
    """Return whether a dedicated POSIX group contains any non-zombie process."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform.startswith("linux"):
        # A dead grandchild can remain a zombie until init reaps it. It no longer
        # owns CUDA resources and must not make teardown wait indefinitely.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    return True
            except PermissionError:
                try:
                    if os.getpgid(int(entry.name)) == pgid:
                        return True
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    return True


def terminate_owned_process_group(pgid: int, *, grace_seconds: float = 3.0) -> None:
    """Stop an owned session's entire group, even after its leader has exited."""
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError("refusing to terminate the caller's process group")
    for sig, wait_seconds in ((signal.SIGTERM, max(grace_seconds, 0.0)), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait_seconds
        while process_group_alive(pgid):
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
        else:
            return
    if process_group_alive(pgid):
        raise TimeoutError(f"process group {pgid} still alive after SIGKILL")


def terminate_process_group(process: subprocess.Popen[Any], *, grace_seconds: float = 3.0) -> None:
    """Terminate a process and all descendants in its dedicated process group.

    POSIX path sends ``SIGTERM`` to the group (``os.killpg``), waits
    *grace_seconds*, then ``SIGKILL``. Windows falls back to
    ``terminate`` / ``kill`` on the parent only. Already-exited
    groups are a no-op. The child must have been started with
    ``start_new_session=True`` (see :func:`run_logged_subprocess`).
    """

    if os.name == "posix":
        terminate_owned_process_group(process.pid, grace_seconds=grace_seconds)
        process.wait(timeout=2)
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=max(float(grace_seconds), 0.0))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass
    process.wait()


def terminate_process_tree(process: subprocess.Popen[Any], *, grace_seconds: float = 3.0) -> None:
    """Terminate one process and recursively terminate descendants without a process group.

    Used when the child was *not* started in a new session. With
    ``psutil`` installed, children are ``terminate``'d first, then the
    parent, then any survivors are ``kill``'d. Without psutil only the
    parent is signalled — grandchildren may leak.
    """

    if process.poll() is not None:
        return
    descendants: list[Any] = []
    try:
        import psutil
    except ImportError:
        descendants = []
    else:
        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
        except (OSError, psutil.AccessDenied, psutil.NoSuchProcess):
            descendants = []
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=max(float(grace_seconds), 0.0))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
        process.wait()
    if descendants:
        import psutil

        _, alive = psutil.wait_procs(descendants, timeout=max(float(grace_seconds), 0.0))
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if alive:
            psutil.wait_procs(alive, timeout=max(float(grace_seconds), 0.0))


# ──────────────────────────────────────────────────────────────────────────
# Launch + observe — bounded logs, JSONL lifecycle, timeout → SIGTERM/KILL
# ──────────────────────────────────────────────────────────────────────────


def run_logged_subprocess(
    command: str | Sequence[str],
    *,
    stdout_path: str | Path,
    stderr_path: str | Path,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
    append: bool = False,
    start_new_session: bool | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run a subprocess with bounded log memory and timeout-safe descendants.

    The child receives the current WorldFoundry correlation context and gets a
    dedicated JSONL lifecycle artifact beside its raw stdout/stderr captures.
    This keeps framework-owned workers observable without requiring changes to
    vendored official benchmark code.
    """

    from worldfoundry.core.observability.logging_setup import log_context_environment, write_jsonl_event

    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    owns_process_group = os.name == "posix" and start_new_session is not False
    lifecycle_path = stdout_path.parent / "logs" / f"{stdout_path.stem}.events.jsonl"
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    process_env.update(log_context_environment())
    # Avoid concurrent workers rotating the parent CLI's event file.  A child
    # that enters WorldFoundry's CLI/configuration gets its own structured
    # worker sink; third-party scripts simply ignore these variables.
    process_env["WORLDFOUNDRY_LOG_FILE"] = str(lifecycle_path)
    process_env["WORLDFOUNDRY_LOG_JSON"] = "1"
    command_name = (
        command.split(maxsplit=1)[0]
        if isinstance(command, str) and command.strip()
        else str(command[0])
        if not isinstance(command, str) and command
        else ""
    )
    start = time.monotonic()
    write_jsonl_event(
        lifecycle_path,
        level="INFO",
        event="subprocess.started",
        message="WorldFoundry subprocess started",
        logger_name=__name__,
        executable=command_name,
        cwd=None if cwd is None else str(cwd),
        stdout_path=str(stdout_path.resolve()),
        stderr_path=str(stderr_path.resolve()),
        timeout_seconds=timeout,
    )
    with (
        stdout_path.open(mode, encoding="utf-8", errors="replace") as stdout,
        stderr_path.open(mode, encoding="utf-8", errors="replace") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            shell=shell,
            start_new_session=owns_process_group,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if owns_process_group:
                terminate_process_group(process)
            else:
                terminate_process_tree(process)
            write_jsonl_event(
                lifecycle_path,
                level="ERROR",
                event="subprocess.timed_out",
                message="WorldFoundry subprocess timed out",
                logger_name=__name__,
                duration_seconds=round(time.monotonic() - start, 6),
                timeout_seconds=timeout,
            )
            raise subprocess.TimeoutExpired(command, timeout) from None
        finally:
            # The leader exiting does not imply its descendants have stopped.
            if owns_process_group:
                terminate_process_group(process)
            elif process.poll() is None:
                terminate_process_tree(process)
    write_jsonl_event(
        lifecycle_path,
        level="INFO" if returncode == 0 else "ERROR",
        event="subprocess.finished" if returncode == 0 else "subprocess.failed",
        message="WorldFoundry subprocess finished",
        logger_name=__name__,
        returncode=returncode,
        duration_seconds=round(time.monotonic() - start, 6),
    )
    return subprocess.CompletedProcess(command, returncode)


def read_text_tail(
    path: str | Path,
    *,
    max_lines: int = 20,
    max_bytes: int = 64 * 1024,
) -> str:
    """Read a bounded UTF-8 tail from a potentially very large log file.

    Seeks to ``max(size - max_bytes, 0)`` and keeps the last
    *max_lines* decoded lines. A mid-file seek drops the first partial
    line so the result does not start with a torn UTF-8 sequence.
    Missing files yield ``""``.
    """

    if max_lines < 1 or max_bytes < 1:
        raise ValueError("max_lines and max_bytes must be positive")
    resolved = Path(path)
    if not resolved.is_file():
        return ""
    with resolved.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        offset = max(size - int(max_bytes), 0)
        handle.seek(offset)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    if offset:
        _, separator, text = text.partition("\n")
        if not separator:
            text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-int(max_lines) :])


# ──────────────────────────────────────────────────────────────────────────
# torchrun helper — standalone single-node, no training stack in the parent
# ──────────────────────────────────────────────────────────────────────────


def torchrun_module_command(
    module: str,
    *,
    nproc_per_node: int,
    args: Sequence[str] = (),
    python_executable: str = sys.executable,
) -> list[str]:
    """Build a single-node torchrun command for a Python module.

    Equivalent to ``python -m torch.distributed.run --standalone
    --nproc_per_node N -m MODULE ...``. Does not launch the process.
    """

    if nproc_per_node < 1:
        raise ValueError("nproc_per_node must be positive")
    return [
        python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        "-m",
        module,
        *map(str, args),
    ]


def run_torchrun_module(
    module: str,
    *,
    nproc_per_node: int,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess[str]:
    """Run a single-node Python module under torchrun and capture its logs.

    Uses :func:`subprocess.run` with ``capture_output=True`` and
    ``check=False``. Prefer :func:`run_logged_subprocess` when the
    worker may run long enough that in-memory capture is unsafe.
    """

    return subprocess.run(
        torchrun_module_command(
            module,
            nproc_per_node=nproc_per_node,
            args=args,
            python_executable=python_executable,
        ),
        env=None if env is None else dict(env),
        text=True,
        capture_output=True,
        check=False,
    )


__all__ = [
    "process_identity",
    "process_group_alive",
    "terminate_owned_process_group",
    "read_text_tail",
    "run_logged_subprocess",
    "run_torchrun_module",
    "terminate_process_group",
    "terminate_process_tree",
    "torchrun_module_command",
]
