"""Small durable command supervisor; receipts survive the UI/controller exiting."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Launched by absolute filename so arbitrary command working directories and
# source checkouts work without depending on the controller's sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worldfoundry.core.execution.process import process_identity  # noqa: E402


def _members() -> list[int]:
    """Live members of our owned Linux group, excluding this supervisor."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == os.getpgrp() and fields[0] not in {"Z", "X"}:
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def _stop_children(process: subprocess.Popen, *, grace: float = 1.0) -> None:
    if sys.platform.startswith("linux"):
        # Killing the whole group with SIGKILL would also kill the receipt
        # writer. Signal verified members and rescan until no descendants live.
        for sig, budget in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
            deadline = time.monotonic() + budget
            while True:
                members = _members()
                if not members:
                    process.wait(timeout=2)
                    return
                for pid in members:
                    try:
                        if os.getpgid(pid) == os.getpgrp():
                            os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.03)
        if _members():
            raise TimeoutError("command descendants survived SIGKILL")
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-path", required=True)
    parser.add_argument("--completion-token", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command cannot be empty")
    interrupted = False

    def on_term(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    identity = process_identity(os.getpid())
    started = time.monotonic()
    process = None
    status, error, returncode = "failed", None, None
    try:
        if not interrupted:
            process = subprocess.Popen(command)
        timed_out = False
        while process is not None and process.poll() is None and not interrupted:
            if args.timeout is not None and time.monotonic() - started >= args.timeout:
                timed_out = True
                break
            time.sleep(0.03)
        if process is not None:
            _stop_children(process)
            returncode = process.returncode
        if interrupted:
            status, error = "cancelled", "cancelled by request"
        elif timed_out:
            status, error = "failed", f"job timed out after {args.timeout}s"
        else:
            status = "completed" if returncode == 0 else "failed"
            if status == "failed":
                error = f"command exited with code {returncode}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if process is not None:
            _stop_children(process, grace=0)
    payload = {
        "job_id": args.job_id,
        "completion_token": args.completion_token,
        "pid": os.getpid(),
        "process_identity": identity,
        "status": status,
        "returncode": returncode,
        "error": error,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(args.completion_path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
