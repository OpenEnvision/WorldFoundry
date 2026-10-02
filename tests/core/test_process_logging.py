from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from worldfoundry.core.execution.process import process_group_alive, run_logged_subprocess, terminate_process_group
from worldfoundry.core.observability.logging_setup import log_context


def test_logged_subprocess_persists_lifecycle_and_parent_context(tmp_path):
    stdout_path = tmp_path / "worker.stdout.log"
    stderr_path = tmp_path / "worker.stderr.log"

    with log_context(run_id="run-process", benchmark_id="bench-process"):
        completed = run_logged_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "from worldfoundry.core.observability.logging_setup import configure_logging, get_logger; "
                    "configure_logging(); "
                    "get_logger('child').event('INFO', 'child.ready', 'child logger ready'); "
                    "print('worker stdout')"
                ),
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    assert completed.returncode == 0
    assert stdout_path.read_text() == "worker stdout\n"
    lifecycle_path = tmp_path / "logs" / "worker.stdout.events.jsonl"
    events = [json.loads(line) for line in lifecycle_path.read_text().splitlines() if line]
    assert [event["event"] for event in events] == ["subprocess.started", "child.ready", "subprocess.finished"]
    assert all(event["run_id"] == "run-process" for event in events)
    assert all(event["benchmark_id"] == "bench-process" for event in events)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_parent_exit_does_not_leave_a_term_ignoring_grandchild(tmp_path):
    ready = tmp_path / "grandchild-ready"
    child_script = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); time.sleep(120)"
    )
    parent_script = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
        f"ready=Path({str(ready)!r});\n"
        "while not ready.exists(): time.sleep(0.01)\n"
        "print(child.pid,flush=True); time.sleep(120)"
    )
    process = subprocess.Popen([sys.executable, "-c", parent_script], stdout=subprocess.PIPE,
                               start_new_session=True, text=True)
    try:
        assert int(process.stdout.readline()) > 1
        terminate_process_group(process, grace_seconds=0.1)
        assert process.poll() is not None
        assert not process_group_alive(process.pid)
    finally:
        terminate_process_group(process, grace_seconds=0)
        process.stdout.close()
