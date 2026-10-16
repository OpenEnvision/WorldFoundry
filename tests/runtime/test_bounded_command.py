"""Bounded benchmark commands own their process groups on every exit path."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from worldfoundry.core.execution.process import (
    process_group_alive,
    terminate_owned_process_group,
    terminate_process_group,
)
from worldfoundry.runtime.jobs import run_bounded_command


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_sigint_drains_the_bounded_command_group(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    child_script = (
        f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    parent_script = (
        "import sys; from worldfoundry.runtime.jobs import run_bounded_command;\n"
        "try:\n"
        f" run_bounded_command([{sys.executable!r}, '-c', {child_script!r}], timeout=60)\n"
        "except KeyboardInterrupt: sys.exit(130)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_script], start_new_session=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "bounded command did not start"
        child_pid = int(marker.read_text())
        parent.send_signal(signal.SIGINT)
        assert parent.wait(timeout=10) == 130
        assert not process_group_alive(child_pid)
    finally:
        terminate_process_group(parent, grace_seconds=0)
        if child_pid is not None:
            terminate_owned_process_group(child_pid, grace_seconds=0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("timeout", (False, True))
def test_command_exit_stops_remaining_descendants(tmp_path: Path, timeout: bool) -> None:
    marker = tmp_path / "descendant.pid"
    child_script = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"
    parent_script = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"Path({str(marker)!r}).write_text(str(os.getpgrp())); " + ("time.sleep(120)" if timeout else "print('done')")
    )
    pgid = None
    try:
        result = run_bounded_command([sys.executable, "-c", parent_script], timeout=1 if timeout else 10)
        assert marker.exists()
        pgid = int(marker.read_text())
        assert result["returncode"] == (124 if timeout else 0)
        assert result["timed_out"] is timeout
        assert result["kill_stuck"] is False
        assert not process_group_alive(pgid)
    finally:
        if marker.exists():
            terminate_owned_process_group(int(marker.read_text()), grace_seconds=0)
