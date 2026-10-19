from __future__ import annotations

import json
import sys
from pathlib import Path

from worldfoundry.core.execution.process import process_group_alive
from worldfoundry.runtime.in_tree_cli import execute_in_tree
from worldfoundry.runtime.jobs import pid_alive


def test_execute_in_tree_rejects_unchanged_stale_output(tmp_path):
    output = tmp_path / "result.mp4"
    output.write_bytes(b"stale")

    result = execute_in_tree(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        output_path=output,
    )

    assert result["status"] == "failed"
    assert "produced no media artifact" in result["error"]
    assert output.read_bytes() == b"stale"


def test_execute_in_tree_copies_fresh_artifact_from_explicit_search_root(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    output = tmp_path / "normalized.mp4"

    result = execute_in_tree(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated/result.mp4').write_bytes(b'fresh')",
        ],
        cwd=tmp_path,
        output_path=output,
        search_roots=(generated,),
    )

    assert result["status"] == "succeeded"
    assert output.read_bytes() == b"fresh"
    assert result["metadata"]["source_artifact"].endswith("generated/result.mp4")


def test_execute_in_tree_timeout_reaps_owned_process_group(tmp_path: Path) -> None:
    output = tmp_path / "result.mp4"
    command = (
        "import json, os, subprocess, sys, time; "
        "from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "Path('pids.json').write_text(json.dumps([os.getpid(), child.pid])); "
        "print('runtime ready', flush=True); time.sleep(60)"
    )

    result = execute_in_tree(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        output_path=output,
        timeout=2,
    )

    parent_pid, child_pid = json.loads((tmp_path / "pids.json").read_text())
    assert result["status"] == "failed"
    assert result["metadata"]["timed_out"] is True
    assert result["metadata"]["timeout_seconds"] == 2
    assert "runtime ready" in Path(result["metadata"]["log_path"]).read_text()
    assert not output.exists()
    assert not process_group_alive(parent_pid)
    assert not pid_alive(parent_pid)
    assert not pid_alive(child_pid)
