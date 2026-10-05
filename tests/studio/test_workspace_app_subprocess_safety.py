"""Regression test for SA-8 (PLW1509): no ``preexec_fn`` in threaded server code.

CPython documents ``subprocess.Popen(preexec_fn=...)`` as unsafe in
multi-threaded programs (the child runs Python code between fork and exec and
can deadlock).  ``workspace_app`` launches visualizer subprocesses from
FastAPI's threadpool, so it must use ``start_new_session=True`` — the safe
equivalent of ``preexec_fn=os.setsid`` — instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKSPACE_APP_PATH = (
    Path(__file__).resolve().parents[2] / "worldfoundry" / "studio" / "serving" / "workspace.py"
)


def test_no_preexec_fn_in_workspace_app() -> None:
    tree = ast.parse(WORKSPACE_APP_PATH.read_text())
    offenders = [
        keyword.value.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "preexec_fn"
    ]
    assert offenders == [], (
        f"serving/workspace.py passes preexec_fn at lines {offenders}; use "
        "start_new_session=True instead (thread-unsafe, SA-8/PLW1509)."
    )


def test_visualizer_popen_starts_new_session() -> None:
    """The visualizer Popen must keep process-group isolation via start_new_session."""
    source = WORKSPACE_APP_PATH.read_text()
    assert "start_new_session=" in source
