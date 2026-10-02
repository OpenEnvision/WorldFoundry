from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from worldfoundry.evaluation.tasks.execution.framework.artifact_score_runtime import main
from worldfoundry.evaluation.tasks.execution.runners.devil_dynamics.run_devil_dynamics_official_runner import (
    main as devil_dynamics_main,
)


def test_artifact_score_json_failure_keeps_stdout_machine_clean(tmp_path, capsys):
    """A failed ``--json`` invocation must not put diagnostics on stdout."""
    exit_code = main(
        [
            "--benchmark-id",
            "fetv",
            "--score-dir",
            str(tmp_path / "missing-scores"),
            "--output-path",
            str(tmp_path / "scores.json"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error:" in captured.err


def test_official_runner_json_failure_keeps_stdout_machine_clean(capsys):
    """The shared official-runner entrypoint has the same JSON stdout contract."""
    exit_code = devil_dynamics_main(["--json"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "error:" in captured.err
