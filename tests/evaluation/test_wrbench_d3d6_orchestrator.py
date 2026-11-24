# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import re
import subprocess

from worldfoundry.evals.tasks.execution.runners.wrbench.wrbench_paths import (
    ensure_wrbench_importable,
    runtime_root,
)


def test_d3d6_orchestrator_resolves_inside_the_vendored_runtime() -> None:
    ensure_wrbench_importable()
    eval_runtime = importlib.import_module("wrbench.eval.runtime")

    root = eval_runtime.wrbench_repo_root()
    script = root / "scripts" / "eval" / "score_runtime_v2_d3d6.sh"
    assert root == runtime_root()
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)

    scoring_dir = root / "wrbench" / "eval" / "scoring"
    referenced = set(re.findall(r"\$\{SCORING_DIR\}/(\w+\.py)", script.read_text(encoding="utf-8")))
    assert referenced
    assert [name for name in sorted(referenced) if not (scoring_dir / name).is_file()] == []
