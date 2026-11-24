#!/usr/bin/env python3
"""Backward-compatible wrapper around in-tree PhysVidBench QA execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve()
    from worldfoundry.evals.tasks.execution.runners.physvidbench.physvidbench_captions import resolve_caption_base
    from worldfoundry.evals.tasks.execution.runners.physvidbench.physvidbench_judge import (
        judge_config_from_env,
        run_physvidbench_qa,
    )
    from worldfoundry.evals.tasks.execution.runners.physvidbench.physvidbench_prompts import (
        resolve_prompt_manifest_path,
    )

    api_key = (
        os.environ.get("WORLDFOUNDRY_PHYSVIDBENCH_GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key and judge_config_from_env().backend != "mock":
        print("error: set GEMINI_API_KEY before running PhysVidBench QA", file=sys.stderr)
        return 2
    output_csv = repo_root / "output.csv"
    run_physvidbench_qa(
        question_csv=resolve_prompt_manifest_path(repo_root=repo_root),
        caption_base=resolve_caption_base(repo_root=repo_root),
        output_csv=output_csv,
        config=judge_config_from_env(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
