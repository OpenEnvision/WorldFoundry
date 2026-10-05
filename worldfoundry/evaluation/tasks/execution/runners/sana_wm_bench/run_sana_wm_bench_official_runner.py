#!/usr/bin/env python3
"""Entry point for the in-tree SANA-WM 80-scene benchmark evaluator."""

import sys
from pathlib import Path

# Support both ``python -m ...`` and the documented direct script form.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from worldfoundry.evaluation.tasks.execution.runners.sana_wm_bench.sana_wm_bench_official_impl import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
