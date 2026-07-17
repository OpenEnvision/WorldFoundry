#!/usr/bin/env python3
"""Public wrapper for building the WRBench D1-D6 main table."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wrbench.eval.aggregate.build_wrbench_vnext_main_table import main


if __name__ == "__main__":
    raise SystemExit(main())
