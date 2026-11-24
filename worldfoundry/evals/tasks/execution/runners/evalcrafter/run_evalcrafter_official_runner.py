#!/usr/bin/env python3
"""Official runner for EvalCrafter."""

from __future__ import annotations

from worldfoundry.evals.tasks.execution.runners.evalcrafter.evalcrafter_official_impl import *  # noqa: F403 - Re-export the official runner interface.
from worldfoundry.evals.tasks.execution.runners.evalcrafter.evalcrafter_official_impl import main

if __name__ == "__main__":
    raise SystemExit(main())
