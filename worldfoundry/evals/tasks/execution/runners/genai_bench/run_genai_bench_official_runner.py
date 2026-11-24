#!/usr/bin/env python3
"""Official runner for GenAI-Bench."""

from __future__ import annotations

from worldfoundry.evals.tasks.execution.runners.genai_bench.genai_bench_official_impl import main

if __name__ == "__main__":
    raise SystemExit(main())
