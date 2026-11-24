#!/usr/bin/env python3
"""Entry point for the in-tree Physical AI Bench evaluator."""

from worldfoundry.evals.tasks.execution.runners.physical_ai_bench.physical_ai_bench_official_impl import (
    main,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
