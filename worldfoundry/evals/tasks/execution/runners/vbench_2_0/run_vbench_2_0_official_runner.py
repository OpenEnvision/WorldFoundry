#!/usr/bin/env python3
"""Official runner for VBench 2.0."""

from __future__ import annotations

import sys

from worldfoundry.evals.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import main as series_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not any(arg == "--benchmark-id" or arg.startswith("--benchmark-id=") for arg in args):
        args = ["--benchmark-id", "vbench-2.0", *args]
    if not any(arg == "--variant" or arg.startswith("--variant=") for arg in args):
        args = ["--variant", "vbench2", *args]
    return series_main(
        args,
        variant_choices=("vbench2",),
        description="Run or normalize VBench-2.0 official outputs.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
