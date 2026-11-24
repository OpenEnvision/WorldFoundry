#!/usr/bin/env python3
"""Official runner for VBench++."""

from __future__ import annotations

import sys

from worldfoundry.evals.tasks.execution.runners.vbench_2_0.vbench_shared_official_impl import main as series_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not any(arg == "--benchmark-id" or arg.startswith("--benchmark-id=") for arg in args):
        args = ["--benchmark-id", "vbench-plus-plus", *args]
    if not any(arg == "--variant" or arg.startswith("--variant=") for arg in args):
        args = ["--variant", "i2v", *args]
    return series_main(
        args,
        variant_choices=("i2v", "long", "trustworthiness"),
        description="Run or normalize VBench++ official outputs.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
