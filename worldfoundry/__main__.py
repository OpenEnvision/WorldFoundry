"""Module entrypoint enabling ``python -m worldfoundry`` invocation.

Delegates to :func:`worldfoundry.cli.main.main`, which parses CLI arguments
and orchestrates the evaluation pipeline.  ``SystemExit`` is raised
explicitly so that the exit code from ``main`` propagates to the shell.
"""

from __future__ import annotations

from worldfoundry.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
