"""Entry-point for ``python -m worldfoundry.mcp`` — delegates to :func:`worldfoundry.mcp.main`."""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
