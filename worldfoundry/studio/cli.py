from __future__ import annotations

import os
import sys
from typing import Sequence

_UI_DEPENDENCY_MODULES = {"gradio", "fastapi", "starlette", "uvicorn"}
_FRONTEND_ENV_KEYS = (
    "WORLDFOUNDRY_STUDIO_FRONTEND",
)


def _argv(argv: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def _help_requested(argv: Sequence[str]) -> bool:
    return any(item in {"-h", "--help"} for item in argv)


def _frontend_from_argv(argv: Sequence[str]) -> str:
    for index, item in enumerate(argv):
        if item == "--frontend" and index + 1 < len(argv):
            return argv[index + 1].strip().lower()
        if item.startswith("--frontend="):
            return item.split("=", maxsplit=1)[1].strip().lower()
    return ""


def _frontend_from_env() -> str:
    for key in _FRONTEND_ENV_KEYS:
        value = os.getenv(key, "").strip().lower()
        if value:
            return value
    return ""


def _requires_unified_frontend(argv: Sequence[str]) -> bool:
    return (_frontend_from_argv(argv) or _frontend_from_env()) == "unified"


def _launch_standalone(argv: Sequence[str]) -> None:
    from worldfoundry.studio.ui.launcher import main as standalone_main

    standalone_main(argv)


def _launch_unified(argv: Sequence[str]) -> None:
    try:
        from worldfoundry.studio.ui.gradio_app import main as app_main
    except ModuleNotFoundError as exc:
        missing = exc.name or "optional UI dependency"
        if missing in _UI_DEPENDENCY_MODULES:
            raise SystemExit(
                "WorldFoundry Studio unified frontend requires optional UI dependencies. "
                "Install them with `python -m pip install -e '.[ui]'` and retry. "
                f"Missing: {missing}"
            ) from exc
        raise
    app_main(argv)


def _prepare_cuda_allocator() -> None:
    """Default the CUDA allocator to expandable segments before any torch import.

    Large multi-GPU runtimes (e.g. the LingBot-World fast checkpoint sharded with
    FSDP) can transiently fragment GPU memory while every rank loads its shards in
    parallel, producing ``CUDA error: out of memory`` failures even though the
    settled footprint is small. ``expandable_segments`` lets the allocator reclaim
    fragmented blocks and must be set before CUDA initializes. ``setdefault`` keeps
    any explicit operator override intact.
    """

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main(argv: Sequence[str] | None = None) -> None:
    """Launch Studio without importing the Gradio stack unless the unified UI is requested."""

    _prepare_cuda_allocator()
    args = _argv(argv)
    if _help_requested(args) or not _requires_unified_frontend(args):
        _launch_standalone(args)
        return
    _launch_unified(args)


if __name__ == "__main__":
    main()
