#!/usr/bin/env python3
"""Bootstrap local import paths for 3D metrics scripts."""

from __future__ import annotations

import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DEPTH_ANYTHING_3_SRC = REPO_ROOT / "Depth-Anything-3" / "src"


def setup_paths() -> tuple[Path, Path]:
    """Ensure local project dependencies are importable."""
    for path in (CURRENT_DIR, REPO_ROOT, DEPTH_ANYTHING_3_SRC):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return REPO_ROOT, DEPTH_ANYTHING_3_SRC
