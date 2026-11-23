"""Path helpers for the in-tree OpenS2V-Nexus evaluation runtime."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from worldfoundry.base_models.perception_core.detection.yolo_world import ensure_import_path


def opens2v_eval_root() -> Path:
    return Path(__file__).resolve().parent / "eval"


@lru_cache(maxsize=1)
def ensure_opens2v_eval_path() -> Path:
    root = opens2v_eval_root()
    if not root.is_dir():
        raise FileNotFoundError(f"in-tree OpenS2V-Nexus eval runtime not found at {root}")

    ensure_import_path()
    search_paths = [root, root / "utils" / "yoloworld"]
    for path in search_paths:
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root


__all__ = ["ensure_opens2v_eval_path", "opens2v_eval_root"]
