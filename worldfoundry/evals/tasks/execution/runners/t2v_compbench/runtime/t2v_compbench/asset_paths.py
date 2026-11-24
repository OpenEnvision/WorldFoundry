from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.evals.utils import REPO_ROOT

DEFAULT_ASSET_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "assets" / "t2v-compbench"


def asset_path(relative_path: str) -> Path:
    root = Path(os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_ASSETS", DEFAULT_ASSET_ROOT))
    return root.expanduser() / relative_path
