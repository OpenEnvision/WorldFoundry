from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT

for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# Manual inference demos are run directly, never imported during collection.
# Both directories are also excluded by tool.pytest.ini_options.norecursedirs.
collect_ignore = ["manual", "stream"]
