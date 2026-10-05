"""Locate an optional Pi3 source checkout without importing its model code.

Pi3 upstream sources are excluded from the public distribution. Callers can
supply a checkout through WORLDFOUNDRY_PI3_SOURCE_ROOT, or use the ignored local
``point_clouds/pi3`` directory. This adapter itself has no torch dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(
    os.environ.get("WORLDFOUNDRY_PI3_SOURCE_ROOT")
    or Path(__file__).resolve().parent / "pi3"
).expanduser().resolve()


def ensure_import_paths() -> tuple[Path, ...]:
    """Expose a caller-provided Pi3 checkout for camera geometry inference."""
    if not (SOURCE_ROOT / "pi3" / "models" / "pi3x.py").is_file():
        raise FileNotFoundError(
            "Pi3X camera geometry requires a Pi3 source checkout. Set "
            "WORLDFOUNDRY_PI3_SOURCE_ROOT to the checkout containing "
            "pi3/models/pi3x.py. Pi3 sources are not bundled with WorldFoundry."
        )
    source_root = str(SOURCE_ROOT)
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    return (SOURCE_ROOT,)


__all__ = ["SOURCE_ROOT", "ensure_import_paths"]
