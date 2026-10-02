from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

STUDIO_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
VENDOR_DIR = STUDIO_ASSET_DIR / "vendor"
SPARK_ROOT = VENDOR_DIR / "spark"
THREE_ROOT = VENDOR_DIR / "three"
SPARK_MODULE_PATH = SPARK_ROOT / "spark.module.min.js"
THREE_MODULE_PATH = THREE_ROOT / "three.module.js"
THREE_CORE_MODULE_PATH = THREE_ROOT / "three.core.js"


def local_module_url(path: Path) -> str:
    return f"/gradio_api/file={quote(path.resolve().as_posix(), safe='/')}"


__all__ = [
    "STUDIO_ASSET_DIR",
    "VENDOR_DIR",
    "SPARK_ROOT",
    "THREE_ROOT",
    "SPARK_MODULE_PATH",
    "THREE_MODULE_PATH",
    "THREE_CORE_MODULE_PATH",
    "local_module_url",
]
