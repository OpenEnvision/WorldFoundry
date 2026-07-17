"""Paths and import isolation for the vendored WRBench source tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from worldfoundry.evaluation.utils import worldfoundry_data_path


BENCHMARK_ID = "wrbench"
VENDORED_REVISION = "629595dc60ec08a29711af0377280c4ac9dd40bc"
VENDORED_REPOSITORY = "https://github.com/JinPLu/WRBench.git"


def runtime_root() -> Path:
    """Return the directory whose child is the vendored ``wrbench`` package."""
    return Path(__file__).resolve().parent / "runtime"


def package_root() -> Path:
    return runtime_root() / "wrbench"


def resolve_wrbench_root(explicit: Path | None = None) -> Path:
    """Resolve an override or the checked-in runtime root."""
    env_root = os.environ.get("WORLDFOUNDRY_WRBENCH_ROOT")
    candidate = explicit or (Path(env_root) if env_root else runtime_root())
    candidate = candidate.expanduser().resolve()
    package = candidate / "wrbench"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"WRBench package is missing under {candidate}")
    return candidate


def ensure_wrbench_importable(explicit: Path | None = None) -> Path:
    """Put the in-tree runtime first on ``sys.path`` and return its root."""
    root = resolve_wrbench_root(explicit)
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    return root


def benchmark_assets_root(explicit: Path | None = None) -> Path:
    configured = os.environ.get("WORLDFOUNDRY_WRBENCH_ASSETS_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = worldfoundry_data_path("benchmarks", "assets", "wrbench")
    if root.is_dir():
        return root
    if explicit is not None:
        standalone = resolve_wrbench_root(explicit) / "wrbench" / "data"
        if standalone.is_dir():
            return standalone
    raise FileNotFoundError(f"WRBench benchmark assets are missing: {root}")


def natural25_root(explicit: Path | None = None) -> Path:
    return benchmark_assets_root(explicit) / "natural25"


def model_configs_root(explicit: Path | None = None) -> Path:
    return benchmark_assets_root(explicit) / "model_configs"


def prompt_templates_root(explicit: Path | None = None) -> Path:
    return benchmark_assets_root(explicit) / "prompts" / "templates"


def published_results_path(explicit: Path | None = None) -> Path:
    return benchmark_assets_root(explicit) / "results" / "wrbench_23model_results.json"
