"""In-tree World-in-World official evaluation runtime adapter."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from worldfoundry.core.io import materialize_file

OFFICIAL_RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime" / "official"

METRIC_FILE_NAMES = {
    "metrics.json",
    "world_in_world_metrics.json",
    "summary.json",
}

REQUIRED_RUNTIME_FILES = (
    "downstream/evaluator.py",
    "downstream/downstream_datasets.py",
    "downstream/utils/saver.py",
    "downstream/process_IGnav_dataset/pickle_dataset.py",
    "downstream/wiw_manip/aggregate_results.py",
    "openeqa/evaluate-predictions.py",
    "openeqa/evaluation/llm_match.py",
    "evaluation/FVD/cal_4metrics.py",
    "evaluation/FVD/calculate_fvd.py",
    "evaluation/FVD/calculate_psnr.py",
    "evaluation/FVD/calculate_ssim.py",
    "evaluation/FVD/calculate_lpips.py",
)

PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "torch": "torch",
    "pandas": "pandas",
    "tabulate": "tabulate",
    "openai": "openai",
}


def _has_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def official_runtime_preflight() -> dict[str, Any]:
    runtime_files = {
        relative: (OFFICIAL_RUNTIME_ROOT / relative).exists()
        for relative in REQUIRED_RUNTIME_FILES
    }
    package_imports = {
        name: _has_import(module)
        for name, module in PACKAGE_IMPORTS.items()
    }
    missing = [
        f"runtime:{name}"
        for name, ok in runtime_files.items()
        if not ok
    ]
    missing.extend(
        f"package:{name}"
        for name, ok in package_imports.items()
        if not ok
    )
    return {
        "runtime_root": str(OFFICIAL_RUNTIME_ROOT),
        "runtime_files": runtime_files,
        "package_imports": package_imports,
        "missing": missing,
        "ok_for_import": not any(not ok for ok in runtime_files.values()),
        "ok_for_full_run": not missing,
    }


def _env_results_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def discover_official_result_file(search_roots: list[Path]) -> Path | None:
    env_path = _env_results_path()
    if env_path is not None:
        return env_path
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.name in METRIC_FILE_NAMES:
            return root
        matches = sorted(
            path
            for name in METRIC_FILE_NAMES
            for path in root.glob(f"**/{name}")
            if path.is_file()
        )
        if matches:
            return matches[-1]
    return None


def run_official_world_in_world_runtime(
    *,
    generated_artifact_dir: Path | None,
    output_dir: Path,
    task: str,
    exp_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "world_in_world_metrics.json"
    search_roots = [output_dir]
    if generated_artifact_dir is not None:
        search_roots.insert(0, generated_artifact_dir)
    source_path = discover_official_result_file(search_roots)
    if source_path is None:
        raise FileNotFoundError(
            "World-in-World in-tree official runtime requires an official evaluator summary. "
            "Set WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH or place metrics.json / "
            "world_in_world_metrics.json under --generated-artifact-dir."
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    materialize_file(source_path, results_path)
    return {
        "backend": "official_runtime",
        "runtime_root": str(OFFICIAL_RUNTIME_ROOT),
        "preflight": official_runtime_preflight(),
        "results_path": str(results_path.resolve()),
        "source_results_path": str(source_path.resolve()),
        "task": task,
        "exp_id": exp_id,
    }


__all__ = [
    "OFFICIAL_RUNTIME_ROOT",
    "discover_official_result_file",
    "official_runtime_preflight",
    "run_official_world_in_world_runtime",
]
