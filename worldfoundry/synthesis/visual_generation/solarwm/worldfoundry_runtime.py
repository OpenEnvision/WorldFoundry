"""Bind WorldFoundry to SolarWM Wan2.2-5B Stage2 inference through the vendored solarwm package."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io import resolve_data_path
from worldfoundry.core.io.paths import checkpoint_root_path, local_data_root_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
ENTRYPOINT_MODULE = "worldfoundry.synthesis.visual_generation.solarwm.infer"
OFFICIAL_REPO_URL = "https://github.com/Junchao-cs/SolarWM"
CONFIG_ROOT = Path(resolve_data_path("models/runtime/configs/solarwm"))

DEFAULT_RELEASE_ROOT = checkpoint_root_path("junchaoh-cs--SolarWM-Wan2.2-5B")
DEFAULT_BASE_MODEL_DIR = DEFAULT_RELEASE_ROOT / "SolarWM-5B-base"
DEFAULT_CHECKPOINT_DIR = DEFAULT_RELEASE_ROOT / "SolarWM-5B-sgf-stage2-81f"
DEFAULT_DATA_ROOT = local_data_root_path() / "SolarWM-Data" / "releases-v1"
DEFAULT_CONFIG_RELATIVE = Path("examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
DEFAULT_TEST_INDEX = Path("recipes/clean-81f/raw-wds/test-index.jsonl.gz")

# Requirement diagnostics, rather than a static block, decide whether a staged
# checkpoint/data installation can execute.
BLOCKED_REASON = ""


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _path_option(options: Mapping[str, Any], *names: str, default: Path) -> Path:
    return expand_worldfoundry_path(str(_option(options, *names, default=default)))


def _base_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "base_model_dir",
        "model_base_path",
        "base_path",
        default=DEFAULT_BASE_MODEL_DIR,
    )


def _checkpoint_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "solarwm_checkpoint_dir",
        "checkpoint_dir",
        "checkpoint_path",
        "model_path",
        default=DEFAULT_CHECKPOINT_DIR,
    )


def _data_root(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "data_root",
        "dataset_root",
        "index_root",
        default=DEFAULT_DATA_ROOT,
    )


def _config_path(options: Mapping[str, Any]) -> Path:
    """Resolve the inference config; relative paths name configs bundled under ``CONFIG_ROOT``."""
    raw = _option(options, "config_path", "solarwm_config", "config")
    if raw in (None, ""):
        return CONFIG_ROOT / DEFAULT_CONFIG_RELATIVE
    path = expand_worldfoundry_path(str(raw))
    return path if path.is_absolute() else CONFIG_ROOT / path


def _test_index(options: Mapping[str, Any]) -> Path:
    value = _option(options, "test_index", "dataset_index", default=DEFAULT_TEST_INDEX)
    return Path(str(value))


def _sample_count(options: Mapping[str, Any]) -> int:
    try:
        sample_count = int(_option(options, "sample_count", default=1))
    except (TypeError, ValueError) as exc:
        raise ValueError("SolarWM sample_count must be an integer") from exc
    if sample_count != 1:
        raise ValueError("SolarWM's single-MP4 WorldFoundry binding requires sample_count=1")
    return sample_count


def _missing_file(path: Path, *, kind: str, reason: str) -> dict[str, str] | None:
    if path.is_file() and path.stat().st_size > 0:
        return None
    return {"kind": kind, "path": str(path), "reason": reason}


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report every checkpoint, dataset, configuration, and dependency gap."""
    del runtime_root, entrypoint, profile
    options = dict(options or {})
    missing: list[dict[str, str]] = []

    config_path = _config_path(options)
    item = _missing_file(config_path, kind="config", reason="SolarWM inference config is missing or empty")
    if item is not None:
        missing.append(item)

    base_model = _base_model_dir(options)
    for relative in (
        "text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "vae/Wan2.2_VAE.pth",
    ):
        path = base_model / relative
        item = _missing_file(path, kind="checkpoint", reason="SolarWM Wan2.2-5B base-model asset is missing or empty")
        if item is not None:
            missing.append(item)

    checkpoint = _checkpoint_dir(options)
    for relative in ("COMPLETE.json", "checkpoint-manifest.json", "release-manifest.json", "model.pt"):
        path = checkpoint / relative
        item = _missing_file(path, kind="checkpoint", reason="SolarWM Stage2 checkpoint asset is missing or empty")
        if item is not None:
            missing.append(item)

    data_root = _data_root(options)
    test_index = _test_index(options)
    resolved_test_index = test_index if test_index.is_absolute() else data_root / test_index
    item = _missing_file(
        resolved_test_index,
        kind="dataset",
        reason="SolarWM inference test index is missing or empty",
    )
    if item is not None:
        missing.append(item)

    try:
        _sample_count(options)
    except ValueError as exc:
        missing.append(
            {
                "kind": "configuration",
                "path": "sample_count",
                "reason": str(exc),
            }
        )

    for module_name in ("torch", "diffusers", "transformers", "peft", "flash_attn", "safetensors", "decord"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required SolarWM Wan dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    """Build the launcher command for the vendored SolarWM CLI."""
    settings = command_settings(context)
    args = [
        "--config",
        str(_config_path(settings)),
        "--base-model-dir",
        str(_base_model_dir(settings)),
        "--checkpoint-dir",
        str(_checkpoint_dir(settings)),
        "--data-root",
        str(_data_root(settings)),
        "--output-path",
        str(context["output_path"]),
        "--sample-count",
        str(_sample_count(settings)),
        "--seed",
        str(int(_option(settings, "seed", default=42))),
    ]
    test_index = _option(settings, "test_index", "dataset_index")
    if test_index not in (None, ""):
        args.extend(["--test-index", str(test_index)])
    return package_module_command(ENTRYPOINT_MODULE, args, python_executable=str(context["python"]))


__all__ = [
    "BLOCKED_REASON",
    "CONFIG_ROOT",
    "DEFAULT_BASE_MODEL_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_CONFIG_RELATIVE",
    "DEFAULT_DATA_ROOT",
    "ENTRYPOINT_MODULE",
    "OFFICIAL_ENTRYPOINT",
    "OFFICIAL_REPO_URL",
    "build_command",
    "missing_requirements",
]
