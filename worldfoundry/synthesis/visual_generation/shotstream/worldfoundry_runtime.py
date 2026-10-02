"""Bind WorldFoundry to the official ShotStream causal inference checkout."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, official_runtime_repo_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/KlingAIResearch/ShotStream"
SOURCE_ENV_VAR = "WORLDFOUNDRY_SHOTSTREAM_SOURCE"
SOURCE_DIR_NAME = "ShotStream"
UPSTREAM_ENTRYPOINT = "Inference_Causal.py"

DEFAULT_CHECKPOINT_DIR = checkpoint_root_path("KlingTeam--ShotStream")
DEFAULT_WAN_MODEL_DIR = checkpoint_root_path("Wan-AI--Wan2.1-T2V-1.3B")

# Dynamic requirement checks are authoritative. A static reason would keep a
# fully staged checkout blocked in the shared runtime-manifest facade.
BLOCKED_REASON = ""


def runtime_root() -> Path:
    """Resolve the user-staged official ShotStream checkout."""
    return official_runtime_repo_path(SOURCE_DIR_NAME, specific_env=SOURCE_ENV_VAR)


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _path_option(options: Mapping[str, Any], *names: str, default: Path) -> Path:
    value = _option(options, *names, default=default)
    return expand_worldfoundry_path(str(value))


def _checkpoint_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "shotstream_checkpoint_dir",
        "checkpoint_dir",
        "checkpoint_path",
        "model_path",
        default=DEFAULT_CHECKPOINT_DIR,
    )


def _wan_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "wan_model_dir",
        "base_model_dir",
        "wan_checkpoint_dir",
        default=DEFAULT_WAN_MODEL_DIR,
    )


def _missing_files(root: Path, relative_paths: tuple[str, ...], *, kind: str, label: str) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append({"kind": kind, "path": str(path), "reason": f"{label} is missing or empty"})
    return missing


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report every source, checkpoint, input, and Python dependency gap."""
    del profile
    options = dict(options or {})
    root = Path(str(runtime_root)).expanduser()
    missing: list[dict[str, str]] = []

    if not root.is_dir():
        missing.append(
            {
                "kind": "source_repo",
                "path": str(root),
                "reason": (
                    f"ShotStream checkout is not staged; clone {OFFICIAL_REPO_URL} here or set "
                    f"{SOURCE_ENV_VAR}"
                ),
            }
        )
    elif not (root / UPSTREAM_ENTRYPOINT).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(root / UPSTREAM_ENTRYPOINT),
                "reason": "official ShotStream Inference_Causal.py is missing",
            }
        )

    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "WorldFoundry ShotStream compatibility launcher is missing",
            }
        )

    checkpoint_dir = _checkpoint_dir(options)
    missing.extend(
        _missing_files(
            checkpoint_dir,
            ("default_config.yaml", "shotstream.yaml", "shotstream_merged.pt"),
            kind="checkpoint",
            label="ShotStream checkpoint asset",
        )
    )
    wan_model_dir = _wan_model_dir(options)
    missing.extend(
        _missing_files(
            wan_model_dir,
            (
                "diffusion_pytorch_model.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
            ),
            kind="checkpoint",
            label="Wan2.1-T2V-1.3B base-model asset",
        )
    )
    tokenizer_dir = wan_model_dir / "google" / "umt5-xxl"
    if not tokenizer_dir.is_dir():
        missing.append(
            {
                "kind": "checkpoint",
                "path": str(tokenizer_dir),
                "reason": "Wan2.1 umT5 tokenizer directory is missing",
            }
        )

    data_path = _option(options, "data_path", "input_csv", "dataset_path")
    if data_path not in (None, ""):
        resolved_data_path = expand_worldfoundry_path(str(data_path))
        if not resolved_data_path.is_file():
            missing.append(
                {"kind": "asset", "path": str(resolved_data_path), "reason": "ShotStream input CSV is missing"}
            )

    for module_name in (
        "torch", "torchvision", "omegaconf", "pandas", "decord", "flash_attn", "easydict", "flask"
    ):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required ShotStream dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    """Build the compatibility-launcher command for official causal inference."""
    settings = command_settings(context)
    checkpoint_dir = _checkpoint_dir(settings)
    wan_model_dir = _wan_model_dir(settings)
    command = [
        str(context["python"]),
        str(context["entrypoint"]),
        "--source-dir",
        str(context["runtime_root"]),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--wan-model-dir",
        str(wan_model_dir),
        "--output-path",
        str(context["output_path"]),
        "--prompt",
        str(context.get("prompt") or settings.get("prompt") or "A cinematic scene with smooth natural motion."),
        "--seed",
        str(settings.get("seed", 42)),
    ]
    data_path = _option(settings, "data_path", "input_csv", "dataset_path")
    if data_path not in (None, ""):
        command.extend(["--input-csv", str(expand_worldfoundry_path(str(data_path)))])
    return command


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_WAN_MODEL_DIR",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "SOURCE_ENV_VAR",
    "build_command",
    "missing_requirements",
    "runtime_root",
]
