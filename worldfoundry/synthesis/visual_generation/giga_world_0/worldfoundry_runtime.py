"""Bind WorldFoundry to the released GigaWorld-0 GR1 checkpoint runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, project_root
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings


PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PACKAGE_DIR / "giga_world_0_runtime"
OFFICIAL_ENTRYPOINT = PACKAGE_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/open-gigaai/giga-world-0"

DEFAULT_TRANSFORMER_MODEL_DIR = checkpoint_root_path(
    "open-gigaai--GigaWorld-0-Video-GR1-2b", "transformer"
)
DEFAULT_TEXT_ENCODER_MODEL_DIR = checkpoint_root_path("google-t5--t5-11b-encoder")
DEFAULT_VAE_MODEL_DIR = checkpoint_root_path(
    "Wan-AI--Wan2.1-T2V-1.3B-Diffusers", "vae"
)
DEFAULT_INPUT_IMAGE = (
    project_root(__file__)
    / "worldfoundry"
    / "data"
    / "test_cases"
    / "test_vla_case1"
    / "droid"
    / "exterior_image_1_left.png"
)
DEFAULT_PROMPT = "A robot arm moves toward the objects on the workbench."
BLOCKED_REASON = ""


def runtime_root() -> Path:
    return RUNTIME_DIR


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _path_option(options: Mapping[str, Any], *names: str, default: Path) -> Path:
    return expand_worldfoundry_path(str(_option(options, *names, default=default)))


def _transformer_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "transformer_model_dir",
        "transformer_model_path",
        "checkpoint_path",
        "checkpoint",
        "model_path",
        default=DEFAULT_TRANSFORMER_MODEL_DIR,
    )


def _text_encoder_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "text_encoder_model_dir",
        "text_encoder_model_path",
        default=DEFAULT_TEXT_ENCODER_MODEL_DIR,
    )


def _vae_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "vae_model_dir",
        "vae_model_path",
        default=DEFAULT_VAE_MODEL_DIR,
    )


def _missing_file(path: Path, reason: str, *, kind: str = "checkpoint") -> dict[str, str] | None:
    try:
        ready = path.is_file() and path.stat().st_size > 0
    except OSError:
        ready = False
    return None if ready else {"kind": kind, "path": str(path), "reason": reason}


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    del profile
    options = dict(options or {})
    source = Path(str(runtime_root)).expanduser()
    missing: list[dict[str, str]] = []

    for relative in ("scripts/inference.py",):
        path = source / relative
        if not path.is_file():
            missing.append(
                {
                    "kind": "source_repo",
                    "path": str(path),
                    "reason": "vendored GigaWorld-0 runtime file is missing",
                }
            )
    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "WorldFoundry GigaWorld-0 compatibility launcher is missing",
            }
        )

    for path, reason in (
        (
            _transformer_model_dir(options) / "config.json",
            "GigaWorld-0 GR1 transformer config is missing or empty",
        ),
        (
            _transformer_model_dir(options) / "diffusion_pytorch_model.safetensors",
            "GigaWorld-0 GR1 transformer weights are missing or empty",
        ),
        (
            _text_encoder_model_dir(options) / "model.safetensors",
            "safe local T5-11B encoder weights are missing or empty",
        ),
        (
            _text_encoder_model_dir(options) / "config.json",
            "T5-11B encoder config is missing or empty",
        ),
        (
            _text_encoder_model_dir(options) / "spiece.model",
            "T5-11B tokenizer is missing or empty",
        ),
        (
            _vae_model_dir(options) / "config.json",
            "Wan Diffusers VAE config is missing or empty",
        ),
        (
            _vae_model_dir(options) / "diffusion_pytorch_model.safetensors",
            "Wan Diffusers VAE weights are missing or empty",
        ),
    ):
        item = _missing_file(path, reason)
        if item is not None:
            missing.append(item)

    image = _path_option(options, "image_path", "input_image", default=DEFAULT_INPUT_IMAGE)
    if not image.is_file():
        missing.append(
            {"kind": "asset", "path": str(image), "reason": "GigaWorld-0 input image is missing"}
        )

    modules = [
        "torch",
        "diffusers",
        "transformers",
        "PIL",
        "safetensors",
        "einops",
        "accelerate",
        "peft",
        "giga_models",
    ]
    if str(options.get("attention_backend", "natten")) == "natten":
        modules.append("natten")
    for module_name in modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required GigaWorld-0 dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    settings = command_settings(context)
    image = _path_option(settings, "image_path", "input_image", default=DEFAULT_INPUT_IMAGE)
    return [
        str(context["python"]),
        str(context["entrypoint"]),
        "--transformer-model-dir",
        str(_transformer_model_dir(settings)),
        "--text-encoder-model-dir",
        str(_text_encoder_model_dir(settings)),
        "--vae-model-dir",
        str(_vae_model_dir(settings)),
        "--input-image",
        str(image),
        "--output-path",
        str(context["output_path"]),
        "--prompt",
        str(context.get("prompt") or DEFAULT_PROMPT),
        "--device",
        str(context.get("device") or "cuda"),
        "--attention-backend",
        str(settings.get("attention_backend", "natten")),
        "--height",
        str(settings.get("height", 480)),
        "--width",
        str(settings.get("width", 640)),
        "--num-frames",
        str(settings.get("num_frames", 61)),
        "--num-inference-steps",
        str(settings.get("num_inference_steps", 30)),
        "--guidance-scale",
        str(settings.get("guidance_scale", 1.0)),
        "--fps",
        str(settings.get("fps", 16)),
        "--seed",
        str(settings.get("seed", 6666)),
        "--max-text-length",
        str(settings.get("max_text_length", 512)),
        "--mixed-precision",
        str(settings.get("mixed_precision", "bf16")),
    ]


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_INPUT_IMAGE",
    "DEFAULT_PROMPT",
    "DEFAULT_TEXT_ENCODER_MODEL_DIR",
    "DEFAULT_TRANSFORMER_MODEL_DIR",
    "DEFAULT_VAE_MODEL_DIR",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
    "runtime_root",
]
