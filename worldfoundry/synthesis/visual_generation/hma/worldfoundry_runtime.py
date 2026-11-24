"""Bind WorldFoundry to the vendored HMA continuous and discrete runtimes."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import checkpoint_root_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
ENTRYPOINT_MODULE = "worldfoundry.synthesis.visual_generation.hma.infer"
DEFAULT_PROMPT_IMAGE = RUNTIME_DIR / "hma_runtime" / "assets" / "langtable_prompt" / "frame_00.png"

DEFAULT_CHECKPOINT_DIR = checkpoint_root_path("liruiw--hma-base-cont")
DEFAULT_BASE_MODEL_DIR = checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")
DEFAULT_MAGVIT_CHECKPOINT = checkpoint_root_path("1x-technologies--worldmodel", "magvit2.ckpt")
BLOCKED_REASON = ""


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _path_option(options: Mapping[str, Any], *names: str, default: Path) -> Path:
    return expand_worldfoundry_path(str(_option(options, *names, default=default)))


def _checkpoint_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "hma_checkpoint_dir",
        "checkpoint_dir",
        "checkpoint_path",
        "model_path",
        default=DEFAULT_CHECKPOINT_DIR,
    )


def _base_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "svd_base_model_dir",
        "base_model_dir",
        "image_encoder_ckpt",
        default=DEFAULT_BASE_MODEL_DIR,
    )


def _magvit_checkpoint(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "hma_magvit_checkpoint",
        "magvit_checkpoint",
        default=DEFAULT_MAGVIT_CHECKPOINT,
    )


def _is_discrete(checkpoint: Path) -> bool:
    config = checkpoint / "config.json"
    if not config.is_file():
        return False
    return json.loads(config.read_text()).get("image_vocab_size") is not None


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report missing model, prompt, or Python assets."""
    del runtime_root, entrypoint, profile
    options = dict(options or {})
    missing: list[dict[str, str]] = []

    checkpoint = _checkpoint_dir(options)
    discrete = _is_discrete(checkpoint)
    trajectory_path = _option(options, "trajectory_path", "trajectory_json")

    for relative in ("config.json", "model.safetensors"):
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(
                {
                    "kind": "checkpoint",
                    "path": str(path),
                    "reason": "HMA checkpoint asset is missing or empty",
                }
            )

    if discrete:
        tokenizer = _magvit_checkpoint(options)
        if not tokenizer.is_file() or tokenizer.stat().st_size == 0:
            missing.append(
                {"kind": "checkpoint", "path": str(tokenizer), "reason": "MAGVIT2 tokenizer asset is missing or empty"}
            )
    else:
        base_model = _base_model_dir(options)
        for relative in ("vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"):
            path = base_model / relative
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(
                    {"kind": "checkpoint", "path": str(path), "reason": "SVD temporal VAE asset is missing or empty"}
                )

    if trajectory_path not in (None, ""):
        trajectory = expand_worldfoundry_path(str(trajectory_path))
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            missing.append(
                {"kind": "asset", "path": str(trajectory), "reason": "HMA trajectory JSON is missing or empty"}
            )
    image_path = _option(options, "image_path", "input_image")
    if trajectory_path in (None, "") and image_path not in (None, ""):
        image = expand_worldfoundry_path(str(image_path))
        if not image.is_file():
            missing.append({"kind": "asset", "path": str(image), "reason": "HMA prompt image is missing"})

    for module_name in ("torch", "diffusers", "transformers", "PIL", "einops", "mup", "cv2"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required HMA dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    """Build the HMA launcher command."""
    settings = command_settings(context)
    args = [
        "--checkpoint-dir",
        str(_checkpoint_dir(settings)),
        "--base-model-dir",
        str(_base_model_dir(settings)),
        "--magvit-checkpoint",
        str(_magvit_checkpoint(settings)),
        "--output-path",
        str(context["output_path"]),
        "--device",
        str(context.get("device") or "cuda"),
        "--generated-frames",
        str(settings.get("generated_frames", settings.get("num_frames", 6))),
        "--prompt-horizon",
        str(settings.get("prompt_horizon", 3)),
        "--maskgit-steps",
        str(settings.get("maskgit_steps", 2)),
        "--fps",
        str(settings.get("fps", 2)),
        "--seed",
        str(settings.get("seed", 42)),
    ]
    trajectory_path = _option(settings, "trajectory_path", "trajectory_json")
    if trajectory_path not in (None, ""):
        args.extend(["--trajectory-json", str(expand_worldfoundry_path(str(trajectory_path)))])
    else:
        image_path = _option(settings, "image_path", "input_image", default=DEFAULT_PROMPT_IMAGE)
        args.extend(
            [
                "--input-image",
                str(expand_worldfoundry_path(str(image_path))),
                "--direction",
                str(settings.get("direction", "right")),
                "--action-scale",
                str(settings.get("action_scale", 0.05)),
            ]
        )
    return package_module_command(ENTRYPOINT_MODULE, args, python_executable=str(context["python"]))


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_BASE_MODEL_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_MAGVIT_CHECKPOINT",
    "DEFAULT_PROMPT_IMAGE",
    "ENTRYPOINT_MODULE",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
]
