"""Bind WorldFoundry to the official HMA continuous and discrete runtimes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, official_runtime_repo_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/liruiw/HMA"
SOURCE_ENV_VAR = "WORLDFOUNDRY_HMA_SOURCE"
SOURCE_DIR_NAME = "HMA"

DEFAULT_CHECKPOINT_DIR = checkpoint_root_path("liruiw--hma-base-cont")
DEFAULT_BASE_MODEL_DIR = checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")
BLOCKED_REASON = ""


def runtime_root() -> Path:
    """Resolve the user-staged official HMA checkout."""
    return official_runtime_repo_path(SOURCE_DIR_NAME, specific_env=SOURCE_ENV_VAR)


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


def _magvit_checkpoint(options: Mapping[str, Any], source: Path) -> Path:
    return _path_option(
        options,
        "hma_magvit_checkpoint",
        "magvit_checkpoint",
        default=source / "data" / "magvit2.ckpt",
    )


def _is_discrete(checkpoint: Path) -> bool:
    config = checkpoint / "config.json"
    if not config.is_file():
        return False
    return json.loads(config.read_text()).get("image_vocab_size") is not None


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report missing source, model, prompt, or Python assets."""
    del profile
    options = dict(options or {})
    source = Path(str(runtime_root)).expanduser()
    missing: list[dict[str, str]] = []

    checkpoint = _checkpoint_dir(options)
    discrete = _is_discrete(checkpoint)
    trajectory_path = _option(options, "trajectory_path", "trajectory_json")
    required_source_files = ["hma/model/st_mask_git.py" if discrete else "hma/model/st_mar.py"]
    if trajectory_path in (None, ""):
        required_source_files.append("assets/langtable_prompt/frame_00.png")
    if not source.is_dir():
        missing.append(
            {
                "kind": "source_repo",
                "path": str(source),
                "reason": f"HMA checkout is not staged; clone {OFFICIAL_REPO_URL} here or set {SOURCE_ENV_VAR}",
            }
        )
    else:
        for relative in required_source_files:
            path = source / relative
            if not path.is_file():
                missing.append(
                    {"kind": "source_repo", "path": str(path), "reason": "official HMA source asset is missing"}
                )

    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "WorldFoundry HMA compatibility launcher is missing",
            }
        )

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
        tokenizer = _magvit_checkpoint(options, source)
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
            missing.append({"kind": "asset", "path": str(trajectory), "reason": "HMA trajectory JSON is missing or empty"})
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
    """Build the official-model compatibility launcher command."""
    settings = command_settings(context)
    source = Path(str(context["runtime_root"]))
    command = [
        str(context["python"]),
        str(context["entrypoint"]),
        "--source-dir",
        str(source),
        "--checkpoint-dir",
        str(_checkpoint_dir(settings)),
        "--base-model-dir",
        str(_base_model_dir(settings)),
        "--magvit-checkpoint",
        str(_magvit_checkpoint(settings, source)),
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
        command.extend(["--trajectory-json", str(expand_worldfoundry_path(str(trajectory_path)))])
    else:
        image_path = _option(
            settings,
            "image_path",
            "input_image",
            default=source / "assets/langtable_prompt/frame_00.png",
        )
        command.extend(
            [
                "--input-image",
                str(expand_worldfoundry_path(str(image_path))),
                "--direction",
                str(settings.get("direction", "right")),
                "--action-scale",
                str(settings.get("action_scale", 0.05)),
            ]
        )
    return command


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_BASE_MODEL_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "SOURCE_ENV_VAR",
    "build_command",
    "missing_requirements",
    "runtime_root",
]
