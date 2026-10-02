"""Bind WorldFoundry to the vendored TesserAct RGB-depth-normal runtime."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, project_root
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings


PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PACKAGE_DIR / "tesseract_runtime"
OFFICIAL_ENTRYPOINT = PACKAGE_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/UMass-Embodied-AGI/TesserAct"

DEFAULT_CHECKPOINT_DIR = checkpoint_root_path("anyeZHY--tesseract", "tesseract_v01e_rgbdn_sft")
DEFAULT_BASE_MODEL_DIR = checkpoint_root_path("THUDM--CogVideoX-5b-I2V")
DEFAULT_INPUT_IMAGE = (
    project_root(__file__)
    / "worldfoundry"
    / "data"
    / "test_cases"
    / "test_vla_case1"
    / "droid"
    / "exterior_image_1_left.png"
)
DEFAULT_PROMPT = "The robot moves its gripper toward the object on the table."
BLOCKED_REASON = ""


def runtime_root() -> Path:
    """Return the in-tree official TesserAct runtime source root."""
    return RUNTIME_DIR


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
        "tesseract_checkpoint_dir",
        "checkpoint_dir",
        "checkpoint_path",
        "weights_path",
        "model_path",
        default=DEFAULT_CHECKPOINT_DIR,
    )


def _base_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "cogvideox_base_model_dir",
        "base_model_dir",
        "pretrained_model_path",
        default=DEFAULT_BASE_MODEL_DIR,
    )


def _missing_file(path: Path, reason: str) -> dict[str, str] | None:
    try:
        ready = path.is_file() and path.stat().st_size > 0
    except OSError:
        ready = False
    if ready:
        return None
    return {"kind": "checkpoint", "path": str(path), "reason": reason}


def _sharded_checkpoint_requirements(root: Path, index_name: str, role: str) -> list[dict[str, str]]:
    index_path = root / index_name
    missing = _missing_file(index_path, f"{role} shard index is missing or empty")
    if missing is not None:
        return [missing]
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        shard_names = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []
    except (OSError, json.JSONDecodeError, TypeError):
        shard_names = []
    if not shard_names:
        return [
            {
                "kind": "checkpoint",
                "path": str(index_path),
                "reason": f"{role} shard index has no readable weight map",
            }
        ]
    missing_shards: list[dict[str, str]] = []
    for name in shard_names:
        shard = _missing_file(root / str(name), f"{role} checkpoint shard is missing or empty")
        if shard is not None:
            missing_shards.append(shard)
    return missing_shards


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report missing runtime code, checkpoints, conditioning assets, or modules."""
    del profile
    options = dict(options or {})
    source = Path(str(runtime_root)).expanduser()
    missing: list[dict[str, str]] = []

    for relative in (
        "tesseract/modules/tesseract_pipeline.py",
        "tesseract/modules/tesseract_model.py",
        "tesseract/utils.py",
    ):
        path = source / relative
        if not path.is_file():
            missing.append(
                {"kind": "source_repo", "path": str(path), "reason": "vendored TesserAct runtime file is missing"}
            )

    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "WorldFoundry TesserAct compatibility launcher is missing",
            }
        )

    checkpoint = _checkpoint_dir(options)
    config_missing = _missing_file(checkpoint / "config.json", "TesserAct RGBDN transformer config is missing")
    if config_missing is not None:
        missing.append(config_missing)
    missing.extend(
        _sharded_checkpoint_requirements(
            checkpoint,
            "diffusion_pytorch_model.safetensors.index.json",
            "TesserAct RGBDN transformer",
        )
    )

    base_model = _base_model_dir(options)
    for relative, reason in (
        ("model_index.json", "CogVideoX base model index is missing"),
        ("scheduler/scheduler_config.json", "CogVideoX scheduler config is missing"),
        ("tokenizer/tokenizer_config.json", "CogVideoX tokenizer config is missing"),
        ("tokenizer/spiece.model", "CogVideoX tokenizer vocabulary is missing"),
        ("text_encoder/config.json", "CogVideoX text encoder config is missing"),
        ("vae/config.json", "CogVideoX VAE config is missing"),
        ("vae/diffusion_pytorch_model.safetensors", "CogVideoX VAE weights are missing"),
    ):
        item = _missing_file(base_model / relative, reason)
        if item is not None:
            missing.append(item)
    missing.extend(
        _sharded_checkpoint_requirements(
            base_model / "text_encoder",
            "model.safetensors.index.json",
            "CogVideoX T5 text encoder",
        )
    )

    image = _path_option(options, "image_path", "input_image", default=DEFAULT_INPUT_IMAGE)
    if not image.is_file():
        missing.append({"kind": "asset", "path": str(image), "reason": "TesserAct input image is missing"})

    depth_value = _option(options, "depth_path")
    normal_value = _option(options, "normal_path")
    if bool(depth_value) != bool(normal_value):
        missing.append(
            {
                "kind": "asset",
                "path": str(depth_value or normal_value or ""),
                "reason": "official geometry conditioning requires both depth_path and normal_path",
            }
        )
    for value, label in ((depth_value, "depth"), (normal_value, "normal")):
        if value:
            path = expand_worldfoundry_path(str(value))
            if not path.is_file():
                missing.append(
                    {"kind": "asset", "path": str(path), "reason": f"TesserAct {label} conditioning is missing"}
                )

    for module_name in ("torch", "diffusers", "transformers", "PIL", "accelerate", "safetensors", "cv2", "numpy"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required TesserAct dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    """Build the TesserAct RGBDN compatibility launcher command."""
    settings = command_settings(context)
    image_path = _path_option(settings, "image_path", "input_image", default=DEFAULT_INPUT_IMAGE)
    command = [
        str(context["python"]),
        str(context["entrypoint"]),
        "--checkpoint-dir",
        str(_checkpoint_dir(settings)),
        "--base-model-dir",
        str(_base_model_dir(settings)),
        "--input-image",
        str(image_path),
        "--output-path",
        str(context["output_path"]),
        "--prompt",
        str(context.get("prompt") or DEFAULT_PROMPT),
        "--device",
        str(context.get("device") or "cuda"),
        "--geometry-mode",
        str(settings.get("geometry_mode", "synthetic-gradient")),
        "--height",
        str(settings.get("height", 256)),
        "--width",
        str(settings.get("width", 320)),
        "--num-frames",
        str(settings.get("num_frames", 49)),
        "--num-inference-steps",
        str(settings.get("num_inference_steps", 4)),
        "--guidance-scale",
        str(settings.get("guidance_scale", 7.5)),
        "--image-guidance-scale",
        str(settings.get("image_guidance_scale", 1.5)),
        "--fps",
        str(settings.get("fps", 8)),
        "--seed",
        str(settings.get("seed", 23)),
        "--mixed-precision",
        str(settings.get("mixed_precision", "bf16")),
    ]
    depth_path = _option(settings, "depth_path")
    normal_path = _option(settings, "normal_path")
    if depth_path:
        command.extend(("--depth-path", str(expand_worldfoundry_path(str(depth_path)))))
    if normal_path:
        command.extend(("--normal-path", str(expand_worldfoundry_path(str(normal_path)))))
    if bool(settings.get("use_dynamic_cfg", False)):
        command.append("--use-dynamic-cfg")
    if bool(settings.get("memory_efficient", True)):
        command.append("--memory-efficient")
    return command


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_BASE_MODEL_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_INPUT_IMAGE",
    "DEFAULT_PROMPT",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
    "runtime_root",
]
