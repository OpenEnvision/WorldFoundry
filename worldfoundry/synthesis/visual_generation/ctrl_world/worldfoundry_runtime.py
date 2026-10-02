"""Bind WorldFoundry to the vendored Ctrl-World checkpoint runtime."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import checkpoint_root_path, project_root
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings


PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PACKAGE_DIR / "ctrl_world_runtime"
OFFICIAL_ENTRYPOINT = PACKAGE_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/Robert-gyj/Ctrl-World"

DEFAULT_CHECKPOINT_PATH = checkpoint_root_path("yjguo--Ctrl-World", "checkpoint-10000.pt")
DEFAULT_BASE_MODEL_DIR = checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")
DEFAULT_CLIP_MODEL_DIR = checkpoint_root_path("openai--clip-vit-base-patch32")
DEFAULT_INPUT_IMAGE = (
    project_root(__file__)
    / "worldfoundry"
    / "data"
    / "test_cases"
    / "test_vla_case1"
    / "droid"
    / "exterior_image_1_left.png"
)
DEFAULT_INPUT_VIEWS = tuple(
    project_root(__file__) / "worldfoundry" / "data" / "test_cases" / "test_vla_case1" / "aloha" / name
    for name in (
        "observation_images_cam_high.png",
        "observation_images_cam_left_wrist.png",
        "observation_images_cam_right_wrist.png",
    )
)
DEFAULT_PROMPT = "Move the robot gripper toward the objects on the workbench."
ACTION_STATS_PATH = PACKAGE_DIR / "action_stats.json"
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


def _checkpoint_path(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "ctrl_world_checkpoint_path",
        "checkpoint_path",
        "checkpoint",
        "ckpt_path",
        "model_path",
        default=DEFAULT_CHECKPOINT_PATH,
    )


def _base_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "svd_model_path",
        "base_model_dir",
        default=DEFAULT_BASE_MODEL_DIR,
    )


def _clip_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(options, "clip_model_path", "clip_model_dir", default=DEFAULT_CLIP_MODEL_DIR)


def _input_views(options: Mapping[str, Any]) -> tuple[Path, ...]:
    value = _option(options, "input_views", "view_images", default=DEFAULT_INPUT_VIEWS)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(expand_worldfoundry_path(str(path)) for path in value)


def _initial_pose(options: Mapping[str, Any]) -> tuple[float, ...] | None:
    value = options.get("initial_pose")
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    try:
        pose = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None
    return pose if len(pose) == 7 and all(math.isfinite(part) for part in pose) else None


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

    for relative in (
        "models/ctrl_world.py",
        "models/pipeline_ctrl_world.py",
        "models/unet_spatio_temporal_condition.py",
    ):
        path = source / relative
        if not path.is_file():
            missing.append(
                {"kind": "source_repo", "path": str(path), "reason": "vendored Ctrl-World runtime file is missing"}
            )
    if not ACTION_STATS_PATH.is_file():
        missing.append({"kind": "source_repo", "path": str(ACTION_STATS_PATH), "reason": "official DROID action statistics are missing"})
    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "WorldFoundry Ctrl-World compatibility launcher is missing",
            }
        )

    checkpoint_missing = _missing_file(_checkpoint_path(options), "Ctrl-World checkpoint is missing or empty")
    if checkpoint_missing is not None:
        missing.append(checkpoint_missing)

    base = _base_model_dir(options)
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "feature_extractor/preprocessor_config.json",
        "image_encoder/config.json",
        "image_encoder/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ):
        item = _missing_file(base / relative, "Stable Video Diffusion component is missing or empty")
        if item is not None:
            missing.append(item)

    clip = _clip_model_dir(options)
    for relative in ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        item = _missing_file(clip / relative, "CLIP text component is missing or empty")
        if item is not None:
            missing.append(item)

    input_mode = str(options.get("input_mode", "explicit-three-view"))
    if input_mode == "explicit-three-view":
        views = _input_views(options)
        if len(views) != 3:
            missing.append(
                {
                    "kind": "asset",
                    "path": "input_views",
                    "reason": "Ctrl-World explicit-three-view mode requires exactly three image paths",
                }
            )
        else:
            for index, view in enumerate(views):
                if not view.is_file():
                    missing.append(
                        {
                            "kind": "asset",
                            "path": str(view),
                            "reason": f"Ctrl-World input view {index} is missing",
                        }
                    )
    else:
        image = _path_option(options, "image_path", "input_image", default=DEFAULT_INPUT_IMAGE)
        if not image.is_file():
            missing.append({"kind": "asset", "path": str(image), "reason": "Ctrl-World input image is missing"})

    action_mode = str(options.get("action_mode", "absolute-pose"))
    if action_mode == "absolute-pose":
        if _initial_pose(options) is None:
            missing.append({"kind": "input", "path": "initial_pose", "reason": "absolute-pose mode requires seven finite physical Cartesian pose values"})
        if options.get("action_scale") is not None:
            missing.append({"kind": "input", "path": "action_scale", "reason": "normalized action_scale is only valid for synthetic-zero-smoke mode"})
    elif action_mode == "synthetic-zero-smoke":
        if options.get("initial_pose") is not None:
            missing.append({"kind": "input", "path": "initial_pose", "reason": "physical initial_pose is only valid for absolute-pose mode"})
    else:
        missing.append({"kind": "input", "path": "action_mode", "reason": f"unsupported Ctrl-World action mode: {action_mode}"})

    for module_name in ("torch", "diffusers", "transformers", "PIL", "accelerate", "einops", "numpy"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required Ctrl-World dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    settings = command_settings(context)
    input_mode = str(settings.get("input_mode", "explicit-three-view"))
    action_mode = str(settings.get("action_mode", "absolute-pose"))
    command = [
        str(context["python"]),
        str(context["entrypoint"]),
        "--checkpoint-path",
        str(_checkpoint_path(settings)),
        "--base-model-dir",
        str(_base_model_dir(settings)),
        "--clip-model-dir",
        str(_clip_model_dir(settings)),
        "--output-path",
        str(context["output_path"]),
        "--prompt",
        str(context.get("prompt") or DEFAULT_PROMPT),
        "--device",
        str(context.get("device") or "cuda"),
        "--input-mode",
        input_mode,
        "--action-direction",
        str(settings.get("action_direction", "right")),
        "--action-mode",
        action_mode,
        "--height",
        str(settings.get("height", 192)),
        "--width",
        str(settings.get("width", 320)),
        "--num-frames",
        str(settings.get("num_frames", 5)),
        "--num-inference-steps",
        str(settings.get("num_inference_steps", 25)),
        "--guidance-scale",
        str(settings.get("guidance_scale", 1.0)),
        "--motion-bucket-id",
        str(settings.get("motion_bucket_id", 127)),
        "--fps",
        str(settings.get("fps", 4)),
        "--seed",
        str(settings.get("seed", 42)),
        "--mixed-precision",
        str(settings.get("mixed_precision", "bf16")),
    ]
    if input_mode == "explicit-three-view":
        views = _input_views(settings)
        if len(views) != 3:
            raise ValueError("Ctrl-World explicit-three-view mode requires exactly three input_views")
        command[6:6] = ["--input-views", *(str(view) for view in views)]
    else:
        image = _path_option(settings, "image_path", "input_image", default=DEFAULT_INPUT_IMAGE)
        command[6:6] = ["--input-image", str(image)]
    if action_mode == "absolute-pose":
        pose = _initial_pose(settings)
        if pose is None:
            raise ValueError("Ctrl-World absolute-pose mode requires seven finite initial_pose values")
        if settings.get("action_scale") is not None:
            raise ValueError("Ctrl-World normalized action_scale is only valid for synthetic-zero-smoke mode")
        command.extend(["--initial-pose", *(str(value) for value in pose), "--action-distance", str(settings.get("action_distance", 0.08))])
    elif action_mode == "synthetic-zero-smoke":
        if settings.get("initial_pose") is not None:
            raise ValueError("Ctrl-World physical initial_pose is only valid for absolute-pose mode")
        command.extend(["--action-scale", str(settings.get("action_scale", 0.2))])
    else:
        raise ValueError(f"unsupported Ctrl-World action mode: {action_mode}")
    return command


__all__ = [
    "BLOCKED_REASON",
    "ACTION_STATS_PATH",
    "DEFAULT_BASE_MODEL_DIR",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CLIP_MODEL_DIR",
    "DEFAULT_INPUT_IMAGE",
    "DEFAULT_INPUT_VIEWS",
    "DEFAULT_PROMPT",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
    "runtime_root",
]
