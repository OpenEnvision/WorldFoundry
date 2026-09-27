from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "worldmodel" / "sample.py"
BLOCKED_REASON = ""


def _option(options: dict, *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _path_option(options: dict, *names: str, default: str | None = None) -> str | None:
    value = _option(options, *names, default=default)
    return str(Path(value).expanduser().resolve()) if value else None


def _default_config() -> str:
    return str(resolve_data_path("models", "runtime", "configs", "adaworld", "worldmodel/inference/adaworld.yaml"))


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    checkpoint = _path_option(options, "checkpoint", "checkpoint_path", "ckpt", "ckpt_path", "model_path")
    config = _path_option(options, "config", "config_path", default=_default_config())
    data_root = _path_option(options, "data_root", "input_dir")
    source_video = _path_option(options, "source_video", "input_path", "video_path")
    missing = []
    for key, value in {"checkpoint": checkpoint, "config": config}.items():
        if not value:
            missing.append({"kind": "option", "path": key, "reason": f"required AdaWorld option `{key}` is not set"})
        elif not Path(value).expanduser().exists():
            missing.append({"kind": "asset", "path": value, "reason": f"AdaWorld `{key}` path does not exist"})
    if not data_root and not source_video:
        missing.append({"kind": "option", "path": "data_root", "reason": "AdaWorld requires data_root or source_video"})
    elif data_root and not Path(data_root).expanduser().exists():
        missing.append({"kind": "asset", "path": data_root, "reason": "AdaWorld data_root does not exist"})
    elif source_video and not Path(source_video).expanduser().is_file():
        missing.append({"kind": "asset", "path": source_video, "reason": "AdaWorld source_video does not exist"})
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "AdaWorld worldmodel/sample.py is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    command = [
        context["python"],
        context["entrypoint"],
        "--checkpoint",
        _path_option(options, "checkpoint", "checkpoint_path", "ckpt", "ckpt_path", "model_path") or "",
        "--config",
        _path_option(options, "config", "config_path", default=_default_config()) or "",
        "--output-path",
        context["output_path"],
        "--num-samples",
        str(options.get("num_samples", 1)),
        "--start-index",
        str(options.get("start_index", 50)),
        "--resolution",
        str(options.get("resolution", 256)),
        "--video-len",
        str(options.get("video_len", 20)),
        "--context-frame",
        str(options.get("context_frame", 6)),
        "--num-steps",
        str(options.get("num_steps", 5)),
        "--cfg-scale",
        str(options.get("cfg_scale", 1.1)),
        "--aug-level",
        str(options.get("aug_level", 0.1)),
        "--fps",
        str(options.get("fps", 5)),
        "--seed",
        str(options.get("seed", 32)),
    ]
    data_root = _path_option(options, "data_root", "input_dir")
    source_video = _path_option(options, "source_video", "input_path", "video_path")
    target_video = _path_option(options, "target_video")
    if data_root:
        command.extend(["--data-root", data_root])
    if source_video:
        command.extend(["--source-video", source_video])
    if target_video:
        command.extend(["--target-video", target_video])
    return command


__all__ = ["BLOCKED_REASON", "OFFICIAL_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
