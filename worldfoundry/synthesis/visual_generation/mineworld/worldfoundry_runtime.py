from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path


RUNTIME_DIR = Path(__file__).resolve().parent
INFERENCE_ENTRYPOINT = RUNTIME_DIR / "inference.py"
BLOCKED_REASON = (
    "MineWorld official source is vendored in-tree; execution still requires "
    "the official dependency environment, checkpoints, and task assets."
)


def _option(options: dict, *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _default_config(frames: int) -> str:
    filename = "300M_16f.yaml" if int(frames) <= 16 else "700M_32f.yaml"
    return str(resolve_data_path("models", "runtime", "configs", "mineworld", filename))


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    frames = int(options.get("frames", 16))
    checks = {
        "data_root": _option(options, "data_root", "input_dir", "input_path"),
        "model_ckpt": _option(options, "model_ckpt", "checkpoint_path", "model_path"),
        "config": _option(options, "config", "config_path", default=_default_config(frames)),
    }
    missing = []
    for key, value in checks.items():
        if not value:
            missing.append({"kind": "option", "path": key, "reason": f"required MineWorld option `{key}` is not set"})
        elif not Path(value).expanduser().exists():
            missing.append({"kind": "asset", "path": value, "reason": f"MineWorld `{key}` path does not exist"})
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "MineWorld inference.py is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    frames = int(options.get("frames", 16))
    command = [
        context["python"],
        context["entrypoint"],
        "--data_root",
        _option(options, "data_root", "input_dir", "input_path", default="") or "",
        "--model_ckpt",
        _option(options, "model_ckpt", "checkpoint_path", "model_path", default="") or "",
        "--config",
        _option(options, "config", "config_path", default=_default_config(frames)) or "",
        "--output_dir",
        context["output_dir"],
        "--demo_num",
        str(options.get("demo_num", 1)),
        "--frames",
        str(frames),
        "--window_size",
        str(options.get("window_size", 2)),
        "--accelerate-algo",
        str(options.get("accelerate_algo", options.get("accelerate-algo", "naive"))),
        "--fps",
        str(options.get("fps", 6)),
        "--val_data_num",
        str(options.get("val_data_num", 1)),
    ]
    if options.get("top_p") not in (None, ""):
        command.extend(["--top_p", str(options["top_p"])])
    else:
        command.extend(["--top_k", str(options.get("top_k", 100))])
    return command


__all__ = ["BLOCKED_REASON", "INFERENCE_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
