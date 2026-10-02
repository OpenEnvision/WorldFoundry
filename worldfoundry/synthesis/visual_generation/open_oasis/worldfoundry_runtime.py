from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import project_root
from worldfoundry.synthesis.visual_generation.runtime_manifest import (
    command_settings,
    plan_payload,
)


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "generate.py"
# The model-specific requirement hook below is the source of truth. Keeping a
# static blocker here would reject execution even after every asset is staged.
BLOCKED_REASON = ""


def _option(options: dict, *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _runtime_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root(__file__) / path
    return str(path.resolve())


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    checks = {
        "oasis_ckpt": _option(options, "oasis_ckpt", "checkpoint_path", "model_path"),
        "vae_ckpt": _option(options, "vae_ckpt", "vae_checkpoint_path"),
        "prompt_path": _option(options, "prompt_path", "input_path", "image_path", "video_path"),
        "actions_path": _option(options, "actions_path", "action_path"),
    }
    missing = []
    for key, value in checks.items():
        if not value:
            missing.append({"kind": "option", "path": key, "reason": f"required Open-Oasis option `{key}` is not set"})
        elif not Path(value).expanduser().exists():
            missing.append({"kind": "asset", "path": value, "reason": f"Open-Oasis `{key}` path does not exist"})
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "Open-Oasis generate.py is missing"})
    return missing


def build_command(context):
    options = command_settings(context)
    plan = plan_payload(context)
    if plan.get("fps") is not None:
        options["fps"] = plan["fps"]
    return [
        context["python"],
        context["entrypoint"],
        "--oasis-ckpt",
        _runtime_path(_option(options, "oasis_ckpt", "checkpoint_path", "model_path", default="") or ""),
        "--vae-ckpt",
        _runtime_path(_option(options, "vae_ckpt", "vae_checkpoint_path", default="") or ""),
        "--prompt-path",
        _runtime_path(_option(options, "prompt_path", "input_path", "image_path", "video_path", default="") or ""),
        "--actions-path",
        _runtime_path(_option(options, "actions_path", "action_path", default="") or ""),
        "--output-path",
        context["output_path"],
        "--num-frames",
        str(options.get("num_frames", 32)),
        "--fps",
        str(options.get("fps", 20)),
        "--ddim-steps",
        str(options.get("ddim_steps", 10)),
    ]


__all__ = ["BLOCKED_REASON", "OFFICIAL_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
