from __future__ import annotations

from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "isolated_nwm_infer.py"
BLOCKED_REASON = (
    "NWM official source is vendored in-tree; execution still requires "
    "the official dependency environment, checkpoints, and task assets."
)


def _option(options: dict, *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    missing = []
    for key in ("exp", "datasets", "eval_type"):
        if not _option(options, key):
            missing.append({"kind": "option", "path": key, "reason": f"required NWM option `{key}` is not set"})
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "NWM isolated_nwm_infer.py is missing"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    return [
        context["python"],
        context["entrypoint"],
        "--output_dir",
        context["output_dir"],
        "--exp",
        _option(options, "exp", "config", "config_path", default="") or "",
        "--ckp",
        str(options.get("ckp", options.get("checkpoint_step", "0100000"))),
        "--datasets",
        _option(options, "datasets", "dataset", default="") or "",
        "--eval_type",
        _option(options, "eval_type", default="") or "",
        "--num_sec_eval",
        str(options.get("num_sec_eval", 5)),
        "--input_fps",
        str(options.get("input_fps", 4)),
        "--num_workers",
        str(options.get("num_workers", 8)),
        "--batch_size",
        str(options.get("batch_size", 16)),
        "--rollout_fps_values",
        str(options.get("rollout_fps_values", "1,4")),
        "--gt",
        str(options.get("gt", 0)),
    ]


__all__ = ["BLOCKED_REASON", "OFFICIAL_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
