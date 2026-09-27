from __future__ import annotations

import importlib.util
import os
from importlib.machinery import PathFinder
from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
DEFAULT_CONFIG = resolve_data_path("models", "runtime", "configs", "dino_wm", "conf", "plan_wall.yaml")
# Asset and dependency gating lives in missing_requirements so a fully staged
# official run can pass the public runtime-manifest preflight.
BLOCKED_REASON = ""


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    options = dict(options or {})
    missing = []
    config = Path(str(options.get("config") or options.get("config_path") or DEFAULT_CONFIG)).expanduser()
    ckpt_base_path = options.get("ckpt_base_path") or options.get("checkpoint_dir") or options.get("checkpoint_path")
    ckpt_base_dir = Path(str(ckpt_base_path)).expanduser() if ckpt_base_path else None
    model_name = options.get("model_name") or options.get("run_name")
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "DINO-WM infer.py is missing"})
    if not config.is_file():
        missing.append({"kind": "asset", "path": str(config), "reason": "DINO-WM planning config does not exist"})
    for module_name in ("gym", "hydra", "omegaconf", "einops", "torch"):
        if importlib.util.find_spec(module_name) is None:
            missing.append({"kind": "python_module", "path": module_name, "reason": "required DINO-WM runtime package is not importable"})
    # The official planner loads task trajectories before it constructs the
    # checkpoint-backed model.  A staged checkpoint alone is not runnable.
    dataset_package = importlib.util.find_spec("datasets")
    dataset_loader = (
        PathFinder.find_spec("datasets.wall_dset", dataset_package.submodule_search_locations)
        if dataset_package is not None and dataset_package.submodule_search_locations
        else None
    )
    if dataset_loader is None:
        missing.append({
            "kind": "python_module",
            "path": "datasets.wall_dset",
            "reason": "official DINO-WM task dataset loader is not importable",
        })
    dataset_root = os.environ.get("DATASET_DIR", "").strip()
    if not dataset_root:
        missing.append({
            "kind": "environment",
            "path": "DATASET_DIR",
            "reason": "official DINO-WM planning requires task trajectories under DATASET_DIR",
        })
    elif not Path(dataset_root).expanduser().is_dir():
        missing.append({
            "kind": "asset",
            "path": dataset_root,
            "reason": "DINO-WM DATASET_DIR does not exist",
        })
    if not ckpt_base_path:
        missing.append({"kind": "checkpoint", "path": "ckpt_base_path", "reason": "DINO-WM requires ckpt_base_path/checkpoint_dir"})
    elif not ckpt_base_dir.is_dir():
        missing.append({"kind": "checkpoint", "path": str(ckpt_base_path), "reason": "DINO-WM checkpoint base path is not a directory"})
    if not model_name:
        missing.append({"kind": "option", "path": "model_name", "reason": "DINO-WM requires model_name/run_name"})
    if ckpt_base_dir and model_name and ckpt_base_dir.is_dir():
        model_dir = Path(f"{ckpt_base_dir}/outputs/{model_name}")
        hydra_config = model_dir / "hydra.yaml"
        model_epoch = str(options.get("model_epoch") or "latest")
        checkpoint = model_dir / "checkpoints" / f"model_{model_epoch}.pth"
        if not hydra_config.is_file():
            missing.append({"kind": "asset", "path": str(hydra_config), "reason": "DINO-WM model hydra.yaml does not exist"})
        if not checkpoint.is_file():
            missing.append({"kind": "checkpoint", "path": str(checkpoint), "reason": "DINO-WM model checkpoint does not exist"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    ckpt_base_path = options.get("ckpt_base_path") or options.get("checkpoint_dir") or options.get("checkpoint_path") or ""
    model_name = options.get("model_name") or options.get("run_name") or ""
    command = [
        context["python"],
        context["entrypoint"],
        "--config",
        str(options.get("config") or options.get("config_path") or DEFAULT_CONFIG),
        "--ckpt-base-path",
        str(ckpt_base_path),
        "--model-name",
        str(model_name),
        "--model-epoch",
        str(options.get("model_epoch", "latest")),
        "--output-dir",
        context["output_dir"],
    ]
    for option_key, flag in (
        ("seed", "--seed"),
        ("n_evals", "--n-evals"),
        ("goal_source", "--goal-source"),
        ("goal_H", "--goal-h"),
    ):
        if option_key in options and options[option_key] not in (None, ""):
            command.extend([flag, str(options[option_key])])
    return command


__all__ = ["BLOCKED_REASON", "DEFAULT_CONFIG", "OFFICIAL_ENTRYPOINT", "RUNTIME_DIR", "build_command", "missing_requirements"]
