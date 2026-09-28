from __future__ import annotations

import importlib.util
import os
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
    config = Path(str(options.get("config") or options.get("config_path") or DEFAULT_CONFIG)).expanduser().resolve()
    ckpt_base_path = options.get("ckpt_base_path") or options.get("checkpoint_dir") or options.get("checkpoint_path")
    ckpt_base_dir = Path(str(ckpt_base_path)).expanduser().resolve() if ckpt_base_path else None
    model_name = options.get("model_name") or options.get("run_name")
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "DINO-WM infer.py is missing"})
    if not config.is_file():
        missing.append({"kind": "asset", "path": str(config), "reason": "DINO-WM planning config does not exist"})
    for module_name in ("gym", "hydra", "omegaconf", "einops", "torch", "decord", "torchvision"):
        if importlib.util.find_spec(module_name) is None:
            missing.append({"kind": "python_module", "path": module_name, "reason": "required DINO-WM runtime package is not importable"})
    # The official planner loads task trajectories before it constructs the
    # checkpoint-backed model.  A staged checkpoint alone is not runnable.
    dataset_modules = {
        "wall_single": "wall_dset.py",
        "point_maze": "point_maze_dset.py",
        "pusht": "pusht_dset.py",
        "deformable_env": "deformable_env_dset.py",
    }
    dataset_module = dataset_modules.get(str(model_name), "wall_dset.py")
    if not (RUNTIME_DIR / "datasets" / dataset_module).is_file():
        missing.append({
            "kind": "python_module",
            "path": f"worldfoundry.synthesis.visual_generation.world_model.dino_wm.datasets.{dataset_module[:-3]}",
            "reason": "vendored DINO-WM task dataset loader is missing",
        })
    dataset_root = os.environ.get("DATASET_DIR", "").strip()
    if not dataset_root:
        missing.append({
            "kind": "environment",
            "path": "DATASET_DIR",
            "reason": "official DINO-WM planning requires task trajectories under DATASET_DIR",
        })
    elif not Path(dataset_root).expanduser().resolve().is_dir():
        missing.append({
            "kind": "asset",
            "path": dataset_root,
            "reason": "DINO-WM DATASET_DIR does not exist",
        })
    else:
        dataset_dir = Path(dataset_root).expanduser().resolve()
        required_files = {
            "wall_single": ("states.pth", "actions.pth", "door_locations.pth", "wall_locations.pth", "obses/episode_000.pth"),
            "point_maze": ("states.pth", "actions.pth", "seq_lengths.pth", "obses/episode_000.pth"),
            "pusht": (
                "train/states.pth", "train/rel_actions.pth", "train/seq_lengths.pkl", "train/velocities.pth", "train/obses/episode_000.mp4",
                "val/states.pth", "val/rel_actions.pth", "val/seq_lengths.pkl", "val/velocities.pth", "val/obses/episode_000.mp4",
            ),
        }.get(str(model_name))
        if required_files:
            task_dir = dataset_dir / ("pusht_noise" if str(model_name) == "pusht" else str(model_name))
            for relative_path in required_files:
                asset = task_dir / relative_path
                if not asset.is_file():
                    missing.append({"kind": "asset", "path": str(asset), "reason": "DINO-WM task trajectory asset is missing"})
            if str(model_name) == "wall_single" and (task_dir / "states.pth").is_file():
                try:
                    import torch

                    rollout_count = len(torch.load(task_dir / "states.pth", map_location="cpu", weights_only=True))
                    available_episodes = {path.name for path in (task_dir / "obses").glob("episode_*.pth")}
                    available_count = sum(
                        f"episode_{index:03d}.pth" in available_episodes for index in range(rollout_count)
                    )
                    if available_count != rollout_count:
                        missing.append({
                            "kind": "asset",
                            "path": str(task_dir / "obses"),
                            "reason": f"DINO-WM wall trajectories are incomplete ({available_count}/{rollout_count} episodes)",
                        })
                except Exception as exc:
                    missing.append({
                        "kind": "asset",
                        "path": str(task_dir / "states.pth"),
                        "reason": f"DINO-WM wall trajectory index cannot be read safely: {exc}",
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
    config = Path(str(options.get("config") or options.get("config_path") or DEFAULT_CONFIG)).expanduser().resolve()
    command = [
        context["python"],
        context["entrypoint"],
        "--config",
        str(config),
        "--ckpt-base-path",
        str(Path(str(ckpt_base_path)).expanduser().resolve()) if ckpt_base_path else "",
        "--model-name",
        str(model_name),
        "--model-epoch",
        str(options.get("model_epoch", "latest")),
        "--output-dir",
        context["output_dir"],
    ]
    dataset_root = os.environ.get("DATASET_DIR", "").strip()
    if dataset_root:
        command.extend(["--dataset-dir", str(Path(dataset_root).expanduser().resolve())])
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
