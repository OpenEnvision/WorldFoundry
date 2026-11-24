from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import resolve_data_path, resolve_worldfoundry_path

RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "le_wm", "config", "eval")
# The requirements hook below is authoritative.  Keeping a non-empty static
# reason would make the shared runtime-manifest facade report this route as
# blocked even after its environment and dataset have been staged.
BLOCKED_REASON = ""


def _dataset_candidates(dataset_name: str, cache_dir: Path) -> list[Path]:
    raw = resolve_worldfoundry_path(dataset_name)
    if raw.suffix in {".h5", ".hdf5"} or raw.is_absolute():
        return [raw]
    return [
        cache_dir / f"{dataset_name}.h5",
        cache_dir / "datasets" / f"{dataset_name}.h5",
        cache_dir / dataset_name,
    ]


def policy_checkpoint_path(options: Mapping[str, Any]) -> Path | None:
    """Resolve the learned-policy object checkpoint; random policies have none."""
    policy = str(options.get("policy") or "random")
    if policy == "random":
        return None
    policy_path = resolve_worldfoundry_path(policy)
    if policy_path.is_absolute():
        return policy_path
    cache_dir = resolve_worldfoundry_path(
        str(options.get("cache_dir") or os.environ.get("STABLEWM_HOME") or "~/.stable-wm")
    )
    return cache_dir / f"{policy}_object.ckpt"


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    options = dict(options or {})
    missing = []
    config_dir = resolve_worldfoundry_path(str(options.get("config_dir") or CONFIG_DIR))
    config_name = str(options.get("config_name") or options.get("task") or "pusht")
    config_file = config_dir / (config_name if config_name.endswith(".yaml") else f"{config_name}.yaml")
    cache_dir = resolve_worldfoundry_path(
        str(options.get("cache_dir") or os.environ.get("STABLEWM_HOME") or "~/.stable-wm")
    )

    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append(
            {"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "LeWorldModel infer.py is missing"}
        )
    if not config_file.is_file():
        missing.append({"kind": "asset", "path": str(config_file), "reason": "LeWorldModel eval config does not exist"})
    for module_name in ("stable_worldmodel", "stable_pretraining"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": "required LeWorldModel runtime package is not importable",
                }
            )

    if config_file.is_file():
        try:
            from omegaconf import OmegaConf
        except Exception:
            OmegaConf = None
            missing.append(
                {
                    "kind": "python_module",
                    "path": "omegaconf",
                    "reason": "required LeWorldModel config parser is not importable",
                }
            )
        dataset_name = str(options.get("dataset_name") or "")
        if OmegaConf is not None and not dataset_name:
            cfg = OmegaConf.load(config_file)
            dataset_name = str(cfg.get("eval", {}).get("dataset_name") or "")
        if dataset_name:
            candidates = _dataset_candidates(dataset_name, cache_dir)
            if not any(path.is_file() for path in candidates):
                missing.append(
                    {
                        "kind": "asset",
                        "path": " | ".join(str(path) for path in candidates),
                        "reason": "LeWorldModel eval dataset is missing",
                    }
                )

    checkpoint_path = policy_checkpoint_path(options)
    if checkpoint_path is not None and not checkpoint_path.is_file():
        missing.append(
            {
                "kind": "checkpoint",
                "path": str(checkpoint_path),
                "reason": "LeWorldModel policy checkpoint is missing",
            }
        )
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    output_path = Path(str(context["output_path"]))
    artifact_path = output_path
    if output_path.suffix.lower() in {".mp4", ".mov", ".webm", ".gif"}:
        artifact_path = output_path.with_suffix(".result.json")
    command = [
        "--config-dir",
        str(resolve_worldfoundry_path(str(options.get("config_dir") or CONFIG_DIR))),
        "--config-name",
        str(options.get("config_name") or options.get("task") or "pusht"),
        "--policy",
        str(options.get("policy") or "random"),
        "--output-dir",
        context["output_dir"],
        "--artifact-path",
        str(artifact_path),
        "--device",
        str(options.get("device") or context.get("device") or "cuda"),
    ]
    for option_key, flag in (
        ("cache_dir", "--cache-dir"),
        ("seed", "--seed"),
        ("dataset_name", "--dataset-name"),
        ("num_eval", "--num-eval"),
        ("eval_budget", "--eval-budget"),
        ("goal_offset_steps", "--goal-offset-steps"),
        ("img_size", "--img-size"),
    ):
        if option_key in options and options[option_key] not in (None, ""):
            value = options[option_key]
            if option_key in {"cache_dir", "dataset_name"}:
                value = resolve_worldfoundry_path(str(value))
            command.extend([flag, str(value)])
    return package_module_command(
        "worldfoundry.synthesis.visual_generation.le_wm.infer", command, python_executable=context["python"]
    )


__all__ = [
    "BLOCKED_REASON",
    "CONFIG_DIR",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
    "policy_checkpoint_path",
]
