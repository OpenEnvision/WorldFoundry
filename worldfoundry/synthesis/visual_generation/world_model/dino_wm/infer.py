from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict

from official_plan import planning_main
from utils import cfg_to_dict
from worldfoundry.core.io.paths import resolve_data_path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"DINO-WM result contains an unsupported value: {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry DINO-WM inference/planning entrypoint.")
    parser.add_argument(
        "--config",
        default=str(resolve_data_path("models", "runtime", "configs", "dino_wm", "conf", "plan_wall.yaml")),
        help="DINO-WM planning config YAML.",
    )
    parser.add_argument("--ckpt-base-path", required=True, help="Directory containing outputs/<model_name>/...")
    parser.add_argument("--model-name", required=True, help="Official DINO-WM checkpoint run name under ckpt-base-path/outputs.")
    parser.add_argument("--model-epoch", default="latest", help="Checkpoint epoch suffix, e.g. latest or final.")
    parser.add_argument("--output-dir", required=True, help="Directory for planning logs/artifacts.")
    parser.add_argument("--dataset-dir", default=None, help="Resolved DINO-WM task trajectory root.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-evals", type=int, default=None)
    parser.add_argument("--goal-source", default=None)
    parser.add_argument("--goal-h", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir or os.environ.get("DATASET_DIR")
    if dataset_dir:
        os.environ["DATASET_DIR"] = str(Path(dataset_dir).expanduser().resolve())
    cfg = OmegaConf.load(Path(args.config).expanduser().resolve())
    with open_dict(cfg):
        cfg["saved_folder"] = str(Path(args.output_dir).expanduser().resolve())
        cfg["ckpt_base_path"] = str(Path(args.ckpt_base_path).expanduser().resolve())
        cfg["model_name"] = args.model_name
        cfg["model_epoch"] = args.model_epoch
        cfg["wandb_logging"] = False
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.n_evals is not None:
            cfg["n_evals"] = args.n_evals
        if args.goal_source is not None:
            cfg["goal_source"] = args.goal_source
        if args.goal_h is not None:
            cfg["goal_H"] = args.goal_h

    output_dir = Path(cfg["saved_folder"])
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    logs = planning_main(cfg_to_dict(cfg))
    (output_dir / "worldfoundry_dino_wm_result.json").write_text(
        json.dumps({"status": "succeeded", "logs": logs}, default=_json_default, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
