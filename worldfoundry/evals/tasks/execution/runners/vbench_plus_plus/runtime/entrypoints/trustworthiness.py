from __future__ import annotations

import argparse
from datetime import datetime

import torch

from worldfoundry.evals.tasks.execution.runners.vbench_plus_plus.runtime.vbench2_beta_trustworthiness import (
    VBenchTrustworthiness,
)


def _str_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry in-tree VBench++ trustworthiness runtime")
    parser.add_argument("--output_path", default="./evaluation_trustworthy_results/")
    parser.add_argument("--full_json_dir", required=True)
    parser.add_argument("--videos_path", required=True)
    parser.add_argument("--dimension", nargs="+", required=True)
    parser.add_argument("--load_ckpt_from_local", type=_str_bool, default=False)
    parser.add_argument("--read_frame", type=_str_bool, default=False)
    parser.add_argument("--custom_input", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = VBenchTrustworthiness(torch.device("cuda"), args.full_json_dir, args.output_path)
    runtime.evaluate(
        videos_path=args.videos_path,
        name=f"results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
        dimension_list=args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        custom_prompt=args.custom_input,
    )
    print("done")


if __name__ == "__main__":
    main()
