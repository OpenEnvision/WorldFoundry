from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch
from vbench import VBench
from worldfoundry.core.distributed.collectives.evaluation import dist_init, print0


def _str_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry in-tree VBench runtime")
    parser.add_argument("--output_path", default="./evaluation_results/")
    parser.add_argument("--full_json_dir", required=True)
    parser.add_argument("--videos_path", required=True)
    parser.add_argument("--dimension", nargs="+", required=True)
    parser.add_argument("--load_ckpt_from_local", type=_str_bool, default=False)
    parser.add_argument("--read_frame", type=_str_bool, default=False)
    parser.add_argument("--mode", choices=["custom_input", "vbench_standard", "vbench_category"], default="vbench_standard")
    parser.add_argument("--prompt", default="None")
    parser.add_argument("--prompt_file")
    parser.add_argument("--category")
    parser.add_argument("--imaging_quality_preprocessing_mode", default="longer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_init()
    print0(f"args: {args}")
    prompt: list[str] | dict[str, str] = []
    if args.prompt_file is not None and args.prompt != "None":
        raise ValueError("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "None") and args.mode != "custom_input":
        raise ValueError("must set --mode=custom_input for using external prompt")
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompt = json.load(handle)
        if not isinstance(prompt, dict):
            raise ValueError('invalid prompt file format; expected {"video_path": "prompt", ...}')
    elif args.prompt != "None":
        prompt = [args.prompt]

    kwargs = {"imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing_mode}
    if args.category:
        kwargs["category"] = args.category

    runtime = VBench(torch.device("cuda"), args.full_json_dir, args.output_path)
    runtime.evaluate(
        videos_path=args.videos_path,
        name=f"results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
        prompt_list=prompt,
        dimension_list=args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.mode,
        **kwargs,
    )
    print0("done")


if __name__ == "__main__":
    main()
