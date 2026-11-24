from __future__ import annotations

import argparse
from datetime import datetime

import torch
from vbench2_beta_i2v import VBenchI2V


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry in-tree VBench++ I2V runtime")
    parser.add_argument("--output_path", default="./evaluation_i2v_results/")
    parser.add_argument("--full_json_dir", required=True)
    parser.add_argument("--videos_path", required=True)
    parser.add_argument("--dimension", nargs="+", required=True)
    parser.add_argument("--ratio")
    parser.add_argument("--custom_image_folder")
    parser.add_argument("--mode", choices=["custom_input", "vbench_standard"], default="vbench_standard")
    parser.add_argument("--imaging_quality_preprocessing_mode", default="longer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = VBenchI2V(torch.device("cuda"), args.full_json_dir, args.output_path)
    runtime.evaluate(
        videos_path=args.videos_path,
        name=f"results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
        dimension_list=args.dimension,
        resolution=args.ratio,
        custom_image_folder=args.custom_image_folder,
        mode=args.mode,
        imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
    )
    print("done")


if __name__ == "__main__":
    main()
