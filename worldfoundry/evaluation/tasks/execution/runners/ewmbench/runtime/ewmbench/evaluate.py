import argparse
import os

import torch
import yaml

from EWMBench import EmbodiedWorldModelBenchmark
from worldfoundry.core.distributed.collectives.evaluation import dist_init, print0


def parse_args():
    parser = argparse.ArgumentParser(
        description="EmbodiedWorldModelBenchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dimension",
        nargs="+",
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to config JSON that defines per-dimension settings",
    )

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    dist_init()
    print0(f"args: {args}")
    device = torch.device("cuda")

    config = load_config(args.config_path)

    save_path = config["save_path"]
    data_base = config["data"]["val_base"]
    gt_path = config["data"]["gt_path"]
    data_name = os.path.basename(data_base).replace("_dataset", "")

    kwargs = {}

    for dim in args.dimension:
        if dim == "semantics":
            kwargs[f"{dim}_caption_model_ckpt"] = config.get("ckpt", {}).get(dim, {}).get("caption", None)
            kwargs[f"{dim}_clip_model_ckpt"] = config.get("ckpt", {}).get(dim, {}).get("CLIP", None)
            kwargs[f"{dim}_bleus_model_ckpt"] = config.get(dim, {}).get("BLEUs", None)

            key_caption = f"{dim}_caption_model_ckpt"
            key_clip = f"{dim}_clip_model_ckpt"
            key_bleus = f"{dim}_bleus_model_ckpt"

            print(f"{dim}: caption = {kwargs[key_caption]}, CLIP = {kwargs[key_clip]}, BLEUs = {kwargs[key_bleus]}")
        elif dim in ["psnr", "ssim"]:
            pass
        else:
            kwargs[f"{dim}_model_ckpt"] = config.get("ckpt", {}).get(dim, None)

            key_model_ckpt = f"{dim}_model_ckpt"
            print(f"{dim}: ckpt = {kwargs[key_model_ckpt]}")

    print0("start evaluation")
    my_benchmark = EmbodiedWorldModelBenchmark(device, save_path)

    my_benchmark.evaluate(
        data_base=data_base,
        data_name=data_name,
        dimension_list=args.dimension,
        local=True,
        gt_path=gt_path,
        overwrite=args.overwrite,
        **kwargs,
    )
    print0("done")


if __name__ == "__main__":
    main()
