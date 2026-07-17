import argparse
import logging
import multiprocessing as mp
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from .classification import main as evaluate


def _init_distributed(rank: int, world_size: int) -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--lam", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--loader_timeout", type=int, default=0)
    return parser


def _run_worker(args: argparse.Namespace, rank: int, world_size: int) -> None:
    selected_device = args.devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_device == "cpu" else selected_device.split(":", 1)[-1]
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR)

    with open(args.fname, encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    params["folder"] = os.path.join(os.environ["LARY_LOG_DIR"], "classification", args.dataset, args.lam)
    params["dataset"] = args.dataset
    params["lam"] = args.lam
    params["num_workers"] = args.num_workers
    params["loader_timeout"] = args.loader_timeout
    params["experiment"]["classifier"]["dim"] = args.dim
    params["experiment"]["data"]["num_classes"] = args.classes
    params["experiment"]["optimization"]["batch_size"] = args.batch_size

    metadata_root = os.environ["LARY_METADATA_DIR"]
    params["experiment"]["data"]["dataset_train"] = os.path.join(
        metadata_root, f"train_la_{args.dataset}_{args.lam}.csv"
    )
    params["experiment"]["data"]["dataset_val"] = os.path.join(metadata_root, f"val_la_{args.dataset}_{args.lam}.csv")

    _init_distributed(rank, world_size)
    evaluate(params)


def main() -> None:
    args = _parser().parse_args()
    world_size = len(args.devices)
    mp.set_start_method("spawn")
    processes = [mp.Process(target=_run_worker, args=(args, rank, world_size)) for rank in range(world_size)]
    for process in processes:
        process.start()

    exit_code = 0
    while processes:
        alive = []
        for process in processes:
            process.join(timeout=0)
            if process.exitcode is None:
                alive.append(process)
            elif process.exitcode and exit_code == 0:
                exit_code = process.exitcode
        if exit_code:
            for process in alive:
                process.terminate()
            for process in alive:
                process.join(timeout=10)
                if process.exitcode is None:
                    process.kill()
            break
        processes = alive
        if processes:
            time.sleep(1)

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
