import os
import random
from datetime import datetime

import accelerate
import numpy as np
import torch

from ..data.unified_dataset import _derive_seed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from .cleanup import _set_rank_local_cuda_device
from .config import _dump_run_args_yaml, parse_pipeline_args
from .datasets import build_mosaic_inference_dataset
from .pipeline_module import build_mosaic_pipeline_module
from .runner import run_mosaic_inference_task


def main():
    args = parse_pipeline_args()

    base_output_path = args.output_path

    _pre_accelerate_device = _set_rank_local_cuda_device()
    accelerator = accelerate.Accelerator()
    _set_rank_local_cuda_device() or _pre_accelerate_device
    setattr(args, "rank", accelerator.process_index)

    # Create one shared log directory across ranks. Multi-node wrapper scripts
    # can pass --log_dir_name so every node uses the same run folder before any
    # Python-side timestamp/broadcast timing can diverge.
    log_dir_name = getattr(args, "log_dir_name", None)
    if log_dir_name:
        log_dir_name = str(log_dir_name).strip().strip("/\\")
        if not log_dir_name or os.path.basename(log_dir_name) != log_dir_name:
            raise ValueError(
                f"--log_dir_name must be a single directory name, got {getattr(args, 'log_dir_name', None)!r}."
            )
    else:
        log_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and accelerator.num_processes > 1:
            name_box = [log_dir_name if accelerator.is_main_process else None]
            torch.distributed.broadcast_object_list(name_box, src=0)
            log_dir_name = name_box[0]
    log_dir = os.path.join(base_output_path, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    args.output_path = log_dir
    if accelerator.is_main_process:
        print(f"Log directory: {log_dir}")
        args_yaml_path = _dump_run_args_yaml(args, os.path.join(log_dir, "args.yaml"))
        print(f"[config] dumped run args to {args_yaml_path}", flush=True)
    accelerator.wait_for_everyone()

    # Each rank receives a stable, distinct inference seed.
    _master_seed = _derive_seed(args.seed, "inference", accelerator.process_index)
    torch.manual_seed(_master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_master_seed)
    np.random.seed(_master_seed & 0xFFFFFFFF)
    random.seed(_master_seed)

    inference_dataset = build_mosaic_inference_dataset(args)

    model = build_mosaic_pipeline_module(args, device=accelerator.device)

    model.log_dir = log_dir
    run_mosaic_inference_task(
        accelerator,
        model,
        log_dir=log_dir,
        args=args,
        inference_dataset=inference_dataset,
    )


if __name__ == "__main__":
    main()
