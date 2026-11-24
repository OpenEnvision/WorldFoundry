import atexit
import logging
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
from einops import rearrange

from worldfoundry.core.media.codecs.video import write_video_torchvision


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def init_logging(log_dir, rank):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_rank{rank}_{now}.txt")

    log_file = open(log_filename, "w", buffering=1)  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
    atexit.register(log_file.close)

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(log_file)],
    )

    print(f"[INFO] Logging initialized at: {log_filename}")
    return log_file


def save_video(tensor, save_path, fps=30):
    """
    Input tensor: shape c,t,h,w ranging from -1 to 1
    """
    tensor = ((tensor + 1) / 2 * 255).to(torch.uint8)
    tensor = rearrange(tensor, "c t h w -> t h w c")
    write_video_torchvision(save_path, tensor, fps=fps)


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)
