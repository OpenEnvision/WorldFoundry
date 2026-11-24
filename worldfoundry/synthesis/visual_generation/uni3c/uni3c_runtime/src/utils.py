import logging
import os
import random

import imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.spatial.transform import Rotation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("RANK", 0))


def get_local_rank():
    if torch.cuda.device_count() == 0:
        print("WARNING: No available GPU.")
        return 0
    return get_rank() % torch.cuda.device_count()


def is_distributed():
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process():  # real logger
        additional_args = {}
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_video(video_path):
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    frames = []
    for i in range(total_frames):
        frame = reader.get_data(i)
        frames.append(Image.fromarray(frame))

    reader.close()

    return frames


def rotation_matrix_from_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product)

    # special issue
    if np.linalg.norm(cross_product) < 1e-10:  # if is parallel
        if dot_product > 0:
            return np.eye(3)
        else:
            arbitrary_axis = np.array([1, 0, 0]) if np.all(v1 != np.array([1, 0, 0])) else np.array([0, 1, 0])
            return Rotation.from_rotvec(np.pi * arbitrary_axis).as_matrix()

    # Rodrigues formula to get rotation
    r = Rotation.from_rotvec(angle * cross_product / np.linalg.norm(cross_product))
    return r.as_matrix()
