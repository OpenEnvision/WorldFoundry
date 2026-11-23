import numpy as np
import torch


def shuffle(tensor: torch.Tensor, size=None):
    """Gets randomly ordered subset of `tensor` of `size`"""
    if size is not None:
        size = min(size, len(tensor))

    idx = np.random.choice(len(tensor), size if size else len(tensor), replace=False)
    return tensor[idx]
