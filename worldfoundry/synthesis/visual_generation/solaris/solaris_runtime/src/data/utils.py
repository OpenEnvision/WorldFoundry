import cv2
import jax
import torch

"""
Source: https://github.com/willisma/jax_measure_transport/blob/main/data/utils.py
"""


def torch_pytree_to_numpy(xs):
    def _prepare(x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return x

    return jax.tree.map(_prepare, xs)


def resize_letterbox(image, desired_height, desired_width):
    h, w = image.shape[:2]
    if h == desired_height and w == desired_width:
        return image

    resized = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    letterboxed = resized
    return letterboxed
