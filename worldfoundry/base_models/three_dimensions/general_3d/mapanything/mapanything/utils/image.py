# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.
"""Image conversion needed by MapAnything inference post-processing."""

import numpy as np
import torch

from worldfoundry.base_models.perception_core.general_perception.uniception.uniception.models.encoders.image_normalizations import (
    IMAGE_NORMALIZATION_DICT,
)


def rgb(ftensor, norm_type, true_shape=None):
    """Convert a normalized CHW/BCHW tensor to an RGB array in ``[0, 1]``."""
    if isinstance(ftensor, list):
        return [rgb(value, norm_type, true_shape=true_shape) for value in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        height, width = true_shape
        ftensor = ftensor[:height, :width]
    if ftensor.dtype == np.uint8:
        image = np.float32(ftensor) / 255
    elif norm_type in IMAGE_NORMALIZATION_DICT:
        normalization = IMAGE_NORMALIZATION_DICT[norm_type]
        image = ftensor * normalization.std.numpy() + normalization.mean.numpy()
    elif norm_type == "identity":
        image = ftensor
    else:
        available = ", ".join(sorted(IMAGE_NORMALIZATION_DICT))
        raise ValueError(f"Unknown image normalization {norm_type!r}; expected identity or one of: {available}")
    return image.clip(min=0, max=1)
