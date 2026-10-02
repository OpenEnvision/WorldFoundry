# This file includes code originally from the PriorDA repository:
# https://github.com/SpatialVision/Prior-Depth-Anything
# Licensed under the Apache-2.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> three_dimensions -> depth -> priorda -> dav2 -> transform.py functionality."""

import numpy as np
import torch


class Resize(object):
    """Resize implementation."""
    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method="bilinear",
    ):
        """Init.

        Args:
            width: The width.
            height: The height.
            resize_target: The resize target.
            keep_aspect_ratio: The keep aspect ratio.
            ensure_multiple_of: The ensure multiple of.
            resize_method: The resize method.
            image_interpolation_method: The image interpolation method.
        """
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def get_size(self, width, height):
        """Get size.

        Args:
            width: The width.
            height: The height.
        """
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise NotImplementedError()

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        else:
            raise NotImplementedError()

        return (new_width, new_height)

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """Constrain to multiple of.

        Args:
            x: The x.
            min_val: The min val.
            max_val: The max val.
        """
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(np.int32)

        return y

    def __call__(self, sample):
        """Call.

        Args:
            sample: The sample.
        """
        width, height = self.get_size(sample["image"].shape[-1], sample["image"].shape[-2])
        sample["image"] = torch.nn.functional.interpolate(
            sample["image"], (height, width), mode=self.__image_interpolation_method
        )

        return sample


class NormalizeImage(object):
    """Normalize image implementation."""
    def __init__(self, mean, std, device="cpu"):
        """Init.

        Args:
            mean: The mean.
            std: The std.
            device: The device.
        """
        self.__mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.__std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def __call__(self, sample):
        """Call.

        Args:
            sample: The sample.
        """
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample
