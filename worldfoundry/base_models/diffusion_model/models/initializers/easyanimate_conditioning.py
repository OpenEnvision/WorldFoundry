# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""EasyAnimate and VideoX-Fun image-to-video conditioning preparation.

Adapted from https://github.com/aigc-apps/EasyAnimate.
Pixels are in [0, 1]; the mask retains the reference branch's uint8/float dtype.
"""

import gc
import os
from typing import cast

import numpy as np
import torch
from PIL import Image


def get_image_to_video_latent(
    validation_image_start: str | list[Image.Image] | None,
    validation_image_end: str | list[Image.Image] | None,
    video_length: int,
    sample_size: tuple[int, int] | list[int],
) -> tuple[torch.Tensor, torch.Tensor, Image.Image | None]:
    """Prepare BCTHW conditioning pixels, 0/255 inpaint masks and a CLIP image."""
    if validation_image_start is not None and validation_image_end is not None:
        if isinstance(validation_image_start, str) and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = cast(list[Image.Image], validation_image_start)
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if isinstance(validation_image_end, str) and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = cast(list[Image.Image], validation_image_end)
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if isinstance(image_start, list):
            clip_image = cast(list[Image.Image], clip_image)[0]
            start_video = torch.cat(
                [
                    torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
                    for _image_start in image_start
                ],
                dim=2,
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, : len(image_start)] = start_video

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                [1, 1, video_length, 1, 1],
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if isinstance(image_end, list):
            image_end = [
                _image_end.resize(image_start[0].size if isinstance(image_start, list) else image_start.size)
                for _image_end in image_end
            ]
            end_video = torch.cat(
                [
                    torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
                    for _image_end in image_end
                ],
                dim=2,
            )
            input_video[:, :, -len(end_video) :] = end_video

            input_video_mask[:, :, -len(image_end) :] = 0
        else:
            image_end = image_end.resize(image_start[0].size if isinstance(image_start, list) else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if isinstance(validation_image_start, str) and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = cast(list[Image.Image], validation_image_start)
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None

        if isinstance(image_start, list):
            clip_image = cast(list[Image.Image], clip_image)[0]
            start_video = torch.cat(
                [
                    torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
                    for _image_start in image_start
                ],
                dim=2,
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, : len(image_start)] = start_video
            input_video = input_video / 255

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video = (
                torch.tile(
                    torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
                    [1, 1, video_length, 1, 1],
                )
                / 255
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[
                :,
                :,
                1:,
            ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return input_video, input_video_mask, cast(Image.Image | None, clip_image)
