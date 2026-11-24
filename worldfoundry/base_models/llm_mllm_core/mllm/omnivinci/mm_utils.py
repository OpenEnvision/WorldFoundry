# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Note: dynamic_preprocess and find_closest_aspect_ratio are referenced from https://github.com/OpenGVLab/InternVL


import os
from collections.abc import Sequence

import torch
from PIL import Image
from transformers import SiglipImageProcessor

from .configuration_vila import VILAConfig


def expand2square(pil_img: Image.Image, background_color: tuple[int, ...]) -> Image.Image:
    """
    Expand the given PIL image to a square shape by adding padding.

    Parameters:
    - pil_img: The PIL image to be expanded.
    - background_color: The color of the padding to be added.

    Returns:
    - The expanded PIL image.

    If the image is already square, it is returned as is.
    If the image is wider than it is tall, padding is added to the top and bottom.
    If the image is taller than it is wide, padding is added to the left and right.
    """
    width, height = pil_img.size
    if pil_img.mode == "L":
        background_color = background_color[0]
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: Sequence[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 384, use_thumbnail: bool = True
) -> list[Image.Image]:
    """Dynamically preprocess image into multiple tiles based on aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_s2_preprocess(
    image: Image.Image, s2_scales: Sequence[int] = (384, 768, 1152), max_num: int = 12, image_size: int = 384
) -> tuple[list[Image.Image], tuple[int, int]]:
    """Dynamically preprocess image with multi-scale S2 strategy."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    min_num = (s2_scales[-1] // s2_scales[0]) ** 2

    processed_images = []

    # Add tiles for all but the last scale using fixed square ratio

    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0]) ** 2

        # resize the image
        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

    # Add tiles for the last scale using dynamic aspect ratio

    # Calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    return processed_images, (target_aspect_ratio[1], target_aspect_ratio[0])


def process_image(
    image_file: str | Image.Image,
    data_args: VILAConfig,
    image_folder: str | None,
    enable_dynamic_res: bool = False,
    enable_dynamic_s2: bool = False,
    max_tiles: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int]]:
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    image = image.convert("RGB")
    if hasattr(data_args.image_processor, "crop_size"):
        # CLIP vision tower
        crop_size = data_args.image_processor.crop_size
    else:
        # SIGLIP vision tower
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        images, block_size = dynamic_s2_preprocess(
            image, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images), block_size
    if "dynamic" in data_args.image_aspect_ratio and enable_dynamic_res:
        assert crop_size["height"] == crop_size["width"]
        max_num = max_tiles if max_tiles is not None else data_args.max_tiles
        images = dynamic_preprocess(image, min_num=data_args.min_tiles, max_num=max_num, image_size=crop_size["height"])
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images)

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    elif data_args.image_aspect_ratio == "pad":
        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        # Using default behavior of the vision encoder
        # For CLIP, default is central crop
        # For Radio, default is central crop
        # For Siglip, default is resize
        # For InternVIT, default is resize
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def process_images(
    images: list[Image.Image],
    image_processor: SiglipImageProcessor,
    model_cfg: VILAConfig,
    enable_dynamic_res: bool = False,
    max_tiles: int | None = None,
) -> torch.Tensor:
    """Process a batch of images using the model's image processor."""
    model_cfg.image_processor = image_processor
    new_images = [
        process_image(image, model_cfg, None, enable_dynamic_res=enable_dynamic_res, max_tiles=max_tiles)
        for image in images
    ]

    if all(x.shape == new_images[0].shape for x in new_images):
        if len(new_images[0].shape) == 4:
            new_images = torch.cat(new_images, dim=0)
        elif len(new_images[0].shape) == 3:
            new_images = torch.stack(new_images, dim=0)
        else:
            raise ValueError(f"new_images rank does not equal to 4, rank: {len(new_images[0].shape)}")
    else:
        raise ValueError("The shape of images in new_images is different!")
    return new_images
