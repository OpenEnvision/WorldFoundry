# Inference-only Being-H0.5 runtime retained in-tree.
# Copyright (c) 2024 OpenGVLab
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by BeindBeyond Ltd. and/or its affiliates. on 2026-01-10.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.

import io
from collections import Counter

import torchvision.transforms as T
from PIL import Image

from .constants import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, SIGLIP_MEAN, SIGLIP_STD

try:
    from petrel_client.client import Client  # noqa: F401 - Import checks optional dependency availability.
    from petrel_client.common.config import Config  # noqa: F401 - Import checks optional dependency availability.
except ImportError:
    print("petrel_client is not installed. If you read data locally instead of from ceph, ignore it.")


# ==============================================================================
# Text Quality Utilities
# ==============================================================================


def calculate_ngram_repetition(text, n):
    words = text.split()
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0


def check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10):
    for conversation in conversations:
        if conversation["from"] == "gpt":
            model_answer = conversation["value"]
            repeat_ratio = calculate_ngram_repetition(model_answer, ngram)
            if repeat_ratio > repeat_threshold:
                raise Exception


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert("RGB")


def expand2square(pil_img, background_color):
    """
    Expand image to square by padding.

    Args:
        pil_img: Input PIL image
        background_color: RGB tuple for padding

    Returns:
        Square PIL image
    """
    width, height = pil_img.size
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


def simulate_jpeg_degradation(quality):
    """
    Create transform that simulates JPEG compression.

    Args:
        quality: JPEG quality (1-100)

    Returns:
        Transform function
    """

    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find closest aspect ratio from candidates.

    Args:
        aspect_ratio: Input aspect ratio
        target_ratios: List of (width_ratio, height_ratio) tuples
        width: Input width
        height: Input height
        image_size: Base image size

    Returns:
        Best (width_ratio, height_ratio) tuple
    """
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
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image into multiple crops.

    Splits image into grid based on aspect ratio to minimize padding.

    Args:
        image: Input PIL image
        min_num: Minimum number of crops
        max_num: Maximum number of crops
        image_size: Size of each crop
        use_thumbnail: Whether to add thumbnail as last crop

    Returns:
        List of cropped images
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
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


# ==============================================================================
# Transform Builders
# ==============================================================================


def build_vit_transform_base(force_image_size, transform_type="default", normalize_type="imagenet", **kwargs):
    """
    Build image transforms for vision transformer.

    Args:
        force_image_size: Target image size
        transform_type: 'default' or 'dynamic_size'
        normalize_type: 'imagenet', 'clip', or 'siglip'
        pad2square: Whether to pad to square (unused)
        **kwargs: Ignored compatibility arguments.

    Returns:
        Tuple of (pre_transform, transform)
        pre_transform is None except for 'dynamic_size'
    """
    if normalize_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    pre_transform = None  # this is not None only when transform_type=="dynamic_size"
    if transform_type != "default":
        raise ValueError(f"Unsupported inference transform_type: {transform_type}")
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(force_image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(force_image_size),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )

    return pre_transform, transform
