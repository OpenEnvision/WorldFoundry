"""Deterministic PyTorch observation preprocessing for OpenPI inference."""

from collections.abc import Sequence
from types import SimpleNamespace

import torch

from .. import image_tensor as image_tools

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Resize and mask an OpenPI observation without stochastic augmentation."""

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        channels_first = image.ndim == 4 and image.shape[1] in {1, 3, 4}
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != image_resolution:
            image = image_tools.resize_with_pad_torch(image, *image_resolution)
        if channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image

    batch_shape = observation.state.shape[:-1]
    out_masks = {
        key: observation.image_masks.get(
            key,
            torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device),
        )
        for key in out_images
    }
    return SimpleNamespace(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
    )
