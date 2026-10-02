"""Normalize single-image inputs and materialize them for inference."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.io.paths import resolve_worldfoundry_path


def load_pil_image(image_input) -> Image.Image:
    """Load pil image helper function."""
    if isinstance(image_input, (list, tuple)):
        items = [item for item in image_input if item is not None]
        if len(items) != 1:
            raise TypeError(f"Expected exactly one image input, got {len(items)}")
        return load_pil_image(items[0])
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, Path)):
        path = resolve_worldfoundry_path(str(image_input), env=os.environ)
        if not path.is_absolute():
            path = resolve_worldfoundry_path("${WORLDFOUNDRY_REPO_ROOT}", env=os.environ) / path
        return Image.open(path).convert("RGB")
    if isinstance(image_input, np.ndarray):
        array = image_input
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if torch.is_tensor(image_input):
        tensor = image_input.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def materialize_image_input(
    image_input,
    output_dir: str,
    filename: str = "input.png",
) -> str:
    """Materialize image input helper function."""
    image = load_pil_image(image_input)
    output_path = Path(output_dir).expanduser().resolve() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)
