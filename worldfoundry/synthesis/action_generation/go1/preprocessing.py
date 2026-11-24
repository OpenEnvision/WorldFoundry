# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Inference-only image and prompt preprocessing for GO-1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def to_pil_image(value: Any) -> Any:
    import numpy as np
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f":
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def expand_to_square(image: Any, background: Sequence[int]) -> Any:
    from PIL import Image

    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side), tuple(int(value) for value in background))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def transform_image(
    image: Any,
    *,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    pad_to_square: bool,
) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    source = to_pil_image(image)
    if pad_to_square:
        source = expand_to_square(source, [round(float(value) * 255.0) for value in mean])
    source = source.resize((int(image_size), int(image_size)), Image.Resampling.BICUBIC)
    values = np.asarray(source, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    means = torch.as_tensor(tuple(mean), dtype=tensor.dtype).view(3, 1, 1)
    stds = torch.as_tensor(tuple(std), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - means) / stds


def _closest_ratio(
    aspect_ratio: float,
    ratios: Sequence[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_difference = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference or (
            difference == best_difference and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_difference = difference
            best = ratio
    return best


def dynamic_tiles(
    image: Any,
    *,
    min_patches: int,
    max_patches: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Any]:
    from PIL import Image

    source = to_pil_image(image)
    width, height = source.size
    ratios = sorted(
        {
            (columns, rows)
            for count in range(int(min_patches), int(max_patches) + 1)
            for columns in range(1, count + 1)
            for rows in range(1, count + 1)
            if int(min_patches) <= columns * rows <= int(max_patches)
        },
        key=lambda value: value[0] * value[1],
    )
    columns, rows = _closest_ratio(
        width / height,
        ratios,
        width=width,
        height=height,
        image_size=int(image_size),
    )
    resized = source.resize(
        (int(image_size) * columns, int(image_size) * rows),
        Image.Resampling.BICUBIC,
    )
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * int(image_size)
        top = (index // columns) * int(image_size)
        tiles.append(resized.crop((left, top, left + int(image_size), top + int(image_size))))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(source.resize((int(image_size), int(image_size)), Image.Resampling.BICUBIC))
    return tiles


def _tokenize_segments(tokenizer: Any, segments: list[str], max_length: int) -> Any:
    import numpy as np
    import torch

    if bool(getattr(tokenizer, "add_bos_token", False)):
        segments[0] = str(tokenizer.bos_token) + segments[0]
    encoded = tokenizer(
        segments,
        return_tensors="np",
        padding=False,
        max_length=int(max_length),
        truncation=False,
    ).input_ids
    if bool(getattr(tokenizer, "add_bos_token", False)):
        encoded = [item[1:] for item in encoded]
    return torch.as_tensor(np.concatenate(encoded)[: int(max_length)], dtype=torch.long)


def prepare_inputs(
    *,
    images: Sequence[Any],
    instruction: str,
    tokenizer: Any,
    num_image_tokens: int,
    image_size: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
    min_dynamic_patches: int,
    max_dynamic_patches: int,
    pad_to_square: bool,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    max_sequence_length: int,
    prompt_template: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    if not images:
        raise ValueError("GO-1 requires at least one camera image")
    tiles: list[Any] = []
    tiles_per_image: list[int] = []
    for image in images:
        selected = (
            dynamic_tiles(
                image,
                min_patches=int(min_dynamic_patches),
                max_patches=int(max_dynamic_patches),
                image_size=int(image_size),
                use_thumbnail=bool(use_thumbnail),
            )
            if dynamic_image_size
            else [to_pil_image(image)]
        )
        tiles.extend(selected)
        tiles_per_image.append(len(selected))
    pixel_values = torch.stack(
        [
            transform_image(
                tile,
                image_size=int(image_size),
                mean=normalization_mean,
                std=normalization_std,
                pad_to_square=bool(pad_to_square),
            )
            for tile in tiles
        ]
    )

    required_keys = {
        "image_start",
        "image_context",
        "image_end",
        "message_start",
        "message_end",
        "system_message",
        "instruction_format",
    }
    missing = required_keys.difference(prompt_template)
    if missing:
        raise ValueError(f"GO-1 prompt_template is missing keys: {sorted(missing)}")
    image_prefix = "".join(
        str(prompt_template["image_start"])
        + str(prompt_template["image_context"]) * (int(num_image_tokens) * count)
        + str(prompt_template["image_end"])
        for count in tiles_per_image
    )
    instruction_text = str(prompt_template["instruction_format"]).format(instruction=instruction)
    message_start = str(prompt_template["message_start"])
    message_end = str(prompt_template["message_end"])
    segments = [
        f"{message_start}system\n{prompt_template['system_message']}{message_end}\n",
        f"{message_start}user\n{image_prefix}{instruction_text}{message_end}\n",
        f"{message_start}assistant\n{message_end}\n",
    ]
    input_ids = _tokenize_segments(tokenizer, segments, int(max_sequence_length))
    attention_mask = input_ids.ne(int(tokenizer.pad_token_id))
    image_end_id = int(tokenizer.convert_tokens_to_ids(str(prompt_template["image_end"])))
    actual_images = int(input_ids.eq(image_end_id).sum().item())
    if actual_images != len(images):
        raise ValueError(
            f"GO-1 prompt was truncated or tokenized incorrectly: "
            f"expected {len(images)} image terminators, got {actual_images}"
        )
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(~attention_mask, 1)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_flags": torch.ones(pixel_values.shape[0], dtype=torch.long),
        "tiles_per_image": tiles_per_image,
    }


__all__ = ["dynamic_tiles", "prepare_inputs", "to_pil_image", "transform_image"]
