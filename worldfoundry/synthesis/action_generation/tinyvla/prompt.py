"""Exact TinyVLA Pythia prompt and image-token encoding."""

from __future__ import annotations

import torch

from .constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_prompt(instruction: str, *, use_image_start_end: bool) -> str:
    image_token = DEFAULT_IMAGE_TOKEN
    if use_image_start_end:
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    user_message = image_token + "\n" + instruction
    return f"{_SYSTEM} USER: {user_message} ASSISTANT: <|endoftext|>"


def tokenizer_image_token(prompt: str, tokenizer, image_token_index: int = IMAGE_TOKEN_INDEX) -> torch.Tensor:
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
    input_ids: list[int] = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])
    for index, chunk in enumerate(chunks):
        input_ids.extend(chunk[offset:])
        if index + 1 < len(chunks):
            input_ids.append(image_token_index)
    return torch.tensor(input_ids, dtype=torch.long)


__all__ = ["build_prompt", "tokenizer_image_token"]
