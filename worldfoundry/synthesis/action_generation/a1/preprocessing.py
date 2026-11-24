"""Inference-only tokenizer, image, state, and action-token preprocessing for A1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .configuration import A1Config

IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_START_TOKEN = "<im_start>"
IMAGE_END_TOKEN = "<im_end>"
IMAGE_COLUMN_TOKEN = "<im_col>"
IMAGE_PROMPT_TOKEN = "<|image|>"
ACTION_START_TOKEN = "<action_start>"
ACTION_END_TOKEN = "<action_end>"
EMPTY_ACTION_TOKEN = "<empty_action>"
RIGHT_EEF_TOKEN = "<right_end_effector>"
LEFT_EEF_TOKEN = "<left_end_effector>"
MOBILE_BASE_TOKEN = "<mobile_base>"
PROPRIO_TOKEN = "<proprioception>"
TIMESTEP_TOKEN = "<timestep>"

RIGHT_AXIS_TOKENS = (
    "<r_eef_xxxx>",
    "<r_eef_yyyy>",
    "<r_eef_zzzz>",
    "<r_eef_roll>",
    "<r_eef_pitch>",
    "<r_eef_yaw>",
    "<r_eef_gripper>",
)
LEFT_AXIS_TOKENS = (
    "<left_eef_x>",
    "<left_eef_y>",
    "<left_eef_z>",
    "<left_eef_roll>",
    "<left_eef_pitch>",
    "<left_eef_yaw>",
    "<left_eef_gripper>",
)
EXTRA_TOKENS = (
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    IMAGE_PATCH_TOKEN,
    IMAGE_COLUMN_TOKEN,
    IMAGE_PROMPT_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    EMPTY_ACTION_TOKEN,
    RIGHT_EEF_TOKEN,
    LEFT_EEF_TOKEN,
    MOBILE_BASE_TOKEN,
    PROPRIO_TOKEN,
    TIMESTEP_TOKEN,
    *RIGHT_AXIS_TOKENS,
    *LEFT_AXIS_TOKENS,
)

OPENAI_CLIP_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
OPENAI_CLIP_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


def load_local_tokenizer(location: str | Path, config: A1Config):
    """Load a tokenizer without executing checkpoint code or accessing the network."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("A1 tokenizer loading requires transformers") from error

    path = Path(location).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"A1 tokenizer directory does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )
    added = tokenizer.get_added_vocab()
    if not all(token in added for token in EXTRA_TOKENS):
        padding = max(config.embedding_size - len(tokenizer), 0)
        fillers = [f"|<A1_EXTRA_{index}>|" for index in range(padding)]
        tokenizer.add_special_tokens({"additional_special_tokens": [*fillers, *EXTRA_TOKENS]})
    special_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in EXTRA_TOKENS}
    if len(set(special_ids.values())) != len(EXTRA_TOKENS):
        raise ValueError("A1 tokenizer did not assign distinct IDs to all multimodal tokens")
    maximum = config.embedding_size + config.additional_vocab_size
    invalid = {token: index for token, index in special_ids.items() if index < 0 or index >= maximum}
    if invalid:
        raise ValueError(f"A1 tokenizer special-token IDs exceed checkpoint embeddings: {invalid}")
    return tokenizer, special_ids


def _as_rgb_array(value: Any) -> np.ndarray:
    if isinstance(value, (str, Path)):
        from PIL import Image

        with Image.open(Path(value).expanduser()) as image:
            return np.asarray(image.convert("RGB"))
    if hasattr(value, "convert"):
        return np.asarray(value.convert("RGB"))
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"A1 expects an HWC RGB image, got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if not np.isfinite(array).all():
        raise ValueError("A1 images must contain only finite values")
    if array.dtype != np.uint8:
        low, high = float(array.min()), float(array.max())
        if low >= 0.0 and high <= 1.0:
            array = array * 255.0
        elif low >= -1.0 and high <= 1.0:
            array = (array + 1.0) * 127.5
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize_and_pad(image: np.ndarray, size: int, pad_value: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(np.float32(size) / np.float32(height), np.float32(size) / np.float32(width))
    scaled_height = max(int(np.float32(height) * scale), 1)
    scaled_width = max(int(np.float32(width) * scale), 1)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    tensor = (
        F.interpolate(
            tensor,
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .clamp_(0.0, 1.0)[0]
        .permute(1, 2, 0)
    )
    top = (size - scaled_height) // 2
    left = (size - scaled_width) // 2
    output = torch.full((size, size, 3), float(pad_value), dtype=torch.float32)
    output[top : top + scaled_height, left : left + scaled_width] = tensor
    mask = torch.zeros(size, size, dtype=torch.float32)
    mask[top : top + scaled_height, left : left + scaled_width] = 1.0
    return output.numpy(), mask.numpy()


def _pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    shaped = array.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        *(() if array.ndim == 2 else (channels,)),
    )
    shaped = shaped.transpose(0, 2, 1, 3) if array.ndim == 2 else shaped.transpose(0, 2, 1, 3, 4)
    return shaped.reshape(-1, patch_size * patch_size * channels)


@dataclass
class A1PreparedBatch:
    tensors: dict[str, torch.Tensor]
    original_state_dim: int
    camera_count: int


class A1Processor:
    """Reproduce the released inference collation without dataset/training code."""

    def __init__(self, config: A1Config, tokenizer: Any, special_ids: dict[str, int]) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.special_ids = special_ids

    def _image_block(self) -> list[int]:
        side = self.config.image_size // self.config.image_patch_size
        pooled_h = side // self.config.image_pooling_h
        pooled_w = side // self.config.image_pooling_w
        output = [self.special_ids[IMAGE_START_TOKEN]]
        for _ in range(pooled_h):
            output.extend([self.special_ids[IMAGE_PATCH_TOKEN]] * pooled_w)
            if self.config.use_col_tokens:
                output.append(self.special_ids[IMAGE_COLUMN_TOKEN])
        output.append(self.special_ids[IMAGE_END_TOKEN])
        return output

    def _action_slots(self) -> list[int]:
        ids = self.special_ids
        right_tokens = [ids[token] for token in RIGHT_AXIS_TOKENS]
        if self.config.right_end_effector_dim > len(right_tokens):
            right_tokens.extend([ids[RIGHT_EEF_TOKEN]] * (self.config.right_end_effector_dim - len(right_tokens)))
        slots = right_tokens[: self.config.right_end_effector_dim]
        if self.config.action_use_left_eef:
            left_tokens = [ids[token] for token in LEFT_AXIS_TOKENS]
            if self.config.left_end_effector_dim > len(left_tokens):
                left_tokens.extend([ids[LEFT_EEF_TOKEN]] * (self.config.left_end_effector_dim - len(left_tokens)))
            slots.extend(left_tokens[: self.config.left_end_effector_dim])
        if self.config.action_use_mobile_base:
            slots.extend([ids[MOBILE_BASE_TOKEN]] * self.config.mobile_base_dim)
        if len(slots) < self.config.action_token_dim:
            slots.extend([ids[EMPTY_ACTION_TOKEN]] * (self.config.action_token_dim - len(slots)))
        return slots[: self.config.action_token_dim]

    def _process_images(
        self, values: Sequence[Any]
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_tokens: list[int] = []
        crops: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        feature_positions: list[list[int]] = []
        pooling: list[list[int]] = []
        patch_side = self.config.image_size // self.config.image_patch_size
        pooled_side_h = patch_side // self.config.image_pooling_h
        pooled_side_w = patch_side // self.config.image_pooling_w
        crop_repetitions = 2 if self.config.crop_mode == "overlap-and-resize-c2" else 1
        if self.config.crop_mode not in {"resize", "overlap-and-resize-c2"}:
            raise ValueError(f"Unsupported A1 inference crop mode: {self.config.crop_mode!r}")

        for value in values:
            image = _as_rgb_array(value)
            resized, pixel_mask = _resize_and_pad(image, self.config.image_size, self.config.pad_value)
            normalized = (resized - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD
            patches = _pixels_to_patches(normalized, self.config.image_patch_size).astype(np.float32)
            mask_patches = _pixels_to_patches(pixel_mask, self.config.image_patch_size).mean(-1).astype(np.float32)
            for _ in range(crop_repetitions):
                offset = len(crops) * patch_side * patch_side
                block_offset = len(image_tokens)
                block = self._image_block()
                image_tokens.extend(block)
                positions = [
                    block_offset + index
                    for index, token in enumerate(block)
                    if token == self.special_ids[IMAGE_PATCH_TOKEN]
                ]
                feature_positions.append(positions)
                crops.append(patches)
                masks.append(mask_patches)
                for row in range(pooled_side_h):
                    for column in range(pooled_side_w):
                        indices = []
                        for dy in range(self.config.image_pooling_h):
                            for dx in range(self.config.image_pooling_w):
                                patch_row = row * self.config.image_pooling_h + dy
                                patch_col = column * self.config.image_pooling_w + dx
                                indices.append(offset + patch_row * patch_side + patch_col)
                        pooling.append(indices)
        return (
            image_tokens,
            np.stack(crops),
            np.stack(masks),
            np.asarray(feature_positions, dtype=np.int64),
            np.asarray(pooling, dtype=np.int64),
        )

    def prepare(
        self,
        instruction: str,
        images: Sequence[Any],
        state: Any,
    ) -> A1PreparedBatch:
        if not images:
            raise ValueError("A1 requires at least one camera image")
        raw_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if not raw_state.size or not np.isfinite(raw_state).all():
            raise ValueError("A1 requires a finite, non-empty state vector")
        original_state_dim = int(raw_state.size)
        if raw_state.size > self.config.proprio_dim:
            raise ValueError(f"A1 state has {raw_state.size} values, checkpoint accepts {self.config.proprio_dim}")
        padded_state = np.pad(raw_state, (0, self.config.proprio_dim - raw_state.size))

        image_tokens, crops, image_masks, feature_positions, pooling = self._process_images(images)
        prompt_tokens = self.tokenizer.encode(" " + str(instruction), add_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.eos_token_id
        if bos is None:
            raise ValueError("A1 tokenizer has neither BOS nor EOS token")
        input_ids = [int(bos), *image_tokens, *map(int, prompt_tokens)]
        image_input_idx = feature_positions + 1

        proprio_index = len(input_ids)
        input_ids.append(self.special_ids[PROPRIO_TOKEN])
        if self.config.action_head != "flow_matching":
            input_ids.append(self.special_ids[TIMESTEP_TOKEN])
            input_ids.append(self.special_ids[ACTION_START_TOKEN])
            slots = self._action_slots()
            for _ in range(self.config.num_actions_chunk):
                input_ids.extend(slots)
            input_ids.append(self.special_ids[ACTION_END_TOKEN])
        if len(input_ids) > self.config.max_sequence_length:
            raise ValueError(
                f"A1 encoded sequence length {len(input_ids)} exceeds checkpoint limit "
                f"{self.config.max_sequence_length}"
            )

        tensors = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.bool),
            "position_ids": torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0),
            "images": torch.from_numpy(crops).unsqueeze(0),
            "image_masks": torch.from_numpy(image_masks).unsqueeze(0),
            "image_pooling": torch.from_numpy(pooling).unsqueeze(0),
            "image_input_idx": torch.from_numpy(image_input_idx).unsqueeze(0),
            "action_proprio": torch.from_numpy(padded_state).unsqueeze(0),
            "proprio_token_idx": torch.tensor([proprio_index], dtype=torch.long),
        }
        return A1PreparedBatch(
            tensors=tensors,
            original_state_dim=original_state_dim,
            camera_count=len(images),
        )


__all__ = [
    "ACTION_END_TOKEN",
    "ACTION_START_TOKEN",
    "EXTRA_TOKENS",
    "A1PreparedBatch",
    "A1Processor",
    "load_local_tokenizer",
]
