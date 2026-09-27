"""Inference preprocessing matching the official G0Plus processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


class PaliGemmaTokenizer:
    """Local-only PaliGemma prompt tokenizer with image-token insertion."""

    def __init__(
        self,
        tokenizer_path: str | Path,
        *,
        pad_token_id: int,
        image_token_index: int,
        max_text_tokens: int,
        num_tokens_per_image: int,
        num_input_images: int,
        prompt_template: str,
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.tokenizer.pad_token_id = int(pad_token_id)
        self.pad_token_id = int(pad_token_id)
        self.image_token_index = int(image_token_index)
        self.max_text_tokens = int(max_text_tokens)
        self.total_image_tokens = int(num_input_images) * int(num_tokens_per_image)
        self.max_image_text_tokens = self.total_image_tokens + self.max_text_tokens
        self.prompt_template = str(prompt_template)

    def tokenize(self, instructions: str | Sequence[str]) -> Mapping[str, Any]:
        import torch

        values = [instructions] if isinstance(instructions, str) else list(instructions)
        if not values:
            raise ValueError("G0Plus requires at least one instruction")
        if self.tokenizer.bos_token is None:
            raise ValueError("The staged PaliGemma tokenizer does not define bos_token")
        prompts = [
            self.prompt_template.format(
                bos_token=self.tokenizer.bos_token,
                instruction=value,
            )
            for value in values
        ]
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        text_ids = encoded.input_ids[:, : self.max_text_tokens]
        if text_ids.shape[1] < self.max_text_tokens:
            text_ids = torch.nn.functional.pad(
                text_ids,
                (0, self.max_text_tokens - text_ids.shape[1]),
                value=self.pad_token_id,
            )
        image_ids = torch.full(
            (text_ids.shape[0], self.total_image_tokens),
            self.image_token_index,
            dtype=text_ids.dtype,
        )
        input_ids = torch.cat((text_ids[:, :1], image_ids, text_ids[:, 1:]), dim=1)
        if input_ids.shape[1] != self.max_image_text_tokens:
            raise RuntimeError("G0Plus tokenizer produced an invalid image/text sequence length")
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }


def _frames(value: Any) -> list[Any]:
    import numpy as np
    import torch

    if isinstance(value, (str, Path)):
        from PIL import Image

        value = Image.open(value).convert("RGB")
    if hasattr(value, "convert"):
        value = np.asarray(value.convert("RGB"))
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        array = np.asarray(value)
        # PIL-backed arrays can be read-only; PyTorch warns that a tensor
        # sharing their storage has undefined behavior if modified.
        tensor = torch.as_tensor(array.copy() if not array.flags.writeable else array)
    if tensor.ndim == 3:
        return [tensor]
    if tensor.ndim == 4:
        return [tensor[index] for index in range(tensor.shape[0])]
    raise ValueError(f"G0Plus images must be rank 3 or 4, got shape {tuple(tensor.shape)}")


def prepare_images(
    images: Sequence[Any],
    *,
    image_size: int,
    expected_count: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Any:
    """Resize, tensorize and normalize camera frames without torchvision."""

    import torch
    import torch.nn.functional as functional

    frames = [frame for image in images for frame in _frames(image)]
    if len(frames) != int(expected_count):
        raise ValueError(
            f"G0Plus expects {expected_count} camera/history frames, got {len(frames)}"
        )
    output = []
    for frame in frames:
        if frame.ndim != 3:
            raise ValueError(f"G0Plus image frame must be rank 3, got {tuple(frame.shape)}")
        if frame.shape[-1] in {1, 3, 4}:
            frame = frame[..., :3].permute(2, 0, 1)
        elif frame.shape[0] not in {1, 3, 4}:
            raise ValueError(f"Cannot infer channel axis for image shape {tuple(frame.shape)}")
        frame = frame[:3]
        frame = frame.float()
        if frame.max().item() > 1.0:
            frame = frame / 255.0
        frame = functional.interpolate(
            frame.unsqueeze(0),
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        mean_tensor = torch.as_tensor(mean, dtype=frame.dtype).view(3, 1, 1)
        std_tensor = torch.as_tensor(std, dtype=frame.dtype).view(3, 1, 1)
        output.append((frame - mean_tensor) / std_tensor)
    return torch.stack(output, dim=0)


__all__ = ["PaliGemmaTokenizer", "prepare_images"]
