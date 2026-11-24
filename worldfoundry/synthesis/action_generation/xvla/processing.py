"""Inference processor for local X-VLA checkpoints.

Adapted from ``models/processing_xvla.py`` in 2toINF/X-VLA at revision
``6bc2513f5f1cbec715cc668b414392a6cae5c671``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from transformers import ProcessorMixin


class XVLAProcessor(ProcessorMixin):
    """Combine a Florence-2 image processor and BART tokenizer for X-VLA."""

    attributes: ClassVar[list] = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")
    num_views = 3
    language_max_length = 50

    def __init__(self, image_processor=None, tokenizer=None) -> None:
        super().__init__(image_processor, tokenizer)

    def encode_language(self, language_instruction: str | Sequence[str]) -> dict[str, torch.Tensor]:
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]
        tokenized = self.tokenizer(
            list(language_instruction),
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        return {"input_ids": tokenized["input_ids"]}

    @staticmethod
    def _batch_views(images: Any) -> list[list[Any]]:
        if images is None:
            return []
        if hasattr(images, "shape") or hasattr(images, "convert"):
            return [[images]]
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
            return [[images]]
        values = list(images)
        if not values:
            raise ValueError("X-VLA image input is empty")
        if isinstance(values[0], (list, tuple)):
            return [list(sample) for sample in values]
        return [values]

    def encode_image(self, images: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        batch_images: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        for sample in self._batch_views(images):
            if not sample:
                raise ValueError("Every X-VLA batch item needs at least one image")
            sample = sample[: self.num_views]
            processed = self.image_processor(sample, return_tensors="pt", **kwargs)["pixel_values"]
            valid_views = min(int(processed.shape[0]), self.num_views)
            processed = processed[: self.num_views]
            if valid_views < self.num_views:
                padding = processed.new_zeros(self.num_views - valid_views, *processed.shape[1:])
                processed = torch.cat((processed, padding), dim=0)
            mask = torch.zeros(self.num_views, dtype=torch.bool)
            mask[:valid_views] = True
            batch_images.append(processed)
            batch_masks.append(mask)
        return {
            "image_input": torch.stack(batch_images),
            "image_mask": torch.stack(batch_masks),
        }

    def __call__(
        self,
        images: Any = None,
        language_instruction: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))
        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))
        if "input_ids" in outputs and "image_input" in outputs:  # noqa: SIM102 - Statement form kept for readability.
            if int(outputs["input_ids"].shape[0]) != int(outputs["image_input"].shape[0]):
                raise ValueError("X-VLA text and image batch sizes do not match")
        return outputs


__all__ = ["XVLAProcessor"]
