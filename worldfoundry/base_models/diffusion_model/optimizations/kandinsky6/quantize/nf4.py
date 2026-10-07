"""NF4 load for a Hugging Face module via bitsandbytes.

The text embedder calls this when ``text_embedder.quantized_qwen`` is set. The weights
are quantized at ``from_pretrained`` time and stay on the device they were
loaded on.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import BitsAndBytesConfig


def qwen_load_kwargs(device: torch.device | str, *, quantized: bool) -> dict[str, Any]:
    """Arguments for ``Qwen2_5_VLForConditionalGeneration.from_pretrained``."""
    kwargs: dict[str, Any] = {"device_map": str(device)}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return kwargs
    kwargs["torch_dtype"] = torch.bfloat16
    return kwargs
