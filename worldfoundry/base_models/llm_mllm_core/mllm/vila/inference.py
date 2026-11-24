"""WorldFoundry VILA/NVILA inference helpers.

This module is intentionally narrow: it loads VILA-style Hugging Face
checkpoints through the in-tree model code and exposes image/video/text
generation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GenerationConfig

from .media import Image, Video
from .modeling_vila import VILAForCausalLM


def _as_media_paths(paths: str | Path | Iterable[str | Path] | None) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(path) for path in paths]


def _dtype_from_name(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.replace("torch.", "")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    value = getattr(torch, normalized)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return value


def load_vila_model(
    model_path: str | Path,
    *,
    device_map: str | dict | None = "auto",
    torch_dtype: str | torch.dtype | None = "float16",
    attn_implementation: str | None = "flash_attention_2",
    **kwargs,
) -> VILAForCausalLM:
    """Load a VILA/NVILA checkpoint using WorldFoundry's in-tree model code."""

    load_kwargs = dict(kwargs)
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    dtype = _dtype_from_name(torch_dtype)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = VILAForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    return model


def generate_vila_response(
    model: VILAForCausalLM,
    *,
    prompt: str,
    images: str | Path | Iterable[str | Path] | None = None,
    videos: str | Path | Iterable[str | Path] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float | None = None,
    generation_config: GenerationConfig | None = None,
) -> str:
    """Generate a response from text plus optional image/video media."""

    prompt_parts: list[object] = []
    prompt_parts.extend(Image(path) for path in _as_media_paths(images))
    prompt_parts.extend(Video(path) for path in _as_media_paths(videos))
    prompt_parts.append(prompt)

    config = generation_config or model.default_generation_config
    config = GenerationConfig.from_dict(config.to_dict())
    config.max_new_tokens = max_new_tokens
    config.do_sample = temperature > 0
    if temperature > 0:
        config.temperature = temperature
    if top_p is not None:
        config.top_p = top_p

    return model.generate_content(prompt_parts, generation_config=config)


@dataclass
class VILAGenerator:
    """Small reusable inference wrapper around an in-tree VILA model."""

    model: VILAForCausalLM

    @classmethod
    def from_pretrained(cls, model_path: str | Path, **kwargs) -> VILAGenerator:
        return cls(model=load_vila_model(model_path, **kwargs))

    def generate(
        self,
        prompt: str,
        *,
        images: str | Path | Sequence[str | Path] | None = None,
        videos: str | Path | Sequence[str | Path] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> str:
        return generate_vila_response(
            self.model,
            prompt=prompt,
            images=images,
            videos=videos,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
