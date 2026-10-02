"""Checkpoint loading shared by in-tree Wan inference variants.

``from_pretrained``-style helper for Fun / Camera / Action graphs that ship
``config.json`` next to weights.  Distinct from :mod:`.wan_components`, which
loads official Alibaba bundles.  Missing ``config.json`` or a non-matching
filename raises :exc:`FileNotFoundError` / :exc:`ValueError`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from worldfoundry.core.model_loading.checkpoints.file import load_state_dict


def _model_config(
    root: Path,
    additional_kwargs: Mapping | None,
) -> tuple[dict, dict]:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Wan transformer config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = dict(additional_kwargs or {})
    mapping = overrides.pop("dict_mapping", {})
    for config_key, model_key in mapping.items():
        overrides[model_key] = config[config_key]
    return config, overrides


def _compatible_state_dict(model, state_dict: dict) -> dict:
    model_state = model.state_dict()
    patch_key = "patch_embedding.weight"
    if patch_key in state_dict and patch_key in model_state:
        source = state_dict[patch_key]
        target = model_state[patch_key]
        if source.shape != target.shape:
            adjusted = source.new_zeros(target.shape)
            common = tuple(
                slice(0, min(source_size, target_size))
                for source_size, target_size in zip(source.shape, target.shape)
            )
            adjusted[common] = source[common]
            state_dict = dict(state_dict)
            state_dict[patch_key] = adjusted
    return {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }


def load_wan_transformer(
    model_class,
    pretrained_model_path: str | Path,
    *,
    subfolder: str | None = None,
    additional_kwargs: Mapping | None = None,
    low_cpu_mem_usage: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load one config-plus-state-dict Wan transformer through native PyTorch."""
    root = Path(pretrained_model_path)
    if subfolder is not None:
        root /= subfolder
    config, overrides = _model_config(root, additional_kwargs)
    converter = getattr(model_class, "_convert_from_wan_model_config", None)
    if converter is not None and "has_image_input" in config:
        config = converter(config)
    state_dict = load_state_dict(str(root), device="cpu")
    model = model_class.from_config(config, **overrides)
    state_dict = _compatible_state_dict(model, state_dict)
    # These inference checkpoints are already stored in the requested dtype.
    # ``assign=True`` replaces parameter storage with checkpoint tensors
    # instead of copying BF16 weights into a freshly allocated FP32 model and
    # then converting the entire model back to BF16.  Missing variant-specific
    # parameters retain their constructor initialization and are converted by
    # the final ``to`` below.
    model.load_state_dict(
        state_dict,
        strict=False,
        assign=bool(low_cpu_mem_usage),
    )
    return model.to(dtype=torch_dtype).eval()


def load_wan_config(
    pretrained_model_path: str | Path,
    *,
    subfolder: str | None = None,
) -> dict:
    """Read one Wan architecture config without constructing or loading a module."""

    root = Path(pretrained_model_path)
    if subfolder is not None:
        root /= subfolder
    if root.is_file():
        if root.name != "config.json":
            raise ValueError(f"Wan architecture config must be named config.json: {root}")
        return json.loads(root.read_text(encoding="utf-8"))
    config, _ = _model_config(root, None)
    return config


class WanVariantLoadingMixin:
    """Attach a native ``from_pretrained`` constructor to a Wan variant class."""
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs=None,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
    ):
        return load_wan_transformer(
            cls,
            pretrained_model_path,
            subfolder=subfolder,
            additional_kwargs=transformer_additional_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
        )


__all__ = ["WanVariantLoadingMixin", "load_wan_config", "load_wan_transformer"]
