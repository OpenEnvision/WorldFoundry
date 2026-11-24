"""Merge a released PEFT adapter into its base model for inference."""

from __future__ import annotations

from typing import Any

from torch import nn


def merge_peft_adapter(model: nn.Module) -> nn.Module:
    """Safely merge a loaded LoRA adapter into its base model."""
    if not isinstance(model, nn.Module):
        raise TypeError("PEFT adapter model must be an nn.Module")
    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("PEFT adapter model must expose merge_and_unload")
    get_base_model = getattr(model, "get_base_model", None)
    # Base models carry PEFT's optional config attribute beyond nn.Module's declared state.
    base_model: Any = get_base_model() if callable(get_base_model) else model
    had_config = hasattr(base_model, "config")
    original_config = getattr(base_model, "config", None)
    injected_empty_config = original_config is None
    if injected_empty_config:
        # PEFT 0.20 checks tied embeddings through a config mapping, while
        # custom diffusion modules can omit config or explicitly set it to None.
        base_model.config = {}
    try:
        merged = merge(safe_merge=True)
    except Exception:
        if injected_empty_config:
            if had_config:
                base_model.config = original_config
            else:
                delattr(base_model, "config")
        raise
    if not isinstance(merged, nn.Module):
        raise TypeError(f"PEFT merge returned {type(merged).__name__}, expected nn.Module")
    if injected_empty_config:
        if had_config:
            merged.config = original_config
        elif hasattr(merged, "config"):
            delattr(merged, "config")
    residual = tuple(name for name, _ in merged.named_parameters() if "lora_" in name.lower())
    if residual:
        raise RuntimeError(f"merged model still contains LoRA parameters: {residual}")
    return merged


__all__ = ["merge_peft_adapter"]
