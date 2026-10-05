"""Validate and merge released PEFT adapters for inference."""

from __future__ import annotations

import re

from torch import nn

_WAN_ATTENTION_PATTERN = re.compile(
    r"^blocks\.(?P<block>\d+)\."
    r"(?P<role>(?:self|cross)_attn\.(?:q|k|v|o))$"
)
_WAN_ATTENTION_ROLES = frozenset(
    f"{attention}.{projection}"
    for attention in ("self_attn", "cross_attn")
    for projection in ("q", "k", "v", "o")
)


def audit_wan_lora_targets(model: nn.Module) -> tuple[str, ...]:
    """Return the exact Wan attention targets, rejecting incomplete graphs."""
    if not isinstance(model, nn.Module):
        raise TypeError("LoRA target audit requires an nn.Module")
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("wan-attention requires a non-empty ModuleList named 'blocks'")

    names: list[str] = []
    roles_by_block: dict[int, set[str]] = {index: set() for index in range(len(blocks))}
    for name, module in model.named_modules():
        match = _WAN_ATTENTION_PATTERN.fullmatch(name)
        if match is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name!r} is {type(module).__name__}, expected nn.Linear")
        block = int(match.group("block"))
        if block not in roles_by_block:
            raise ValueError(f"LoRA target {name!r} refers to an unknown block")
        roles_by_block[block].add(match.group("role"))
        names.append(name)

    drift = {
        block: {
            "missing": sorted(_WAN_ATTENTION_ROLES - roles),
            "unexpected": sorted(roles - _WAN_ATTENTION_ROLES),
        }
        for block, roles in roles_by_block.items()
        if roles != _WAN_ATTENTION_ROLES
    }
    if drift:
        raise ValueError(f"wan-attention target graph drifted: {drift}")
    return tuple(sorted(names))


def merge_peft_adapter(model: nn.Module) -> nn.Module:
    """Safely merge a loaded LoRA adapter into its base model."""
    if not isinstance(model, nn.Module):
        raise TypeError("PEFT adapter model must be an nn.Module")
    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("PEFT adapter model must expose merge_and_unload")
    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
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


__all__ = ["audit_wan_lora_targets", "merge_peft_adapter"]
