"""Wan attention LoRA target audit.

The released Wan LoRA adapters only touch the self/cross attention q/k/v/o
projections. This validates that a Wan DiT exposes exactly that target graph
before an adapter is applied. Generic PEFT merge stays in
:mod:`worldfoundry.core.model_loading.peft`.
"""

from __future__ import annotations

import re

from torch import nn

_WAN_ATTENTION_PATTERN = re.compile(
    r"^blocks\.(?P<block>\d+)\."
    r"(?P<role>(?:self|cross)_attn\.(?:q|k|v|o))$"
)
_WAN_ATTENTION_ROLES = frozenset(
    f"{attention}.{projection}" for attention in ("self_attn", "cross_attn") for projection in ("q", "k", "v", "o")
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


__all__ = ["audit_wan_lora_targets"]
