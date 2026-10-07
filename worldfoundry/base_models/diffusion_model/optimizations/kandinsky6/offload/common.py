"""Shared offload types and tensor moves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

OffloadStrategy = Literal["none", "module", "block"]


def _iter_named(modules: dict[str, Any], names: Sequence[str]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name in names:
        mod = modules.get(name)
        if mod is not None:
            out.append((name, mod))
    return out


def _pin_module(module: Any) -> None:
    """Pin CPU parameters/buffers so H2D can be async."""
    if not isinstance(module, nn.Module):
        # Kandinsky6TextEmbedder — pin underlying nn.Modules.
        for attr in ("qwen", "clip"):
            sub = getattr(module, attr, None)
            if isinstance(sub, nn.Module):
                _pin_module(sub)
        return
    for p in module.parameters(recurse=True):
        if p.device.type == "cpu" and not p.is_pinned():
            p.data = p.data.pin_memory()
    for b in module.buffers(recurse=True):
        if b.device.type == "cpu" and b.is_floating_point() and not b.is_pinned():
            b.data = b.data.pin_memory()


def _module_to(module: Any, device: torch.device, *, non_blocking: bool) -> None:
    if isinstance(module, nn.Module):
        module.to(device, non_blocking=non_blocking)
        return
    # TextEmbedder-style facade
    to_fn = getattr(module, "to", None)
    if to_fn is None:
        raise TypeError(f"Cannot move {type(module)!r}: no .to()")
    try:
        to_fn(device, non_blocking=non_blocking)
    except TypeError:
        to_fn(device)
