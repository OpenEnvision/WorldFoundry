"""Disposable mixin: release parameter / persistent-buffer storage without wiping module-ops state."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class DisposableProtocol(Protocol):
    def dispose(self) -> None: ...


class Disposable(DisposableProtocol):
    """Mixin providing :meth:`dispose`; use as ``class Foo(nn.Module, Disposable)``."""

    def dispose(self) -> None:
        """Release parameter / persistent-buffer storage to meta.
        Non-persistent buffers (module-ops state) are left intact. Sync /
        ``empty_cache`` are owned by the surrounding context (e.g. ``gpu_model``).
        """
        dispose_model(cast(nn.Module, self))


def dispose_model(model: nn.Module) -> None:
    """Release persistent tensors while retaining loader-installed scratch buffers."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"{type(model).__name__} must be an nn.Module to dispose")

    persistent = set(model.state_dict().keys())
    for name, param in list(model.named_parameters()):
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr,
            nn.Parameter(torch.empty_like(param, device="meta"), requires_grad=param.requires_grad),
        )

    for name, buf in list(model.named_buffers()):
        if name not in persistent:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent.register_buffer(attr, torch.empty_like(buf, device="meta"), persistent=True)
