"""Utilities for traversing and transforming neural-network module trees.

``named_apply`` walks children in a stable order for init and wrapping
(VRAM / compile). Prefer this over ad-hoc ``modules()`` loops.
"""

from __future__ import annotations

from typing import Callable

from torch import nn


# ──────────────────────────────────────────────────────────────────────────
# Named tree walk — stable qualified names for init and VRAM / compile wrap
# ──────────────────────────────────────────────────────────────────────────


def named_apply(
    fn: Callable[..., object],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply ``fn(module=..., name=...)`` recursively with stable module names.

    This is the named equivalent of :meth:`torch.nn.Module.apply`. By default,
    children are visited depth-first and the supplied root is omitted, matching
    the behavior historically used by the DINO-style backbones in WorldFoundry.
    """

    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        qualified_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=qualified_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


__all__ = ["named_apply"]
