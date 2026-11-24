"""Framework-independent device/dtype properties for tensor-owning modules.

Mixin so a module can report ``device``/``dtype`` without assuming
nn.Module buffers exist (meta init, DiskMap views).

Not this module:
    Parameter movement and DiskMap materialization live in
    :mod:`worldfoundry.core.vram`. This mixin only *reads* the first
    owned tensor.

Public surface:

- :func:`get_parameter_dtype` — first parameter/buffer dtype, or ``None``.
- :class:`ModuleDeviceDtypeMixin` — ``.device`` / ``.dtype`` from the
  first parameter, else the first buffer.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

# ──────────────────────────────────────────────────────────────────────────
# First-tensor probe — meta / DiskMap modules still expose a reference
# ──────────────────────────────────────────────────────────────────────────


def get_parameter_dtype(module: nn.Module) -> torch.dtype | None:
    """Return the first parameter dtype, then first buffer dtype, or None for a tensor-free module."""
    parameters = tuple(module.parameters())
    if parameters:
        return parameters[0].dtype
    buffers = tuple(module.buffers())
    if buffers:
        return buffers[0].dtype
    return None


class ModuleDeviceDtypeMixin:
    """Expose the device and dtype of a module's first tensor.

    Native model adapters use this instead of inheriting an external pipeline
    framework solely for its ``.device`` and ``.dtype`` conveniences.
    """

    def _reference_tensor(self) -> torch.Tensor:
        """Return the first parameter, else the first buffer.

        Raises:
            RuntimeError: the mixed-in module owns neither (empty / fully offloaded).
        """

        module = cast(nn.Module, self)
        reference = next(module.parameters(), None)
        if reference is None:
            reference = next(module.buffers(), None)
        if reference is None:
            raise RuntimeError(f"{type(self).__name__} does not own a parameter or buffer")
        return reference

    @property
    def device(self) -> torch.device:
        """Device of :meth:`_reference_tensor`; fails if the module is empty."""

        return self._reference_tensor().device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of :meth:`_reference_tensor`; fails if the module is empty."""

        return self._reference_tensor().dtype


__all__ = ["ModuleDeviceDtypeMixin", "get_parameter_dtype"]
