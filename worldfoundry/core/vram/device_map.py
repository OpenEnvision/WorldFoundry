"""Resident multi-GPU layer placement without CPU offload.

This is *static* placement, not runtime ping-pong. Decoder layers are split
into contiguous chunks so activations cross a GPU boundary only at chunk
edges. The rest of the model (embed, norm, lm_head) stays on ``home_device``.

Why not TP/CP here: those shard *tensor dims* inside one layer. A device map
shards *modules*. Stacking both often copies the same weight twice — pick a
parallel plan first, then call :func:`enable_balanced_device_map` only when
layers stay whole.

Weights move one submodule at a time so a 64B checkpoint is never fully
materialized on a single 80GB card.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.vram.layerwise_offload import _find_layer_container

# ──────────────────────────────────────────────────────────────────────────
# Result handle and contiguous chunk assignment
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DeviceMapHandle:
    """Result of applying a balanced layer device map."""

    enabled: bool
    layer_count: int
    devices: tuple[str, ...]
    placement: tuple[str, ...]
    reason: str = ""


def balanced_layer_devices(layer_count: int, devices: tuple[torch.device, ...]) -> tuple[torch.device, ...]:
    """Assign contiguous layer chunks so only GPU boundaries copy activations."""

    if layer_count < 0:
        raise ValueError(f"layer_count must be non-negative, got {layer_count}")
    if not devices:
        raise ValueError("balanced device map requires at least one device")
    # Remainder layers go to the first GPUs so chunk sizes differ by at most one
    # and activations still cross a device boundary only at chunk edges.
    base, extra = divmod(layer_count, len(devices))
    assignment: list[torch.device] = []
    for index, device in enumerate(devices):
        assignment.extend([device] * (base + (1 if index < extra else 0)))
    return tuple(assignment)


def visible_cuda_devices() -> tuple[torch.device, ...]:
    """Return ``cuda:0..N-1`` for the current process visibility, or empty."""
    if not torch.cuda.is_available():
        return ()
    return tuple(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))


# ──────────────────────────────────────────────────────────────────────────
# Apply the map — move one submodule at a time; never materialize the full tree
# ──────────────────────────────────────────────────────────────────────────


def enable_balanced_device_map(
    model: nn.Module,
    *,
    layer_container: str | None,
    home_device: torch.device | str,
    devices: tuple[torch.device, ...] | None = None,
) -> DeviceMapHandle:
    """Keep decoder layers resident on visible GPUs; put the rest on ``home_device``.

    Weights are moved from CPU (or their current device) one submodule at a
    time so a 64B checkpoint is never materialized on a single 80GB card.
    """

    home = torch.device(home_device)
    targets = devices if devices is not None else visible_cuda_devices()
    if not targets:
        return DeviceMapHandle(enabled=False, layer_count=0, devices=(), placement=(), reason="no CUDA devices")
    layers = _find_layer_container(model, layer_container)
    if layers is None or len(layers) == 0:
        model.to(home)
        return DeviceMapHandle(
            enabled=False,
            layer_count=0,
            devices=tuple(str(device) for device in targets),
            placement=(),
            reason="no layer container",
        )

    placement = balanced_layer_devices(len(layers), targets)
    for layer, device in zip(layers, placement):
        layer.to(device)
    # Embed, norm, and lm_head stay on ``home_device``. Identity compare (not
    # name) so a renamed ModuleList still skips the already-placed layers.
    for child in model.children():
        if child is layers:
            continue
        child.to(home)
    for _name, param in model.named_parameters(recurse=False):
        if param.device != home:
            param.data = param.data.to(home)
    for name, buffer in model.named_buffers(recurse=False):
        if buffer.device != home:
            model._buffers[name] = buffer.to(home)
    setattr(model, "_worldfoundry_device_map", "balanced")
    return DeviceMapHandle(
        enabled=True,
        layer_count=len(layers),
        devices=tuple(str(device) for device in targets),
        placement=tuple(str(device) for device in placement),
        reason="balanced",
    )


__all__ = [
    "DeviceMapHandle",
    "balanced_layer_devices",
    "enable_balanced_device_map",
    "visible_cuda_devices",
]
