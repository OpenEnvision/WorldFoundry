"""Gradient helpers for transformer-style training and runtime code.

WorldFoundry inference is the common path, but a few in-tree trainers
and activation-checkpoint wrappers still need:

- :func:`create_custom_forward` / :func:`gradient_checkpoint_forward` —
  a non-reentrant ``torch.utils.checkpoint`` wrapper, optionally with
  ``torch.autograd.graph.save_on_cpu`` so activations leave GPU HBM.
- :func:`clip_grad_norm_` / :func:`get_total_norm` /
  :func:`clip_grads_with_norm_` — a foreach-aware clip that also
  all-reduces the norm across a pipeline-parallel :class:`DeviceMesh`
  (MAX for inf-norm, SUM-then-root for p-norms). DTensor norms are
  materialized with ``full_tensor()`` before the reduction.

These are not a replacement for ``torch.nn.utils.clip_grad_norm_``;
they exist so PP/FSDP trainers share one implementation instead of
copying Megatron snippets into each model family.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def create_custom_forward(module):
    """Wrap *module* so :func:`torch.utils.checkpoint.checkpoint` can call it.

    Checkpoint requires a function of ``*inputs``. Returning a nested
    callable keeps ``module`` and its ``**kwargs`` bound without making
    the module itself look like a functional autograd target.
    """

    def custom_forward(*inputs, **kwargs):
        """Forward *module* so checkpoint sees a function, not a Module object."""
        return module(*inputs, **kwargs)

    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    """Run *model* with optional activation checkpointing and CPU offload.

    *use_gradient_checkpointing_offload* implies checkpointing: activations
    are saved on CPU, then the same non-reentrant checkpoint wrapper is
    used. When both flags are false the module is called normally.
    """
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
) -> torch.Tensor:
    """Clip gradients and optionally reduce the norm across pipeline-parallel stages.

    Same contract as ``torch.nn.utils.clip_grad_norm_`` plus *pp_mesh*:
    when set, the local total norm is all-reduced on that DeviceMesh
    (MAX for ``inf``, SUM of ``norm**p`` then ``**(1/p)`` otherwise)
    before scaling. DTensor norms are converted to a dense tensor first.

    Returns:
        The (possibly reduced) total gradient norm.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [param.grad for param in parameters if param.grad is not None]
    total_norm = get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    if _is_dtensor(total_norm):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        import torch.distributed as dist

        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def get_total_norm(
    tensors: torch.Tensor | Iterable[torch.Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """Compute the total norm of tensors as if their flattened values were concatenated.

    Groups tensors by device/dtype and uses ``torch._foreach_norm`` when
    the device supports it (or *foreach* is forced). Empty input yields
    a CPU scalar ``0.0``. Set *error_if_nonfinite* to raise instead of
    returning NaN/Inf.
    """

    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)

    first_device = tensors[0].device
    grouped_tensors = _group_tensors_by_device_and_dtype([tensors])
    norms = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            norms.extend(torch.linalg.vector_norm(tensor, norm_type) for tensor in device_tensors)

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from parameters is non-finite, "
            "so it cannot be clipped. To disable this error and scale the gradients by the "
            "non-finite norm anyway, set error_if_nonfinite=False."
        )
    return total_norm


@torch.no_grad()
def clip_grads_with_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: bool | None = None,
) -> None:
    """Scale parameter gradients in-place using a precomputed total norm.

    The coefficient is ``min(1, max_norm / (total_norm + 1e-6))``.
    Foreach multiply is used per device when available. No-op when no
    parameter has a gradient.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [param.grad for param in parameters if param.grad is not None]
    if len(grads) == 0:
        return

    grouped_grads = _group_tensors_by_device_and_dtype([grads])
    clip_coef = float(max_norm) / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for grad in device_grads:
                grad.mul_(clip_coef_clamped_device)


def _group_tensors_by_device_and_dtype(tensorlistlist):
    """Delegate to torch's clip-grad grouping so foreach kernels stay on one device/dtype."""
    from torch.nn.utils.clip_grad import _group_tensors_by_device_and_dtype as group_fn

    return group_fn(tensorlistlist)


def _has_foreach_support(tensors, device) -> bool:
    """True when this tensor group can use ``torch._foreach_*`` on *device*."""
    from torch.nn.utils.clip_grad import _has_foreach_support as has_support

    return has_support(tensors, device)


def _device_has_foreach_support(device) -> bool:
    """True when *device* implements foreach kernels (CPU/CUDA; not all accelerators)."""
    from torch.nn.utils.clip_grad import _device_has_foreach_support as device_has_support

    return device_has_support(device)


def _is_dtensor(value: object) -> bool:
    """Detect DTensor without importing distributed at module load when torch is CPU-only."""
    try:
        from torch.distributed._tensor.api import DTensor
    except Exception:
        return False
    return isinstance(value, DTensor)


__all__ = [
    "clip_grad_norm_",
    "clip_grads_with_norm_",
    "create_custom_forward",
    "get_total_norm",
    "gradient_checkpoint_forward",
]
