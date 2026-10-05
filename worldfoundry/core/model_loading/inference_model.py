"""Minimal model base and state helpers required by inference runtimes.

:class:`InferenceModel` is a thin ``nn.Module`` with a
:meth:`~InferenceModel.prepare_inference` hook that upstream models use
for tokenizer / weight setup. :func:`instantiate_inference_network`
builds a network on the meta device, materializes it, optionally FSDP-
shards it, and broadcasts DTensor states.

Not a training Lightning module and not a full model zoo.
"""

from __future__ import annotations

from typing import Any

import torch

# ──────────────────────────────────────────────────────────────────────────
# Thin nn.Module — prepare_inference hook only; not a training Lightning module
# ──────────────────────────────────────────────────────────────────────────


class InferenceModel(torch.nn.Module):
    """Compatibility base containing inference lifecycle hooks only."""

    def __init__(self) -> None:
        """Construct an empty ``nn.Module``; subclasses add weights and hooks."""
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Subclasses must implement the inference forward pass."""
        raise NotImplementedError

    def prepare_inference(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """Initialize weights/tokenizers; retained because upstream models use this hook for loading."""


# ──────────────────────────────────────────────────────────────────────────
# Meta construct → materialize → optional FSDP shard → DTensor broadcast
# ──────────────────────────────────────────────────────────────────────────


def instantiate_inference_network(
    config: Any,
    *,
    device: str = "cuda",
    device_mesh: Any = None,
    internal_mixed_precision_policy: Any = None,
    root_mixed_precision_policy: Any = None,
    shard_before_materialize: bool = False,
) -> torch.nn.Module:
    """Instantiate, materialize, initialize, and optionally shard an inference network."""

    from worldfoundry.core.configuration.lazy_config import instantiate

    with torch.device("meta"):
        network = instantiate(config)

    internal_kwargs = (
        {"mp_policy": internal_mixed_precision_policy} if internal_mixed_precision_policy is not None else {}
    )
    root_kwargs = {"mp_policy": root_mixed_precision_policy} if root_mixed_precision_policy is not None else {}

    def shard(module: torch.nn.Module) -> torch.nn.Module:
        """FSDP-wrap *module* on ``device_mesh``; reshard after each forward.

        Internal children get ``internal_kwargs``; the root gets
        ``root_kwargs`` so mixed-precision policy can differ at the boundary.
        """
        from torch.distributed._composable.fsdp import fully_shard

        module.fully_shard(mesh=device_mesh, **internal_kwargs)
        return fully_shard(
            module,
            mesh=device_mesh,
            reshard_after_forward=True,
            **root_kwargs,
        )

    if device_mesh is not None and shard_before_materialize:
        network = shard(network)
    network.to_empty(device=device)
    network.init_weights()
    if device_mesh is not None:
        if not shard_before_materialize:
            network = shard(network)
        from torch.distributed.tensor import DTensor

        from worldfoundry.core.distributed.collectives.device_mesh import (
            broadcast_dtensor_model_states,
        )

        broadcast_dtensor_model_states(network, device_mesh)
        for name, parameter in network.named_parameters():
            if not isinstance(parameter, DTensor):
                raise TypeError(f"expected DTensor parameter {name}, got {type(parameter)}")
    return network


__all__ = [
    "InferenceModel",
    "instantiate_inference_network",
]
