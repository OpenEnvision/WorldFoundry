# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/cuda_communicator.py
"""CUDA NCCL communicator for sequence-parallel tensor collectives.

:class:`CudaCommunicator` is the GPU :class:`DeviceCommunicatorBase`
implementation. For ``world_size > 1`` it owns a
:class:`PyNcclCommunicator` and routes ``all_reduce`` / ``all_gather`` /
``all_to_all`` through it so Ulysses attention can be CUDA-Graph
captured. Rank-0 metadata still uses the CPU process group.

Prefer this over calling ``torch.distributed`` directly from an SP
attention kernel.
"""

import torch
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase


# ──────────────────────────────────────────────────────────────────────────
# CUDA path — PyNCCL for Graph-safe all-reduce / P2P; torch as test fallback
# ──────────────────────────────────────────────────────────────────────────


class CudaCommunicator(DeviceCommunicatorBase):
    """GPU communicator that prefers in-tree PyNCCL for Graph-safe collectives."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        """Attach PyNCCL to the *CPU* group so unique-id broadcast stays off NCCL."""
        super().__init__(cpu_group, device, device_group, unique_name)

        from .pynccl import PyNcclCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

    def all_reduce(self, input_, op: torch.distributed.ReduceOp | None = None):
        """PyNCCL out-of-place reduce; torch clone+reduce if the wrapper is disabled."""
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_, op=op)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group, op=op)
        return out

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:
        """Drop the PyNCCL handle; ``ncclCommDestroy`` is intentionally not called."""
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
