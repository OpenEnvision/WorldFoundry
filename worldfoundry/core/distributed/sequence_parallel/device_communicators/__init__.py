"""Device communicators for sequence parallel: CUDA NCCL vs CPU gloo.

Pick CUDA when tensors are on GPU; CPU communicator is the gloo fallback
for metadata. ``pynccl`` is an optional fast path when the extension
builds.

This package is imported on demand by :class:`~.parallel_state.GroupCoordinator`
so a CPU-only process never ``dlopen``s ``libnccl``. It is not a public
facade — call :mod:`~worldfoundry.core.distributed.sequence_parallel`
collectives instead of constructing a communicator here.

Public surface (internal): :class:`DeviceCommunicatorBase`,
:class:`CudaCommunicator`, :class:`CpuCommunicator`,
:class:`PyNcclCommunicator`.
"""
