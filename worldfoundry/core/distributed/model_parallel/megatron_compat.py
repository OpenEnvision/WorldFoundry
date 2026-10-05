"""Compatibility wrapper for optional Megatron model-parallel state.

Vendored checkpoints may call ``mpu.get_tensor_model_parallel_rank``.
This module maps those names onto WorldFoundry groups when Megatron is
absent so a DiT does not hard-depend on megatron-core.

Public surface: ``parallel_state``, ``mpu``, the Linear/Embedding stubs,
and the identity TP mappings. Prefer :mod:`model_parallel_groups` in new code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ──────────────────────────────────────────────────────────────────────────
# Prefer real megatron-core; otherwise install 1×1 stubs (except block below)
# ──────────────────────────────────────────────────────────────────────────

try:
    from megatron.core import ModelParallelConfig, mpu, parallel_state
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.tensor_parallel.utils import VocabUtility
except Exception:  # pragma: no cover - Megatron is optional for many runtimes.

    # ──────────────────────────────────────────────────────────────────────
    # Fallback stubs — TP stays 1×1 so vendored DiT imports do not require megatron-core
    # ──────────────────────────────────────────────────────────────────────

    @dataclass
    class ModelParallelConfig:
        """Minimal Megatron config: only the parallel-size fields DiT constructors read."""

        pipeline_model_parallel_size: int = 1
        tensor_model_parallel_size: int = 1
        context_parallel_size: int = 1
        sequence_parallel: bool = False

    class _NoParallelState:
        """In-process stand-in for ``megatron.core.parallel_state``.

        Builds CP/DP subgroups from the default process group when one exists.
        TP and PP stay size 1: this stub is for inference-time rank queries,
        not for training a Megatron tensor-parallel Linear.
        """

        _initialized = False
        _context_parallel_group = None
        _context_parallel_size = 1
        _context_parallel_rank = 0
        _data_parallel_group = None
        _data_parallel_rank = 0
        _data_parallel_size = 1

        @classmethod
        def is_initialized(cls):
            """True after :meth:`initialize_model_parallel` has run once."""

            return cls._initialized

        @classmethod
        def initialize_model_parallel(cls, *args, **kwargs):
            """Create CP (and optional DP) subgroups from the live default group.

            Without a process group the stub marks itself initialized at CP=1
            so later rank queries return 0 instead of raising. ``world_size``
            must divide by ``context_parallel_size``.
            """

            import torch.distributed as dist

            context_parallel_size = int(kwargs.get("context_parallel_size", 1) or 1)
            if not dist.is_available() or not dist.is_initialized():
                cls._initialized = True
                cls._context_parallel_size = 1
                cls._context_parallel_rank = 0
                return None

            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if context_parallel_size < 1:
                context_parallel_size = 1
            if world_size % context_parallel_size != 0:
                raise RuntimeError(
                    f"world_size {world_size} must be divisible by context_parallel_size {context_parallel_size}"
                )

            if context_parallel_size == world_size:
                cls._context_parallel_group = dist.group.WORLD
                cls._context_parallel_rank = rank
            else:
                for start in range(0, world_size, context_parallel_size):
                    ranks = list(range(start, start + context_parallel_size))
                    group = dist.new_group(ranks=ranks)
                    if rank in ranks:
                        cls._context_parallel_group = group
                        cls._context_parallel_rank = ranks.index(rank)

            data_parallel_size = world_size // context_parallel_size
            if data_parallel_size == 1:
                cls._data_parallel_group = None
                cls._data_parallel_rank = 0
            else:
                context_offset = rank % context_parallel_size
                for offset in range(context_parallel_size):
                    ranks = list(range(offset, world_size, context_parallel_size))
                    group = dist.new_group(ranks=ranks)
                    if offset == context_offset:
                        cls._data_parallel_group = group
                        cls._data_parallel_rank = ranks.index(rank)

            cls._context_parallel_size = context_parallel_size
            cls._data_parallel_size = data_parallel_size
            cls._initialized = True
            return None

        @classmethod
        def destroy_model_parallel(cls):
            """Drop cached groups without calling ``destroy_process_group``.

            The default group is owned by the launcher; destroying it here
            would abort an unrelated NCCL job that still needs WORLD.
            """

            cls._initialized = False
            cls._context_parallel_group = None
            cls._context_parallel_size = 1
            cls._context_parallel_rank = 0
            cls._data_parallel_group = None
            cls._data_parallel_rank = 0
            cls._data_parallel_size = 1
            return None

        @staticmethod
        def get_tensor_model_parallel_world_size():
            """Always 1 — this stub never shards hidden dimensions."""

            return 1

        @staticmethod
        def get_tensor_model_parallel_rank():
            """Always 0 — there is no tensor-parallel subgroup."""

            return 0

        @staticmethod
        def get_tensor_model_parallel_group():
            """``None``: callers must treat TP as disabled."""

            return None

        @classmethod
        def get_context_parallel_world_size(cls):
            """Cached CP size from the last :meth:`initialize_model_parallel`."""

            return cls._context_parallel_size

        @classmethod
        def get_context_parallel_rank(cls):
            """This process's index inside the CP subgroup."""

            return cls._context_parallel_rank

        @classmethod
        def get_context_parallel_group(cls):
            """CP process group, or ``None`` / WORLD when CP equals the world."""

            return cls._context_parallel_group

        @classmethod
        def get_data_parallel_group(cls, with_context_parallel: bool = False):
            """DP group orthogonal to CP. The Megatron flag is ignored here."""

            del with_context_parallel
            return cls._data_parallel_group

        @classmethod
        def get_data_parallel_rank(cls, with_context_parallel: bool = False):
            """This process's DP index. The Megatron flag is ignored here."""

            del with_context_parallel
            return cls._data_parallel_rank

        @classmethod
        def get_data_parallel_world_size(cls, with_context_parallel: bool = False):
            """Number of DP replicas. The Megatron flag is ignored here."""

            del with_context_parallel
            return cls._data_parallel_size

        @staticmethod
        def get_pipeline_model_parallel_rank():
            """Always 0 — this stub never pipelines stages."""

            return 0

    parallel_state = _NoParallelState()
    mpu = parallel_state

    class VocabUtility:
        """Single-process vocabulary partition helper."""

        @staticmethod
        def vocab_range_from_global_vocab_size(global_size: int, rank: int, world_size: int) -> tuple[int, int]:
            """Return ``[start, end)`` for an even vocab split; refuse a remainder."""

            if global_size % world_size != 0:
                raise ValueError(
                    f"vocabulary size {global_size} must be divisible by tensor parallel size {world_size}"
                )
            partition_size = global_size // world_size
            return rank * partition_size, (rank + 1) * partition_size

    class VocabParallelEmbedding(torch.nn.Embedding):
        """Megatron-compatible embedding for single-process inference."""

        def __init__(self, num_embeddings, embedding_dim, *, init_method=None, config=None, **kwargs):
            """Ignore Megatron ``config`` / extra kwargs; store a full vocab range."""

            del config, kwargs
            super().__init__(num_embeddings, embedding_dim)
            self.tensor_model_parallel_size = 1
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings
            if init_method is not None:
                init_method(self.weight)

    class _LinearBase(torch.nn.Linear):
        """Unsharded Linear that still returns Megatron's ``(output, bias)`` tuple."""

        def __init__(self, input_size, output_size, *, bias=True, init_method=None, config=None, **kwargs):
            """Drop Megatron-only kwargs so a vendored checkpoint can construct us."""

            del config, kwargs
            super().__init__(input_size, output_size, bias=bias)
            if init_method is not None:
                init_method(self.weight)

        def forward(self, input_: torch.Tensor, *args, **kwargs):
            """Return ``(y, None)`` so callers that unpack a bias residual keep working."""

            del args, kwargs
            return super().forward(input_), None

    class ColumnParallelLinear(_LinearBase):
        """Megatron-compatible column linear for single-process inference."""

    class RowParallelLinear(_LinearBase):
        """Megatron-compatible row linear for single-process inference."""

    def reduce_from_tensor_model_parallel_region(tensor):
        """Identity: there is no TP group to all-reduce across."""

        return tensor

    def reduce_scatter_to_sequence_parallel_region(tensor):
        """Identity: sequence-parallel reduce-scatter is a no-op at SP=1."""

        return tensor

    def model_parallel_cuda_manual_seed(seed: int) -> None:
        """Seed CPU and every visible CUDA device; no per-TP offset in the stub."""

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "ColumnParallelLinear",
    "ModelParallelConfig",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabUtility",
    "model_parallel_cuda_manual_seed",
    "mpu",
    "parallel_state",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
]
