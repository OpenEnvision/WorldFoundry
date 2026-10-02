"""Small compatibility helpers around Megatron model-parallel state.

Sibling of ``megatron_compat`` for the subset of state queries used by
older training loops. Prefer ``model_parallel_groups`` in new code.
"""

from __future__ import annotations

from worldfoundry.core.distributed.model_parallel.megatron_compat import parallel_state

# ──────────────────────────────────────────────────────────────────────────
# Rank-0 predicate — only this rank should write shared checkpoints / logs
# ──────────────────────────────────────────────────────────────────────────


def is_tp_cp_pp_rank0() -> bool:
    """Return true when tensor, context, and pipeline parallel ranks are all zero."""

    return (
        parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_pipeline_model_parallel_rank() == 0
        and parallel_state.get_context_parallel_rank() == 0
    )


__all__ = ["is_tp_cp_pp_rank0"]
