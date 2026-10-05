"""Causal Ulysses attention: sequence-sharded QKV plus the in-tree causal kernel.

Ulysses all-to-alls heads into full-sequence shards, then runs
local attention. This wrapper pins that local kernel to
:func:`~worldfoundry.core.attention.sequence.varlen.flash_attention` so causal video
(LingBot) keeps one mask/window contract. Generic (possibly non-causal)
Ulysses lives in :mod:`worldfoundry.core.attention.parallel.ulysses_attention`.
"""

from worldfoundry.core.attention.parallel.ulysses_attention import distributed_attention as _distributed_attention
from worldfoundry.core.attention.sequence.varlen import flash_attention


def distributed_attention(
    q,
    k,
    v,
    seq_lens,
    window_size=(-1, -1),
):
    """Run Ulysses attention with the causal LingBot attention kernel."""
    return _distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=window_size,
        attention_fn=flash_attention,
    )
