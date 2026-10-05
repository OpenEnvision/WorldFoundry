"""Release process-local inference runtime caches and unused CUDA blocks.

Long-lived Studio / CLI processes keep compiled graphs, compiled
modules, and policy runtimes in ordinary Python dicts. Clearing those
dicts is not enough: cyclic references keep tensors alive, and
``torch.cuda.empty_cache`` only returns *already-unreferenced* blocks
to the caching allocator.

:func:`clear_inference_runtime_cache` therefore (1) drops the mapping,
(2) runs ``gc.collect()``, then (3) lazily imports torch and calls
``empty_cache`` when CUDA is present. The torch import is delayed so
model-discovery processes that never execute a policy do not pay for
an accelerator runtime.
"""

from __future__ import annotations

import gc
from collections.abc import MutableMapping
from typing import Any


def clear_inference_runtime_cache(cache: MutableMapping[Any, Any]) -> None:
    """Drop cached runtimes and release unreferenced accelerator allocations.

    The torch import is intentionally lazy so model discovery remains cheap in
    processes that never execute a PyTorch policy.  ``empty_cache`` only returns
    already-unreferenced blocks to the CUDA allocator; clearing the strong
    references and collecting cycles must happen first.
    """

    cache.clear()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["clear_inference_runtime_cache"]
