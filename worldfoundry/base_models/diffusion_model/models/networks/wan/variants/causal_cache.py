"""Host-owned positions for managed causal KV caches.

The streaming pipeline owns cache creation and reset. Its Python positions
drive slicing and rollover without copying per-layer CUDA scalars to the host;
the original tensor counters remain mirrored for external consumers. Legacy
caches containing only tensor counters retain their original behavior.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def read_kv_cache_positions(cache: MutableMapping[str, Any]) -> tuple[int, int]:
    global_key, local_key = "_host_global_end_index", "_host_local_end_index"
    if global_key in cache or local_key in cache:
        if global_key not in cache or local_key not in cache:
            raise ValueError("managed causal cache requires both host position counters")
        global_end, local_end = cache[global_key], cache[local_key]
        if type(global_end) is not int or type(local_end) is not int:
            raise TypeError("managed causal cache host positions must be Python integers")
        return global_end, local_end
    return cache["global_end_index"].item(), cache["local_end_index"].item()


def commit_kv_cache_positions(cache: MutableMapping[str, Any], global_end: int, local_end: int) -> None:
    cache["global_end_index"].fill_(global_end)
    cache["local_end_index"].fill_(local_end)
    if "_host_global_end_index" in cache:
        cache["_host_global_end_index"] = global_end
        cache["_host_local_end_index"] = local_end
