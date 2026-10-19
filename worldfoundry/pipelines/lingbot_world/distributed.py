"""LingBot loader and prediction defaults for an existing torchrun process group."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence


def prepare_torchrun_kwargs(
    *,
    model_id: str,
    load_kwargs: Mapping[str, Any],
    call_kwargs: Mapping[str, Any],
    world_size: int,
    minimum_gpu_vram_gib: float | None,
    sequence_parallel: bool,
    explicit_load_params: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve LingBot FSDP/offload policy without changing caller-owned options."""
    load = dict(load_kwargs)
    call = dict(call_kwargs)
    if model_id == "lingbot-world" and explicit_load_params is not None:
        # Catalog FSDP defaults select distributed launch, not a fixed loader policy.
        for key in ("t5_fsdp", "dit_fsdp"):
            if key not in explicit_load_params:
                load.pop(key, None)
    replicate_model = False
    if minimum_gpu_vram_gib is not None:
        try:
            min_gib = float(os.getenv("WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB", "72") or "72")
            replicate_model = minimum_gpu_vram_gib >= max(min_gib, 0.0)
        except ValueError:
            replicate_model = False
    # Replication avoids per-block FSDP all-gathers on 80GB-class GPUs.
    # Lower-memory and unknown devices use sharding; explicit choices win.
    load.setdefault("t5_fsdp", not replicate_model)
    load.setdefault("dit_fsdp", not replicate_model)
    load.setdefault("ulysses_size", world_size if sequence_parallel else 1)
    load.setdefault("t5_cpu", False)
    load["offload_model"] = bool(call.get("offload_model", load.get("offload_model", False)))
    if model_id in {"lingbot-world-v2", "lingbot-world-v2-1.3b"}:
        call["nproc_per_node"] = world_size
    call.setdefault("offload_model", False)
    return load, call
