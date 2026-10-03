"""Propagate reproducibility settings through the isolated inference stages."""

from __future__ import annotations

import os
from typing import Mapping

SEED_ENV = "WORLDFOUNDRY_INSPATIO_WORLD_SEED"
DETERMINISTIC_ENV = "WORLDFOUNDRY_DETERMINISTIC"


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    return seed


def seed_for_rank(seed: int, rank: int) -> int:
    """Keep rank offsets valid for NumPy's 32-bit seed range."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    return (validate_seed(seed) + rank) % 2**32


def reproducibility_env(
    env: Mapping[str, str], seed: int | None = None, deterministic: bool = False,
) -> dict[str, str]:
    """Return child settings without changing the caller's environment or RNG."""
    result = dict(env)
    strict = deterministic or result.get(DETERMINISTIC_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if seed is None and strict and SEED_ENV not in result:
        seed = 0
    if seed is not None:
        result[SEED_ENV] = str(validate_seed(seed))
    if SEED_ENV in result:
        resolved_seed = validate_seed(int(result[SEED_ENV]))
        result["PYTHONHASHSEED"] = str(resolved_seed)
    if strict:
        result[DETERMINISTIC_ENV] = "1"
        result["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return result


def initialize_reproducibility() -> None:
    """Initialize every leaf process before model construction or geometry work."""
    env = reproducibility_env(os.environ)
    seed = env.get(SEED_ENV)
    if seed is None:
        return
    from worldfoundry.core.utils.tensors.torch import set_seed_everywhere

    set_seed_everywhere(
        int(seed), deterministic=env.get(DETERMINISTIC_ENV) == "1",
        handle_invalid_seed="raise",
    )
