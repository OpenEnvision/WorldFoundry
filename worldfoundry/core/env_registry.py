"""Lightweight environment-name compatibility shared across layers.

This module depends only on the standard library. Core code must not import
:mod:`worldfoundry.runtime` merely to resolve a registered environment
variable; the runtime module re-exports this API.

:func:`getenv_registered` reads the canonical ``WORLDFOUNDRY_*`` name first.
A legacy-only ``TRAINER_*`` / ``WM_*`` value is accepted and emits at most one
:class:`DeprecationWarning` per process (the warning set is guarded by
``_LEGACY_ENV_WARNING_LOCK`` so concurrent readers do not spam).
"""

from __future__ import annotations

import os
import threading
import warnings

# Canonical WORLDFOUNDRY_* name -> deprecated TRAINER_*/WM_* name (XC-8).
# New code should read the canonical name via :func:`getenv_registered`.
LEGACY_ENV_ALIASES: dict[str, str] = {
    "WORLDFOUNDRY_TRAINER_TARGET_DEVICE": "TRAINER_TARGET_DEVICE",
    "WORLDFOUNDRY_TRAINER_USE_PRECOMPILED": "TRAINER_USE_PRECOMPILED",
    "WORLDFOUNDRY_TRAINER_CONFIG_ROOT": "TRAINER_CONFIG_ROOT",
    "WORLDFOUNDRY_TRAINER_CACHE_ROOT": "TRAINER_CACHE_ROOT",
    "WORLDFOUNDRY_TRAINER_RINGBUFFER_WARNING_INTERVAL": "TRAINER_RINGBUFFER_WARNING_INTERVAL",
    "WORLDFOUNDRY_TRAINER_NCCL_SO_PATH": "TRAINER_NCCL_SO_PATH",
    "WORLDFOUNDRY_TRAINER_TEST_DYNAMO_FULLGRAPH_CAPTURE": "TRAINER_TEST_DYNAMO_FULLGRAPH_CAPTURE",
    "WORLDFOUNDRY_TRAINER_ENGINE_ITERATION_TIMEOUT_S": "TRAINER_ENGINE_ITERATION_TIMEOUT_S",
    "WORLDFOUNDRY_TRAINER_CONFIGURE_LOGGING": "TRAINER_CONFIGURE_LOGGING",
    "WORLDFOUNDRY_TRAINER_LOGGING_CONFIG_PATH": "TRAINER_LOGGING_CONFIG_PATH",
    "WORLDFOUNDRY_TRAINER_LOGGING_LEVEL": "TRAINER_LOGGING_LEVEL",
    "WORLDFOUNDRY_TRAINER_LOGGING_PREFIX": "TRAINER_LOGGING_PREFIX",
    "WORLDFOUNDRY_TRAINER_TRACE_FUNCTION": "TRAINER_TRACE_FUNCTION",
    "WORLDFOUNDRY_TRAINER_WORKER_MULTIPROC_METHOD": "TRAINER_WORKER_MULTIPROC_METHOD",
    "WORLDFOUNDRY_TRAINER_TORCH_PROFILER_DIR": "TRAINER_TORCH_PROFILER_DIR",
    "WORLDFOUNDRY_TRAINER_SERVER_DEV_MODE": "TRAINER_SERVER_DEV_MODE",
    "WORLDFOUNDRY_TRAINER_STAGE_LOGGING": "TRAINER_STAGE_LOGGING",
    "WORLDFOUNDRY_AUTO_CUDA_VISIBLE_DEVICES": "WM_AUTO_CUDA_VISIBLE_DEVICES",
    "WORLDFOUNDRY_AUTO_GPU_MAX_MEMORY_FRACTION": "WM_AUTO_GPU_MAX_MEMORY_FRACTION",
    "WORLDFOUNDRY_AUTO_GPU_MAX_UTILIZATION_FRACTION": "WM_AUTO_GPU_MAX_UTILIZATION_FRACTION",
    "WORLDFOUNDRY_MATRIXGAME3_CUDA_VISIBLE_DEVICES": "WM_MATRIXGAME3_CUDA_VISIBLE_DEVICES",
}

_LEGACY_ENV_WARNED: set[str] = set()
_LEGACY_ENV_WARNING_LOCK = threading.Lock()


def getenv_registered(canonical: str, default: str | None = None) -> str | None:
    """Read a canonical env var, falling back to its deprecated alias.

    The canonical value always wins.  A legacy-only value emits at most one
    :class:`DeprecationWarning` per process, including under concurrent reads.
    """

    value = os.environ.get(canonical)
    if value is not None:
        return value
    legacy = LEGACY_ENV_ALIASES.get(canonical)
    if legacy is None:
        return default
    legacy_value = os.environ.get(legacy)
    if legacy_value is None:
        return default

    should_warn = False
    with _LEGACY_ENV_WARNING_LOCK:
        if legacy not in _LEGACY_ENV_WARNED:
            _LEGACY_ENV_WARNED.add(legacy)
            should_warn = True
    if should_warn:
        warnings.warn(
            f"{legacy} is deprecated; use {canonical} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return legacy_value


__all__ = ["LEGACY_ENV_ALIASES", "getenv_registered"]
