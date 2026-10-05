# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py
"""Environment flags for sequence-parallel communicators.

Lazy env reads (NCCL, CUDA, debug). Read at first use so importing SP
does not parse the whole environment for unused backends.

Not the WorldFoundry process env registry — that lives in
:mod:`worldfoundry.core.env_registry`. Flags here keep the historical
``TRAINER_*`` names via :func:`_trainer` (``WORLDFOUNDRY_*`` first).

Public surface: attribute access on this module (``envs.LOCAL_RANK``,
``envs.TRAINER_NCCL_SO_PATH``, …) resolved by :func:`__getattr__`.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from worldfoundry.core.env_registry import getenv_registered

if TYPE_CHECKING:
    TRAINER_RINGBUFFER_WARNING_INTERVAL: int = 60
    TRAINER_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    TRAINER_CACHE_ROOT: str = os.path.expanduser("~/.cache/trainer")
    TRAINER_CONFIG_ROOT: str = os.path.expanduser("~/.config/trainer")
    TRAINER_CONFIGURE_LOGGING: int = 1
    TRAINER_LOGGING_LEVEL: str = "INFO"
    TRAINER_LOGGING_PREFIX: str = ""
    TRAINER_LOGGING_CONFIG_PATH: str | None = None
    TRAINER_TRACE_FUNCTION: int = 0
    TRAINER_WORKER_MULTIPROC_METHOD: str = "fork"
    TRAINER_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    TRAINER_SERVER_DEV_MODE: bool = False
    TRAINER_STAGE_LOGGING: bool = False


# ──────────────────────────────────────────────────────────────────────────
# XDG defaults — honor the user cache/config layout when TRAINER_* is unset
# ──────────────────────────────────────────────────────────────────────────


def get_default_cache_root() -> str:
    """Return ``XDG_CACHE_HOME`` or ``~/.cache`` for trainer cache files."""
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    """Return ``XDG_CONFIG_HOME`` or ``~/.config`` for trainer config files."""
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    """Parse an optional env string; leave ``None`` unset so ``0`` stays distinct."""
    if value is None:
        return None
    return int(value)


def _trainer(name: str, default: str | None = None) -> str | None:
    """Read ``WORLDFOUNDRY_<name>`` with deprecated ``TRAINER_*`` fallback (XC-8)."""

    return getenv_registered(f"WORLDFOUNDRY_{name}", default)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# ──────────────────────────────────────────────────────────────────────────
# Lazy env table — each value is a thunk so unused backends stay unparsed
# ──────────────────────────────────────────────────────────────────────────

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of Trainer, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "TRAINER_TARGET_DEVICE": lambda: _trainer("TRAINER_TARGET_DEVICE", "cuda") or "cuda",
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # If set, trainer will use precompiled binaries (*.so)
    "TRAINER_USE_PRECOMPILED": lambda: (
        bool(_trainer("TRAINER_USE_PRECOMPILED")) or bool(os.environ.get("TRAINER_PRECOMPILED_WHEEL_LOCATION"))
    ),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # If set, trainer will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for TRAINER configuration files
    # Defaults to `~/.config/trainer` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how trainer finds its configuration files
    # during runtime, but also affects how trainer installs its configuration
    # files during **installation**.
    "TRAINER_CONFIG_ROOT": lambda: os.path.expanduser(
        _trainer(
            "TRAINER_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "trainer"),
        )
        or os.path.join(get_default_config_root(), "trainer")
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for TRAINER cache files
    # Defaults to `~/.cache/trainer` unless `XDG_CACHE_HOME` is set
    "TRAINER_CACHE_ROOT": lambda: os.path.expanduser(
        _trainer(
            "TRAINER_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "trainer"),
        )
        or os.path.join(get_default_cache_root(), "trainer")
    ),
    # Interval in seconds to log a warning message when the ring buffer is full
    "TRAINER_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        _trainer("TRAINER_RINGBUFFER_WARNING_INTERVAL", "60") or "60"
    ),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "TRAINER_NCCL_SO_PATH": lambda: _trainer("TRAINER_NCCL_SO_PATH"),
    # when `TRAINER_NCCL_SO_PATH` is not set, trainer will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # Internal flag to enable Dynamo fullgraph capture
    "TRAINER_TEST_DYNAMO_FULLGRAPH_CAPTURE": lambda: bool(
        (_trainer("TRAINER_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") or "1") != "0"
    ),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # timeout for each iteration in the engine
    "TRAINER_ENGINE_ITERATION_TIMEOUT_S": lambda: int(
        _trainer("TRAINER_ENGINE_ITERATION_TIMEOUT_S", "60") or "60"
    ),
    # Logging configuration
    # If set to 0, trainer will not configure logging
    # If set to 1, trainer will configure logging using the default configuration
    #    or the configuration file specified by TRAINER_LOGGING_CONFIG_PATH
    "TRAINER_CONFIGURE_LOGGING": lambda: int(_trainer("TRAINER_CONFIGURE_LOGGING", "1") or "1"),
    "TRAINER_LOGGING_CONFIG_PATH": lambda: _trainer("TRAINER_LOGGING_CONFIG_PATH"),
    # this is used for configuring the default logging level
    "TRAINER_LOGGING_LEVEL": lambda: _trainer("TRAINER_LOGGING_LEVEL", "INFO") or "INFO",
    # if set, TRAINER_LOGGING_PREFIX will be prepended to all log messages
    "TRAINER_LOGGING_PREFIX": lambda: _trainer("TRAINER_LOGGING_PREFIX", "") or "",
    # Trace function calls
    # If set to 1, trainer will trace function calls
    # Useful for debugging
    "TRAINER_TRACE_FUNCTION": lambda: int(_trainer("TRAINER_TRACE_FUNCTION", "0") or "0"),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "TRAINER_WORKER_MULTIPROC_METHOD": lambda: (
        _trainer("TRAINER_WORKER_MULTIPROC_METHOD", "fork") or "fork"
    ),
    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "TRAINER_TORCH_PROFILER_DIR": lambda: (
        None
        if _trainer("TRAINER_TORCH_PROFILER_DIR") is None
        else os.path.expanduser(_trainer("TRAINER_TORCH_PROFILER_DIR") or ".")
    ),
    # If set, trainer will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "TRAINER_SERVER_DEV_MODE": lambda: bool(int(_trainer("TRAINER_SERVER_DEV_MODE", "0") or "0")),
    # If set, trainer will enable stage logging, which will print the time
    # taken for each stage
    "TRAINER_STAGE_LOGGING": lambda: bool(int(_trainer("TRAINER_STAGE_LOGGING", "0") or "0")),
}

# end-env-vars-definition


def __getattr__(name: str):
    """Resolve a trainer env flag on first attribute access, then recompute each time.

    Values are not cached: a test or launcher may rewrite ``os.environ``
    after import. Unknown names raise :class:`AttributeError` so typos
    fail loudly instead of returning ``None``.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose only the documented env-flag names to completion tools."""
    return list(environment_variables.keys())
