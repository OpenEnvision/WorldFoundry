"""Device detection and torch backend helpers for CPU, CUDA, and NPU.

Availability is queried at call time (not cached at import) so a process that
starts before CUDA/NPU is visible still sees a live answer. Bare ``cuda``
resolves to ``torch.cuda.current_device()`` so distributed launchers that
called ``set_device`` do not collapse every local rank onto GPU 0.

``cuda:N`` in :func:`cuda_visible_devices_from_device` is a *local* index into
an inherited ``CUDA_VISIBLE_DEVICES`` list by default — the mapping expected
by nested subprocess launchers under a scheduler or torchrun. ``auto`` dtype
selects bf16 on Ampere/Hopper/Blackwell-or-newer (compute capability >= 8),
fp16 on older CUDA devices, and fp32 on CPU.
"""

import importlib
import logging
import os
import warnings
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Live availability — never cache at import; CUDA/NPU can appear later
# ──────────────────────────────────────────────────────────────────────────


def is_torch_npu_available() -> bool:
    """Return whether the optional ``torch_npu`` package is installed."""
    return importlib.util.find_spec("torch_npu") is not None


def is_cuda_available() -> bool:
    """Query CUDA availability at call time instead of caching it at import."""

    return bool(torch.cuda.is_available())


def is_npu_available() -> bool:
    """Query NPU availability at call time without changing NPU configuration."""

    return bool(is_torch_npu_available() and hasattr(torch, "npu") and torch.npu.is_available())


def setup_npu(*, allow_internal_format: bool = False) -> Any:
    """Explicitly configure and return the available NPU namespace.

    Historically importing :mod:`worldfoundry.core.execution.device` silently set
    ``allow_internal_format=False`` process-wide.  Callers that require that
    numerical/layout policy must now opt in through this function.
    """

    if not is_npu_available():
        raise RuntimeError("NPU setup was requested, but torch_npu is unavailable")
    torch.npu.config.allow_internal_format = bool(allow_internal_format)
    return torch.npu


def __getattr__(name: str) -> bool:
    """Resolve deprecated availability constants lazily for compatibility."""

    resolvers = {
        "IS_CUDA_AVAILABLE": (is_cuda_available, "is_cuda_available"),
        "IS_NPU_AVAILABLE": (is_npu_available, "is_npu_available"),
    }
    entry = resolvers.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    resolver, replacement = entry
    warnings.warn(
        f"{name} is deprecated; call {replacement}() for a live availability check.",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolver()


# ──────────────────────────────────────────────────────────────────────────
# Current device + inference resolution (bare cuda → current_device, not GPU 0)
# ──────────────────────────────────────────────────────────────────────────


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if is_cuda_available():
        device = "cuda"
    elif is_npu_available():
        device = "npu"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning("Device namespace %r not found in torch, falling back to 'torch.cuda'.", device_name)
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_device_name() -> str:
    """Get current device name based on device type."""
    return f"{get_device_type()}:{get_device_id()}"


def get_current_torch_device() -> torch.device:
    """Return the accelerator selected for this process, or CPU.

    Distributed launchers select a process-local device with ``set_device``.
    Using that current index avoids collapsing every local rank onto CUDA 0.
    """

    if is_cuda_available():
        return torch.device("cuda", torch.cuda.current_device())
    if is_npu_available():
        return torch.device("npu", torch.npu.current_device())
    return torch.device("cpu")


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def get_nccl_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if is_cuda_available():
        return "nccl"
    elif is_npu_available():
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def enable_high_precision_for_bf16() -> None:
    """Disable reduced-precision bf16 matmul accumulation on CUDA and NPU."""
    if is_cuda_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if is_npu_available():
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def parse_device_type(device) -> str:
    """Normalize a device string or :class:`torch.device` to ``cpu``, ``cuda``, or ``npu``."""
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        elif device.startswith("npu"):
            return "npu"
        else:
            return "cpu"
    elif isinstance(device, torch.device):
        return device.type
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────
# Nested-launcher mapping — cuda:N indexes inherited CUDA_VISIBLE_DEVICES
# ──────────────────────────────────────────────────────────────────────────


def cuda_visible_devices_from_device(
    device: str | torch.device | None,
    *,
    inherited: str | None = None,
    map_inherited: bool = True,
    default_cuda: str = "0",
) -> str | None:
    """Convert a device string into a ``CUDA_VISIBLE_DEVICES`` value.

    ``cuda:N`` is interpreted as a local index into an inherited
    ``CUDA_VISIBLE_DEVICES`` list by default, which is the behavior expected by
    subprocess launchers nested under a scheduler or torchrun process.
    """

    if device is None:
        return None
    normalized = str(device).strip().lower()
    if not normalized:
        return None
    inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES") if inherited is None else inherited
    if normalized == "cuda":
        return inherited_devices if inherited_devices else default_cuda
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return None
        if "," in suffix:
            return suffix if all(part.strip().isdigit() for part in suffix.split(",") if part.strip()) else None
        if not suffix.isdigit():
            return None
        visible_index = int(suffix)
        # ``cuda:N`` is an index into the inherited CUDA_VISIBLE_DEVICES list,
        # not a physical PCI bus id. Nested launchers must remap or they pin
        # every child onto the parent's first visible GPU.
        if map_inherited and inherited_devices:
            inherited_parts = [part.strip() for part in inherited_devices.split(",") if part.strip()]
            if visible_index < len(inherited_parts):
                return inherited_parts[visible_index]
        return str(visible_index)
    if normalized.isdigit() or (
        "," in normalized and all(part.strip().isdigit() for part in normalized.split(",") if part.strip())
    ):
        return normalized
    return None


def parse_nccl_backend(device_type: str) -> str:
    """Return the distributed backend name (``nccl`` or ``hccl``) for *device_type*."""
    if device_type == "cuda":
        return "nccl"
    elif device_type == "npu":
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {device_type}.")


def get_available_device_type() -> str:
    """Return the best available device type on this host."""
    return get_device_type()


def resolve_inference_device(device: str | torch.device | None = "cuda", *, allow_cpu_fallback: bool = False) -> str:
    """Resolve a concrete inference device without silently selecting the wrong GPU.

    Bare ``cuda`` resolves to the CUDA device currently selected for this
    process. Explicit indices are preserved, which is important when a caller
    deliberately selects (for example) ``cuda:4`` under an eight-GPU workspace.
    """

    requested = str(device or "cuda").strip().lower()
    if requested == "cuda":
        requested = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if allow_cpu_fallback:
            return "cpu"
        raise RuntimeError(f"CUDA device {requested!r} was requested, but CUDA is unavailable")
    if requested.startswith("cuda"):
        parsed = torch.device(requested)
        index = 0 if parsed.index is None else parsed.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is out of range for {torch.cuda.device_count()} visible device(s)"
            )
    return requested


def resolve_inference_dtype(
    device: str | torch.device,
    dtype: str | torch.dtype | None = "auto",
    *,
    strict: bool = True,
) -> torch.dtype:
    """Resolve an inference dtype using the selected accelerator's capability.

    ``auto`` selects bf16 on Ampere/Hopper/Blackwell-or-newer CUDA devices,
    fp16 on older CUDA devices, and fp32 on CPU. Explicit unsupported bf16 is
    rejected in strict mode instead of producing a later kernel failure.
    """

    if isinstance(dtype, torch.dtype):
        selected = dtype
    else:
        name = str(dtype or "auto").strip().lower()
        aliases = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "full": torch.float32,
        }
        if name != "auto" and name not in aliases:
            raise ValueError(f"unsupported inference dtype: {dtype!r}")
        selected = aliases.get(name, torch.float32)

    parsed = torch.device(device)
    # ``auto``: SM >= 8 (Ampere+) gets bf16; older CUDA gets fp16; CPU stays fp32.
    if str(dtype or "auto").strip().lower() == "auto":
        if parsed.type == "cuda":
            index = torch.cuda.current_device() if parsed.index is None else parsed.index
            major, _minor = torch.cuda.get_device_capability(index)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32

    if selected is torch.bfloat16 and parsed.type == "cuda":
        index = torch.cuda.current_device() if parsed.index is None else parsed.index
        major, _minor = torch.cuda.get_device_capability(index)
        if major < 8:
            if strict:
                raise RuntimeError(f"bfloat16 is unsupported on CUDA device {index} (compute capability {major}.x)")
            return torch.float16
    return selected


__all__ = [
    "IS_CUDA_AVAILABLE",  # noqa: F822 - evaluated on access by __getattr__
    "IS_NPU_AVAILABLE",  # noqa: F822 - evaluated on access by __getattr__
    "cuda_visible_devices_from_device",
    "empty_cache",
    "enable_high_precision_for_bf16",
    "get_available_device_type",
    "get_current_torch_device",
    "get_device_id",
    "get_device_name",
    "get_device_type",
    "get_nccl_backend",
    "get_torch_device",
    "is_cuda_available",
    "is_npu_available",
    "is_torch_npu_available",
    "parse_device_type",
    "parse_nccl_backend",
    "resolve_inference_device",
    "resolve_inference_dtype",
    "setup_npu",
    "synchronize",
]
