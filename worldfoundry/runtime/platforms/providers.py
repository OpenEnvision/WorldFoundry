"""In-tree providers for accelerator discovery.

PyTorch is imported only inside ``detect`` helpers.  Importing this module is
therefore safe in lightweight control-plane processes and CPU-only installs.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform as host_platform
from typing import Any

from .base import BasePlatformProvider
from .types import (
    AcceleratorDescriptor,
    CapabilitySet,
    MemoryInfo,
    PlatformKind,
)


def _load_torch() -> Any | None:
    """Load torch lazily, returning ``None`` when it is not installed."""

    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _torch_compile_available(torch: Any) -> bool:
    return callable(getattr(torch, "compile", None))


def _torch_distributed_available(torch: Any) -> bool:
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return False
    is_available = getattr(distributed, "is_available", None)
    try:
        return bool(is_available()) if callable(is_available) else False
    except Exception:
        return False


def _cuda_memory(cuda: Any, index: int, properties: Any) -> MemoryInfo:
    total = _safe_int(getattr(properties, "total_memory", None))
    free: int | None = None
    mem_get_info = getattr(cuda, "mem_get_info", None)
    if callable(mem_get_info):
        try:
            free_value, runtime_total = mem_get_info(index)
            free = _safe_int(free_value)
            total = _safe_int(runtime_total) or total
        except Exception:
            # Some PyTorch versions require an initialized device context.
            pass

    def memory_stat(name: str) -> int | None:
        function = getattr(cuda, name, None)
        if not callable(function):
            return None
        try:
            return _safe_int(function(index))
        except Exception:
            return None

    return MemoryInfo(
        total_bytes=total,
        free_bytes=free,
        allocated_bytes=memory_stat("memory_allocated"),
        reserved_bytes=memory_stat("memory_reserved"),
    )


def _cuda_dtypes(cuda: Any, index: int) -> tuple[str, ...]:
    dtypes = ["float32", "float16"]
    is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        try:
            if is_bf16_supported(including_emulation=False):
                dtypes.append("bfloat16")
        except TypeError:
            try:
                if is_bf16_supported():
                    dtypes.append("bfloat16")
            except Exception:
                pass
        except Exception:
            pass
    return tuple(dtypes)


def _cuda_compute_capability(cuda: Any, index: int) -> tuple[int, int] | None:
    getter = getattr(cuda, "get_device_capability", None)
    if not callable(getter):
        return None
    try:
        major, minor = getter(index)
        return int(major), int(minor)
    except Exception:
        return None


class CudaPlatformProvider(BasePlatformProvider):
    kind = PlatformKind.CUDA

    def detect(self) -> list[AcceleratorDescriptor]:
        torch = _load_torch()
        if torch is None or getattr(getattr(torch, "version", None), "hip", None):
            return []
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return []

        devices: list[AcceleratorDescriptor] = []
        for index in range(cuda.device_count()):
            properties = cuda.get_device_properties(index)
            compute_capability = _cuda_compute_capability(cuda, index)
            arch = f"sm_{compute_capability[0]}{compute_capability[1]}" if compute_capability is not None else "unknown"
            features = {"cuda"}
            if compute_capability is not None and compute_capability[0] >= 8:
                features.update(("tensor_cores", "tf32"))
            devices.append(
                AcceleratorDescriptor(
                    id=f"cuda:{index}",
                    platform=self.kind,
                    vendor="nvidia",
                    name=str(getattr(properties, "name", f"CUDA device {index}")),
                    arch=arch,
                    index=index,
                    memory=_cuda_memory(cuda, index, properties),
                    capabilities=CapabilitySet(
                        dtypes=_cuda_dtypes(cuda, index),
                        compute_capability=compute_capability,
                        supports_compile=_torch_compile_available(torch),
                        supports_graphs=hasattr(cuda, "CUDAGraph"),
                        supports_async_copy=(compute_capability is not None and compute_capability[0] >= 8),
                        supports_distributed=_torch_distributed_available(torch),
                        features=frozenset(features),
                    ),
                    metadata={
                        "runtime": "cuda",
                        "runtime_version": getattr(getattr(torch, "version", None), "cuda", None),
                    },
                )
            )
        return devices


class RocmPlatformProvider(BasePlatformProvider):
    kind = PlatformKind.ROCM

    def detect(self) -> list[AcceleratorDescriptor]:
        torch = _load_torch()
        hip_version = getattr(getattr(torch, "version", None), "hip", None) if torch else None
        if torch is None or not hip_version:
            return []
        # PyTorch intentionally exposes ROCm devices through torch.cuda.
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return []

        devices: list[AcceleratorDescriptor] = []
        for index in range(cuda.device_count()):
            properties = cuda.get_device_properties(index)
            raw_arch = str(getattr(properties, "gcnArchName", "unknown"))
            arch = raw_arch.split(":", 1)[0] or "unknown"
            devices.append(
                AcceleratorDescriptor(
                    id=f"rocm:{index}",
                    platform=self.kind,
                    vendor="amd",
                    name=str(getattr(properties, "name", f"ROCm device {index}")),
                    arch=arch,
                    index=index,
                    memory=_cuda_memory(cuda, index, properties),
                    capabilities=CapabilitySet(
                        dtypes=_cuda_dtypes(cuda, index),
                        supports_compile=_torch_compile_available(torch),
                        supports_graphs=hasattr(cuda, "CUDAGraph"),
                        supports_async_copy=False,
                        supports_distributed=_torch_distributed_available(torch),
                        features=frozenset(("hip", "rocm")),
                    ),
                    metadata={
                        "runtime": "rocm",
                        "runtime_version": str(hip_version),
                        "gcn_arch_name": raw_arch,
                    },
                )
            )
        return devices


class XpuPlatformProvider(BasePlatformProvider):
    kind = PlatformKind.XPU

    def detect(self) -> list[AcceleratorDescriptor]:
        torch = _load_torch()
        xpu = getattr(torch, "xpu", None) if torch is not None else None
        if xpu is None or not xpu.is_available():
            return []

        devices: list[AcceleratorDescriptor] = []
        for index in range(xpu.device_count()):
            properties = xpu.get_device_properties(index)
            total = _safe_int(getattr(properties, "total_memory", None))
            free: int | None = None
            mem_get_info = getattr(xpu, "mem_get_info", None)
            if callable(mem_get_info):
                try:
                    free_value, runtime_total = mem_get_info(index)
                    free = _safe_int(free_value)
                    total = _safe_int(runtime_total) or total
                except Exception:
                    pass
            arch_value = getattr(properties, "architecture", None)
            if arch_value is None:
                arch_value = getattr(properties, "gpu_eu_count", "unknown")
            devices.append(
                AcceleratorDescriptor(
                    id=f"xpu:{index}",
                    platform=self.kind,
                    vendor="intel",
                    name=str(getattr(properties, "name", f"XPU device {index}")),
                    arch=str(arch_value),
                    index=index,
                    memory=MemoryInfo(total_bytes=total, free_bytes=free),
                    capabilities=CapabilitySet(
                        dtypes=("float32", "float16", "bfloat16"),
                        supports_compile=_torch_compile_available(torch),
                        supports_graphs=False,
                        supports_async_copy=False,
                        supports_distributed=_torch_distributed_available(torch),
                        features=frozenset(("xpu",)),
                    ),
                    metadata={"runtime": "xpu"},
                )
            )
        return devices


class MpsPlatformProvider(BasePlatformProvider):
    kind = PlatformKind.MPS

    def detect(self) -> list[AcceleratorDescriptor]:
        torch = _load_torch()
        backends = getattr(torch, "backends", None) if torch is not None else None
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        if mps_backend is None or not mps_backend.is_available():
            return []

        mps = getattr(torch, "mps", None)
        total: int | None = None
        allocated: int | None = None
        if mps is not None:
            recommended = getattr(mps, "recommended_max_memory", None)
            current = getattr(mps, "current_allocated_memory", None)
            with contextlib.suppress(Exception):
                total = _safe_int(recommended()) if callable(recommended) else None
            with contextlib.suppress(Exception):
                allocated = _safe_int(current()) if callable(current) else None
        free = max(0, total - allocated) if total is not None and allocated is not None else None
        return [
            AcceleratorDescriptor(
                id="mps:0",
                platform=self.kind,
                vendor="apple",
                name="Apple Metal Performance Shaders",
                arch=host_platform.machine() or "unknown",
                index=0,
                memory=MemoryInfo(
                    total_bytes=total,
                    free_bytes=free,
                    allocated_bytes=allocated,
                ),
                capabilities=CapabilitySet(
                    dtypes=("float32", "float16"),
                    supports_compile=_torch_compile_available(torch),
                    features=frozenset(("metal", "unified_memory")),
                ),
                metadata={"runtime": "mps"},
            )
        ]


def _host_memory() -> MemoryInfo:
    total: int | None = None
    free: int | None = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(page_size * os.sysconf("SC_PHYS_PAGES"))
        available_pages = os.sysconf_names.get("SC_AVPHYS_PAGES")
        if available_pages is not None:
            free = int(page_size * os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return MemoryInfo(total_bytes=total, free_bytes=free)


class CpuPlatformProvider(BasePlatformProvider):
    kind = PlatformKind.CPU

    def detect(self) -> list[AcceleratorDescriptor]:
        machine = host_platform.machine() or "unknown"
        processor = host_platform.processor()
        return [
            AcceleratorDescriptor(
                id="cpu:0",
                platform=self.kind,
                vendor="generic",
                name=processor or f"{machine} CPU",
                arch=machine,
                index=0,
                memory=_host_memory(),
                capabilities=CapabilitySet(
                    dtypes=("float32", "float64"),
                    supports_distributed=True,
                    features=frozenset(("host",)),
                ),
                metadata={
                    "runtime": "python",
                    "logical_cpu_count": os.cpu_count(),
                },
            )
        ]


def builtin_accelerator_providers() -> tuple[BasePlatformProvider, ...]:
    """Return fresh in-tree accelerator providers in default probe order."""

    return (
        CudaPlatformProvider(),
        RocmPlatformProvider(),
        XpuPlatformProvider(),
        MpsPlatformProvider(),
    )
