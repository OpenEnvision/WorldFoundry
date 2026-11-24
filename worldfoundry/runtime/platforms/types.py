"""Platform-neutral accelerator descriptions.

This module deliberately contains no imports from torch or vendor runtimes.  The
descriptors are safe to use in configuration, scheduling, and diagnostic code
even when WorldFoundry is installed without an accelerator-enabled PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class PlatformKind(str, Enum):
    """Execution platforms understood by the in-tree runtime."""

    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    MPS = "mps"
    CPU = "cpu"

    @classmethod
    def parse(cls, value: PlatformKind | str) -> PlatformKind:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Portable capabilities used to build an execution plan.

    ``features`` is intentionally open-ended.  New hardware features can be
    advertised without introducing a closed enumeration of GPU products.
    """

    dtypes: tuple[str, ...] = ("float32",)
    compute_capability: tuple[int, int] | None = None
    supports_compile: bool = False
    supports_graphs: bool = False
    supports_async_copy: bool = False
    supports_distributed: bool = False
    features: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtypes": list(self.dtypes),
            "compute_capability": (list(self.compute_capability) if self.compute_capability is not None else None),
            "supports_compile": self.supports_compile,
            "supports_graphs": self.supports_graphs,
            "supports_async_copy": self.supports_async_copy,
            "supports_distributed": self.supports_distributed,
            "features": sorted(self.features),
        }


@dataclass(frozen=True, slots=True)
class MemoryInfo:
    """A point-in-time memory snapshot in bytes."""

    total_bytes: int | None = None
    free_bytes: int | None = None
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "total_bytes",
            "free_bytes",
            "allocated_bytes",
            "reserved_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class AcceleratorDescriptor:
    """A serializable accelerator identity and capability snapshot."""

    id: str
    platform: PlatformKind
    vendor: str
    name: str
    arch: str
    index: int | None
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Prevent a caller from mutating scheduling metadata behind a frozen
        # descriptor.  Values are expected to be JSON-compatible primitives.
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform.value,
            "vendor": self.vendor,
            "name": self.name,
            "arch": self.arch,
            "index": self.index,
            "memory": self.memory.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "metadata": dict(self.metadata),
        }
