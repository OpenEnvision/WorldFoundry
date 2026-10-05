"""Model-independent runtime placement and optimization policy values.

Policies live in core so diffusion, autoregressive, perception, and other
model families can use the same loading and execution vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import torch

# ──────────────────────────────────────────────────────────────────────────
# Vocabulary enums — string values are the on-the-wire / YAML spellings
# ──────────────────────────────────────────────────────────────────────────


class OffloadMode(str, Enum):
    """Where inactive weights live: resident, whole-component, per-block, or disk."""

    NONE = "none"
    COMPONENT = "component"
    BLOCK = "block"
    DISK = "disk"


class QuantizationMode(str, Enum):
    """Weight-level precision transform applied at load time, not per request."""

    NONE = "none"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "nvfp4"
    GGUF = "gguf"


class AttentionBackend(str, Enum):
    """Request-scoped attention provider names understood by the dispatcher.

    ``_missing_`` accepts public aliases (``flash_attn``, ``sage3``, …) so
    YAML and env strings do not have to match the enum member spelling.
    """

    AUTO = "auto"
    TORCH = "torch"
    SDPA = "sdpa"
    FLASH = "flash"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SAGE = "sage"
    SAGE_ATTENTION_3 = "sage_attention_3"
    XFORMERS = "xformers"
    SOL_ATTN = "sol_attn"

    @classmethod
    def _missing_(cls, value: object) -> "AttentionBackend | None":
        """Accept the public aliases understood by the attention dispatcher."""

        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "sol": cls.SOL_ATTN,
            "sol_attention": cls.SOL_ATTN,
            "default": cls.AUTO,
            "torch_sdpa": cls.TORCH,
            "math": cls.TORCH,
            "flash_attention": cls.FLASH,
            "flash_attn": cls.FLASH,
            "flash2": cls.FLASH_ATTENTION_2,
            "flash_attn_2": cls.FLASH_ATTENTION_2,
            "flash_attn2": cls.FLASH_ATTENTION_2,
            "flash3": cls.FLASH_ATTENTION_3,
            "flash_attn_3": cls.FLASH_ATTENTION_3,
            "flash_attn3": cls.FLASH_ATTENTION_3,
            "flash4": cls.FLASH_ATTENTION_4,
            "flash_attn_4": cls.FLASH_ATTENTION_4,
            "flash_attn4": cls.FLASH_ATTENTION_4,
            "sage_attention": cls.SAGE,
            "sage_attn": cls.SAGE,
            "sageattn": cls.SAGE,
            "sage3": cls.SAGE_ATTENTION_3,
            "sage_attn_3": cls.SAGE_ATTENTION_3,
            "xformers_attention": cls.XFORMERS,
        }
        return aliases.get(normalized)


# ──────────────────────────────────────────────────────────────────────────
# Frozen policies — coerce enums, freeze options, refuse disk+cpu target
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OffloadPolicy:
    """Offload mode plus target device; ``DISK`` requires a non-CPU target."""

    mode: OffloadMode = OffloadMode.NONE
    target: str = "cpu"
    pin_memory: bool = False
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ``mode``, freeze ``options``; disk + ``target="cpu"`` is invalid."""
        object.__setattr__(self, "mode", OffloadMode(self.mode))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if self.mode is OffloadMode.DISK and self.target == "cpu":
            raise ValueError("disk offload requires a concrete disk target")


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """Load-time weight transform; ``exclude`` names skip matching children."""

    mode: QuantizationMode = QuantizationMode.NONE
    compute_dtype: torch.dtype | None = None
    exclude: tuple[str, ...] = ()
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ``mode``, tuple-ize ``exclude``, and freeze ``options``."""
        object.__setattr__(self, "mode", QuantizationMode(self.mode))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """One model-independent placement and optimization policy."""

    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32
    attention: AttentionBackend = AttentionBackend.AUTO
    offload: OffloadPolicy = OffloadPolicy()
    quantization: QuantizationPolicy = QuantizationPolicy()
    compile: bool = False
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize device/attention and refuse a non-floating ``dtype``."""
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "attention", AttentionBackend(self.attention))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if not self.dtype.is_floating_point:
            raise ValueError("runtime dtype must be floating point")


__all__ = [
    "AttentionBackend",
    "OffloadMode",
    "OffloadPolicy",
    "QuantizationMode",
    "QuantizationPolicy",
    "RuntimePolicy",
]
