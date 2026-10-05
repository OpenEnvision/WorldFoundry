"""Low-bit KV cache storage for long-horizon causal video generation.

WorldFoundry already quantizes *weights*; in streaming generation the tensor that
actually bounds the horizon is the KV cache, which grows with every emitted
chunk. Quantizing it is training-free — the cache is only ever written once and
read back as attention keys/values — so it trades a small reconstruction error
for a large, immediate memory reduction.

Quantization is group-wise and asymmetric along the head dimension: each group of
``group_size`` channels carries its own scale and zero point. Keys benefit from
this most, since their per-channel magnitudes are strongly skewed and a single
tensor-wide scale wastes most of the code range.

Codes below 8 bits are bit-packed into ``uint8``, so ``int4`` genuinely halves
and ``int2`` genuinely quarters the stored bytes rather than merely rounding
values in a full-width tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

KVQuantDType = Literal["none", "fp8", "int8", "int4", "int2"]

# ──────────────────────────────────────────────────────────────────────────
# Config + packed codes — group-wise scales; sub-byte codes live in uint8
# ──────────────────────────────────────────────────────────────────────────

# Integer code widths. fp8 is stored in a native torch dtype, not packed codes.
_INT_BITS: dict[str, int] = {"int8": 8, "int4": 4, "int2": 2}

# float8_e4m3fn covers the KV range with more mantissa than e5m2, which matters
# more than exponent range once values are already scaled per group.
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def _scale_dtype(source_dtype: torch.dtype) -> torch.dtype:
    """Pick the storage dtype for per-group scales and zero points.

    Half-precision sources get float16 scales: same footprint as the bf16 they
    came from but three more mantissa bits, and the metadata is what limits the
    achievable ratio once codes drop to 4 or 2 bits.
    """
    if source_dtype in (torch.float16, torch.bfloat16):
        return torch.float16
    return torch.float32


@dataclass(frozen=True)
class KVQuantConfig:
    """How to store cached keys or values.

    Attributes:
        dtype: Storage format. ``"none"`` disables quantization.
        group_size: Channels sharing one scale/zero point along the last
            dimension. Must divide the head dimension. Smaller groups cost more
            metadata but track per-channel outliers better.
        symmetric: Use a zero-centred range instead of per-group min/max.
            Values tolerate this; keys usually do not.
    """

    dtype: KVQuantDType = "none"
    group_size: int = 64
    symmetric: bool = False

    def __post_init__(self) -> None:
        """Reject unknown dtypes, non-positive groups, and missing fp8 support."""
        if self.dtype not in ("none", "fp8", "int8", "int4", "int2"):
            raise ValueError(f"Unsupported KV quantization dtype {self.dtype!r}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.dtype == "fp8" and _FP8_DTYPE is None:
            raise ValueError("fp8 KV quantization requires a torch build with float8_e4m3fn")

    @property
    def enabled(self) -> bool:
        """Whether this config stores anything other than the source dtype."""
        return self.dtype != "none"

    @property
    def bits(self) -> int:
        """Stored bits per element."""
        return 8 if self.dtype in ("none", "fp8") else _INT_BITS[self.dtype]

    def compression_ratio(self, source_dtype: torch.dtype = torch.bfloat16) -> float:
        """Byte reduction versus storing in ``source_dtype``, metadata included.

        Scale and zero-point cost one value per ``group_size`` channels, which is
        why small groups buy accuracy at a real, quantifiable price.
        """
        if not self.enabled:
            return 1.0
        source_bits = torch.finfo(source_dtype).bits
        metadata_values = 1 if (self.symmetric or self.dtype == "fp8") else 2
        metadata_bits = metadata_values * torch.finfo(_scale_dtype(source_dtype)).bits / self.group_size
        return source_bits / (self.bits + metadata_bits)


@dataclass(frozen=True)
class QuantizedTensor:
    """A quantized tensor plus everything needed to restore it.

    Attributes:
        codes: Packed codes (``uint8``) or an fp8 tensor.
        scale: Per-group scale, shaped like the source with the last dimension
            replaced by the group count.
        zero_point: Per-group zero point, or ``None`` when symmetric.
        shape: Source shape.
        dtype: Source dtype, restored on dequantization.
        config: Config used to produce these codes.
    """

    codes: Tensor
    scale: Tensor
    zero_point: Tensor | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    config: KVQuantConfig

    def nbytes(self) -> int:
        """Bytes actually held, including scale and zero-point metadata."""
        total = self.codes.numel() * self.codes.element_size() + self.scale.numel() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.numel() * self.zero_point.element_size()
        return total


# ──────────────────────────────────────────────────────────────────────────
# Quantize / dequantize — write once, restore on read; never in-place
# ──────────────────────────────────────────────────────────────────────────


def _pack_codes(codes: Tensor, bits: int) -> Tensor:
    """Pack sub-byte codes along the last dimension into ``uint8``."""
    if bits == 8:
        return codes.to(torch.uint8)

    per_byte = 8 // bits
    flat = codes.reshape(*codes.shape[:-1], -1, per_byte).to(torch.uint8)
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.uint8) * bits
    return (flat << shifts).sum(dim=-1).to(torch.uint8)


def _unpack_codes(packed: Tensor, bits: int, width: int) -> Tensor:
    """Invert :func:`_pack_codes` back to ``width`` codes along the last dimension."""
    if bits == 8:
        return packed.to(torch.int64)

    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits
    expanded = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return expanded.reshape(*packed.shape[:-1], -1)[..., :width].to(torch.int64)


def quantize_kv(tensor: Tensor, config: KVQuantConfig) -> QuantizedTensor:
    """Quantize a KV tensor group-wise along its last dimension.

    Args:
        tensor: Keys or values, any shape; the last dimension is the head dim.
        config: Storage configuration.

    Returns:
        A :class:`QuantizedTensor` that :func:`dequantize_kv` restores.

    Raises:
        ValueError: If quantization is disabled or the head dimension is not a
            multiple of ``group_size``.
    """
    if not config.enabled:
        raise ValueError("quantize_kv requires an enabled KVQuantConfig")

    head_dim = tensor.shape[-1]
    if head_dim % config.group_size:
        raise ValueError(f"head dim ({head_dim}) must be divisible by group_size ({config.group_size})")

    per_byte = 8 // config.bits
    if config.dtype != "fp8" and head_dim % per_byte:
        raise ValueError(f"head dim ({head_dim}) must be divisible by {per_byte} to pack {config.dtype} codes")

    scale_dtype = _scale_dtype(tensor.dtype)
    groups = head_dim // config.group_size
    grouped = tensor.reshape(*tensor.shape[:-1], groups, config.group_size).to(torch.float32)

    if config.dtype == "fp8":
        # fp8 keeps its own exponent, so one absmax scale per group is enough.
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / torch.finfo(_FP8_DTYPE).max
        codes = (grouped / scale).to(_FP8_DTYPE)
        return QuantizedTensor(
            codes=codes.reshape(tensor.shape),
            scale=scale.squeeze(-1).to(scale_dtype),
            zero_point=None,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            config=config,
        )

    bits = config.bits
    levels = (1 << bits) - 1

    if config.symmetric:
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / (levels / 2)
        zero_point = None
        codes = torch.round(grouped / scale + (levels + 1) / 2).clamp_(0, levels)
    else:
        minimum = grouped.amin(dim=-1, keepdim=True)
        maximum = grouped.amax(dim=-1, keepdim=True)
        scale = ((maximum - minimum) / levels).clamp_min(1e-12)
        zero_point = minimum
        codes = torch.round((grouped - minimum) / scale).clamp_(0, levels)

    packed = _pack_codes(codes.to(torch.uint8).reshape(*tensor.shape[:-1], head_dim), bits)
    return QuantizedTensor(
        codes=packed,
        scale=scale.squeeze(-1).to(scale_dtype),
        zero_point=None if zero_point is None else zero_point.squeeze(-1).to(scale_dtype),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        config=config,
    )


def dequantize_kv(quantized: QuantizedTensor) -> Tensor:
    """Restore a tensor produced by :func:`quantize_kv`."""
    config = quantized.config
    head_dim = quantized.shape[-1]
    groups = head_dim // config.group_size

    if config.dtype == "fp8":
        grouped = quantized.codes.to(torch.float32).reshape(*quantized.shape[:-1], groups, config.group_size)
        restored = grouped * quantized.scale.unsqueeze(-1).to(torch.float32)
        return restored.reshape(quantized.shape).to(quantized.dtype)

    bits = config.bits
    levels = (1 << bits) - 1
    codes = _unpack_codes(quantized.codes, bits, head_dim)
    grouped = codes.reshape(*quantized.shape[:-1], groups, config.group_size).to(torch.float32)
    scale = quantized.scale.unsqueeze(-1).to(torch.float32)

    if quantized.zero_point is None:
        restored = (grouped - (levels + 1) / 2) * scale
    else:
        restored = grouped * scale + quantized.zero_point.unsqueeze(-1).to(torch.float32)
    return restored.reshape(quantized.shape).to(quantized.dtype)


# ──────────────────────────────────────────────────────────────────────────
# Append-only store — keep chunks compressed until attention materializes
# ──────────────────────────────────────────────────────────────────────────


class QuantizedKVStore:
    """Append-only KV storage that quantizes on write and restores on read.

    Chunks are held quantized and concatenated only when attention asks for them,
    so peak memory tracks the compressed cache plus one materialized copy rather
    than a full-precision buffer.

    Keys and values take separate configs on purpose: keys carry the skewed
    per-channel distribution that needs asymmetric ranges, while values tolerate
    coarser settings.
    """

    def __init__(self, key_config: KVQuantConfig, value_config: KVQuantConfig | None = None, *, seq_dim: int = 1) -> None:
        """Configure per-tensor quantization and the sequence dimension.

        Args:
            key_config: Storage configuration for keys.
            value_config: Storage configuration for values; defaults to ``key_config``.
            seq_dim: Dimension chunks are concatenated along.
        """
        self.key_config = key_config
        self.value_config = value_config if value_config is not None else key_config
        self.seq_dim = seq_dim
        self._keys: list[QuantizedTensor | Tensor] = []
        self._values: list[QuantizedTensor | Tensor] = []

    @property
    def length(self) -> int:
        """Cached tokens along the sequence dimension."""
        return sum(
            (item.shape[self.seq_dim] if isinstance(item, QuantizedTensor) else item.shape[self.seq_dim])
            for item in self._keys
        )

    def nbytes(self) -> int:
        """Bytes held across keys and values."""
        total = 0
        for item in (*self._keys, *self._values):
            total += item.nbytes() if isinstance(item, QuantizedTensor) else item.numel() * item.element_size()
        return total

    @staticmethod
    def _store(tensor: Tensor, config: KVQuantConfig) -> QuantizedTensor | Tensor:
        """Quantize when enabled; otherwise clone a detached dense copy."""
        return quantize_kv(tensor.detach(), config) if config.enabled else tensor.detach().clone()

    @staticmethod
    def _restore(item: QuantizedTensor | Tensor) -> Tensor:
        """Dequantize a :class:`QuantizedTensor`; pass dense tensors through."""
        return dequantize_kv(item) if isinstance(item, QuantizedTensor) else item

    def append(self, keys: Tensor, values: Tensor) -> None:
        """Store one chunk of keys and values."""
        self._keys.append(self._store(keys, self.key_config))
        self._values.append(self._store(values, self.value_config))

    def materialize(self) -> tuple[Tensor, Tensor]:
        """Return the full cache as dense tensors.

        Raises:
            RuntimeError: If nothing has been appended yet.
        """
        if not self._keys:
            raise RuntimeError("QuantizedKVStore is empty; append a chunk before materializing")
        keys = torch.cat([self._restore(item) for item in self._keys], dim=self.seq_dim)
        values = torch.cat([self._restore(item) for item in self._values], dim=self.seq_dim)
        return keys, values

    def compact(self, keep_indices: Tensor) -> None:
        """Keep only ``keep_indices`` along the sequence dimension.

        The store is re-quantized from the surviving tokens, which keeps a single
        chunk afterwards and lets group statistics adapt to what remains.
        """
        keys, values = self.materialize()
        index = keep_indices.to(device=keys.device, dtype=torch.long)
        keys = keys.index_select(self.seq_dim, index)
        values = values.index_select(self.seq_dim, index)
        self._keys = [self._store(keys, self.key_config)]
        self._values = [self._store(values, self.value_config)]

    def reset(self) -> None:
        """Drop all cached chunks."""
        self._keys.clear()
        self._values.clear()


__all__ = [
    "KVQuantConfig",
    "KVQuantDType",
    "QuantizedKVStore",
    "QuantizedTensor",
    "dequantize_kv",
    "quantize_kv",
]
