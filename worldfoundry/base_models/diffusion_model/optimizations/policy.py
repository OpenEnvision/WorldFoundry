"""Diffusion-facing exports of WorldFoundry core runtime policies.

:class:`RuntimePolicy` fields used by the diffusion loader:

- ``device`` / ``dtype``: placement of constructed modules.
- ``attention``: backend hint (``auto`` / ``sdpa`` / ``flash`` / …).
- ``offload``: ``none`` / ``block`` / ``component`` / ``disk``.
- ``quantization``: ``none`` / ``fp8`` / ``int8`` / …; incompatible with
  disk offload at load time.
- ``compile``: compile the bound ``forward`` (not the whole module type).
- ``options``: opt-in flags such as ``cuda_graph``, ``teacache``,
  ``approximate_attention``, ``fuse_qkv``, ``inplace_residual``,
  ``static_cross_kv``, ``rms_norm_precision``, ``device_map``.

Parsers below fail with :exc:`ValueError` on unknown spellings so a typo
cannot silently fall back to a default policy.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.core.model_loading.policy import (
    AttentionBackend,
    OffloadMode,
    OffloadPolicy,
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
)


def parse_attention_backend(value: object, *, owner: str = "diffusion model") -> AttentionBackend:
    """Normalize a public attention provider without mutating process globals."""

    if isinstance(value, AttentionBackend):
        return value
    try:
        return AttentionBackend(str(value or "auto"))
    except ValueError as error:
        allowed = ", ".join(backend.value for backend in AttentionBackend)
        raise ValueError(f"unsupported {owner} attention backend: {value!r}; expected one of {allowed}") from error


def parse_quantization_policy(
    value: object,
    *,
    owner: str = "diffusion model",
) -> QuantizationPolicy:
    """Build a typed, opt-in weight-quantization policy from public options.

    A string selects a mode. A mapping additionally accepts ``compute_dtype``,
    ``exclude`` and an ``options`` mapping (for example ``min_features`` or
    ``scaling``). Unknown modes and malformed mappings fail before loading a
    checkpoint, so a requested acceleration cannot silently become dense.
    """

    if isinstance(value, QuantizationPolicy):
        return value
    if value is None or value is False:
        return QuantizationPolicy()
    if isinstance(value, Mapping):
        raw_mode = value.get("mode", "none")
        raw_compute_dtype = value.get("compute_dtype")
        compute_dtype = (
            None
            if raw_compute_dtype in (None, "", "none")
            else parse_torch_dtype(raw_compute_dtype, owner=f"{owner} quantization compute")
        )
        raw_exclude = value.get("exclude", ())
        if raw_exclude is None:
            exclude = ()
        elif isinstance(raw_exclude, str):
            exclude = (raw_exclude,)
        else:
            exclude = tuple(str(item) for item in raw_exclude)
        raw_options = value.get("options", {})
        if not isinstance(raw_options, Mapping):
            raise TypeError(f"{owner} quantization options must be a mapping")
        options = dict(raw_options)
        for name in ("min_features", "scaling", "keep_dense_fallback", "use_fast_accum"):
            if name in value:
                options[name] = value[name]
    else:
        raw_mode = value
        compute_dtype = None
        exclude = ()
        options = {}
    normalized = str(raw_mode or "none").strip().lower().replace("-", "")
    aliases = {
        "none": QuantizationMode.NONE,
        "false": QuantizationMode.NONE,
        "0": QuantizationMode.NONE,
        "fp8": QuantizationMode.FP8,
        "float8": QuantizationMode.FP8,
        "int8": QuantizationMode.INT8,
        "int4": QuantizationMode.INT4,
        "nvfp4": QuantizationMode.NVFP4,
        "fp4": QuantizationMode.NVFP4,
        "gguf": QuantizationMode.GGUF,
    }
    try:
        mode = aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported {owner} quantization mode: {raw_mode!r}") from error
    return QuantizationPolicy(
        mode=mode,
        compute_dtype=compute_dtype,
        exclude=exclude,
        options=options,
    )


def parse_torch_dtype(value: object, *, owner: str = "diffusion model") -> torch.dtype:
    """Normalize public dtype spellings at the shared diffusion boundary.

    Accepts ``bf16`` / ``bfloat16``, ``fp16`` / ``float16``, ``fp32`` /
    ``float32``, with an optional ``torch.`` prefix.  ``None`` or empty
    becomes ``bfloat16``.

    Raises:
        ValueError: Any other spelling (includes the ``owner`` label).
    """

    normalized = str(value or "bfloat16").lower().removeprefix("torch.")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported {owner} dtype: {value!r}") from error


def parse_offload_policy(value: object, *, allow_disk: bool = True, owner: str = "diffusion model") -> OffloadPolicy:
    """Build the framework-owned offload policy from a public option.

    Mapping:
        ``none`` / ``resident`` / ``fast`` / ``false`` / ``0`` →
        :attr:`OffloadMode.NONE`.
        ``block`` / ``async-block`` / ``cpu`` / ``layer`` → asynchronous
        layerwise CPU offload when the model declares a layer container.
        ``component`` → whole-module CPU offload (needs ``vram_module_map``).
        ``disk`` → disk offload when ``allow_disk`` is true.

    ``None`` or empty defaults to ``block``.

    Raises:
        ValueError: Unknown mode, or ``disk`` when ``allow_disk`` is false.
    """

    normalized = str(value or "block").lower()
    if normalized in {"none", "resident", "fast", "false", "0"}:
        return OffloadPolicy()
    if normalized in {"block", "async-block", "async_block", "cpu", "layer"}:
        return OffloadPolicy(mode=OffloadMode.BLOCK, target="cpu", pin_memory=True)
    if normalized == "component":
        return OffloadPolicy(mode=OffloadMode.COMPONENT, target="cpu", pin_memory=True)
    if normalized == "disk" and allow_disk:
        return OffloadPolicy(mode=OffloadMode.DISK, target="disk")
    raise ValueError(f"unsupported {owner} offload mode: {value!r}")


def parse_device_map(value: object, *, owner: str = "diffusion model") -> str | None:
    """Normalize public device-map spellings. ``None`` means single-device placement.

    ``balanced`` and ``auto`` both become ``\"balanced\"`` (Ulysses-style
    layer sharding).  Empty / ``none`` / ``false`` / ``0`` become ``None``.

    Raises:
        ValueError: Any other spelling.
    """

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "false", "0"}:
        return None
    if normalized in {"balanced", "auto"}:
        return "balanced"
    raise ValueError(f"unsupported {owner} device_map: {value!r}")


__all__ = [
    "AttentionBackend",
    "OffloadMode",
    "OffloadPolicy",
    "QuantizationMode",
    "QuantizationPolicy",
    "RuntimePolicy",
    "parse_attention_backend",
    "parse_device_map",
    "parse_offload_policy",
    "parse_quantization_policy",
    "parse_torch_dtype",
]
