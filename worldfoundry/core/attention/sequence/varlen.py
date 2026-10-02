"""Variable-length / packed attention for vision-language and Wan-style batches.

Dense unpadded batches stay on one SDPA launch — packing them into varlen
is slower for the small batches diffusion uses. External FlashAttention
2/3 is tried only when ``version`` is set; otherwise jagged Flash SDPA
(when batch is large enough) or a per-sample SDPA loop.

:func:`masked_attention` packs non-prefix padding into dense prefixes so
kernels that only understand ``cu_seqlens`` still see valid tokens.
``max_seqlen_*`` avoids a GPU-to-CPU ``.item()`` sync before a varlen launch.

Not this module:
    Backend probing lives in :mod:`.backends`. Exact dense SDPA lives
    in :mod:`.native`. Packed dataclass ABI lives in
    :mod:`.packed_sequence`. This module does not import FlashAttention
    at module scope.

Public surface:

- :func:`flash_attention` — Wan-compatible dense / packed entry.
- :func:`attention` — alias with Cosmos-style keyword names.
- :func:`masked_attention` — pack non-prefix padding, then restore.
- :func:`varlen_scaled_dot_product_attention` — already-packed
  ``cu_seqlens`` path.
"""

from __future__ import annotations

import logging
import operator
import os
import warnings
from typing import Any

import torch

logger = logging.getLogger(__name__)

from worldfoundry.core.attention.backends.probe import probe_attention_backends
from worldfoundry.core.attention.backends.native import native_sdpa_priority, scaled_dot_product_attention

try:
    from torch.nn.attention.bias import causal_lower_right as _causal_lower_right
except ImportError:
    _causal_lower_right = None


# ──────────────────────────────────────────────────────────────────────────
# Public entries — dense SDPA by default; packed / FA2/3 only when asked
# ──────────────────────────────────────────────────────────────────────────


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Wan-compatible attention with an in-tree default execution path.

    ``max_seqlen_q`` and ``max_seqlen_k`` let packing callers pass maxima they
    already computed.  Supplying them avoids a GPU-to-CPU ``.item()`` sync
    immediately before an external variable-length FlashAttention launch.
    """

    half_dtypes = (torch.float16, torch.bfloat16)
    if dtype not in half_dtypes:
        raise ValueError("dtype must be float16 or bfloat16.")
    window_size = _validated_window_size(window_size)

    batch, q_len, k_len, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    if k.size(0) != batch or v.size(0) != batch or v.size(1) != k_len:
        raise ValueError("q, k and v must have matching batch and key/value sequence dimensions")

    def half(value: torch.Tensor) -> torch.Tensor:
        """Cast to the kernel dtype only when the tensor is not already half."""

        return value if value.dtype in half_dtypes else value.to(dtype)

    # The common unpadded path should remain one dense SDPA launch. Packing it
    # and looping per sample is substantially slower for the small batches used
    # by diffusion models. External FlashAttention remains an explicit
    # ``version=2/3`` choice.
    if q_lens is None and k_lens is None and version is None:
        dense_q, dense_k, dense_v = half(q), half(k), half(v)
        dense_q = dense_q.to(dense_v.dtype)
        dense_k = dense_k.to(dense_v.dtype)
        if q_scale is not None:
            dense_q = dense_q * q_scale
        dense_q = dense_q.transpose(1, 2)
        dense_k = dense_k.transpose(1, 2)
        dense_v = dense_v.transpose(1, 2)
        attn_mask = None
        use_causal_flag = causal
        if window_size != (-1, -1) or (causal and q_len != k_len):
            attn_mask = _bottom_right_window_mask(
                q_len,
                k_len,
                dense_q.device,
                window_size=window_size,
                causal=causal,
            )[None, None, :, :]
            use_causal_flag = False
        compute_fp32 = dense_q.device.type == "cpu" and dense_q.dtype in half_dtypes
        if compute_fp32:
            dense_q, dense_k, dense_v = dense_q.float(), dense_k.float(), dense_v.float()
        selected_backends = (
            ()
            if torch.compiler.is_compiling()
            else native_sdpa_priority(
                dense_q.device,
                has_mask=attn_mask is not None,
            )
        )
        output = scaled_dot_product_attention(
            dense_q,
            dense_k,
            dense_v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_causal_flag,
            scale=softmax_scale,
            enable_gqa=dense_q.shape[1] != dense_k.shape[1],
            backends=selected_backends,
        )
        if attn_mask is not None:
            output = torch.nan_to_num(output, nan=0.0)
        return output.transpose(1, 2).to(out_dtype)

    if q_lens is None:
        q_lens = torch.full((batch,), q_len, dtype=torch.int32, device=q.device)
        resolved_max_seqlen_q = q_len
    else:
        q_lens = _validated_lengths(q_lens, batch=batch, maximum=q_len, device=q.device, name="q_lens")
        resolved_max_seqlen_q = _resolve_max_seqlen(
            max_seqlen_q,
            lengths=q_lens,
            padded_length=q_len,
            batch=batch,
            name="max_seqlen_q",
        )
    q_valid = torch.arange(q_len, device=q.device).unsqueeze(0) < q_lens.unsqueeze(1)
    q = half(q[q_valid])

    if k_lens is None:
        k_lens = torch.full((batch,), k_len, dtype=torch.int32, device=k.device)
        resolved_max_seqlen_k = k_len
    else:
        k_lens = _validated_lengths(k_lens, batch=batch, maximum=k_len, device=k.device, name="k_lens")
        resolved_max_seqlen_k = _resolve_max_seqlen(
            max_seqlen_k,
            lengths=k_lens,
            padded_length=k_len,
            batch=batch,
            name="max_seqlen_k",
        )
    k_valid = torch.arange(k_len, device=k.device).unsqueeze(0) < k_lens.unsqueeze(1)
    k = half(k[k_valid])
    v = half(v[k_valid])

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True)

    flash_attn_interface = None
    flash_attn_module = None
    if version == 3:
        try:
            import flash_attn_interface
        except Exception as exc:
            logger.debug("flash_attn_interface import failed: %s", exc)
    elif version == 2:
        try:
            import flash_attn as flash_attn_module
        except Exception as exc:
            logger.debug("flash_attn import failed: %s", exc)
    capabilities = probe_attention_backends(q.device) if version in {2, 3} else {}
    use_fa3 = (
        version == 3
        and flash_attn_interface is not None
        and capabilities["flash_attention_3"].usable
        and dropout_p == 0.0
    )
    use_fa2 = (
        version == 2
        and flash_attn_module is not None
        and capabilities["flash_attention_2"].usable
    )
    if version == 3 and not use_fa3:
        warnings.warn(
            "FlashAttention 3 is unavailable on this GPU/runtime; falling back to PyTorch SDPA."
        )
    if version == 2 and not use_fa2:
        warnings.warn(
            "FlashAttention 2 is unavailable on this GPU/runtime; falling back to PyTorch SDPA."
        )

    if use_fa3:
        output = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=resolved_max_seqlen_q,
            max_seqlen_k=resolved_max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        if isinstance(output, tuple):
            output = output[0]
    elif use_fa2:
        output = flash_attn_module.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=resolved_max_seqlen_q,
            max_seqlen_k=resolved_max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
    else:
        output = _varlen_attention_torch(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            max_seqlen_q=resolved_max_seqlen_q,
            max_seqlen_k=resolved_max_seqlen_k,
        )

    padded = output.new_zeros((batch, q_len, *output.shape[1:]))
    padded[q_valid] = output
    return padded.to(out_dtype)


def attention(
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version: int | None = None,
    version: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    *,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    is_causal: bool | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for Wan- and Cosmos-style attention call signatures."""

    q = query if q is None else q
    k = key if k is None else k
    v = value if v is None else v
    if q is None or k is None or v is None:
        raise TypeError("attention requires query/key/value tensors")
    if is_causal is not None:
        causal = is_causal

    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version if fa_version is not None else version,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
) -> torch.Tensor:
    """Apply attention to padded sequences and restore the query layout.

    Masks use ``True`` for valid tokens and may contain non-prefix padding. This
    adapter is shared by model families whose tensors are padded while optimized
    attention kernels consume packed sequences.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, sequence, heads, head_dim]")
    batch, query_length = query.shape[:2]
    key_length = key.shape[1]
    if key.shape[0] != batch or value.shape[:2] != (batch, key_length):
        raise ValueError("query, key, and value must have matching batch and key/value dimensions")

    query_mask = _validated_padding_mask(
        query_mask,
        batch=batch,
        length=query_length,
        device=query.device,
        name="query_mask",
    )
    key_mask = _validated_padding_mask(
        key_mask,
        batch=batch,
        length=key_length,
        device=key.device,
        name="key_mask",
    )
    if query_mask is None and key_mask is None:
        return flash_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )
    if query_mask is None:
        query_mask = torch.ones((batch, query_length), dtype=torch.bool, device=query.device)
    if key_mask is None:
        key_mask = torch.ones((batch, key_length), dtype=torch.bool, device=key.device)

    query_lengths = query_mask.sum(dim=1, dtype=torch.int32)
    key_lengths = key_mask.sum(dim=1, dtype=torch.int32)
    max_query_length = int(query_lengths.max().item()) if batch else 0
    max_key_length = int(key_lengths.max().item()) if batch else 0
    packed_query = query.new_zeros((batch, max_query_length, *query.shape[2:]))
    packed_key = key.new_zeros((batch, max_key_length, *key.shape[2:]))
    packed_value = value.new_zeros((batch, max_key_length, *value.shape[2:]))
    for index in range(batch):
        query_count = int(query_lengths[index].item())
        key_count = int(key_lengths[index].item())
        packed_query[index, :query_count] = query[index, query_mask[index]]
        packed_key[index, :key_count] = key[index, key_mask[index]]
        packed_value[index, :key_count] = value[index, key_mask[index]]

    packed_output = flash_attention(
        packed_query,
        packed_key,
        packed_value,
        q_lens=query_lengths,
        k_lens=key_lengths,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
        dtype=dtype,
        version=version,
        max_seqlen_q=max_query_length,
        max_seqlen_k=max_key_length,
    )
    output = query.new_zeros(query.shape)
    for index in range(batch):
        query_count = int(query_lengths[index].item())
        output[index, query_mask[index]] = packed_output[index, :query_count]
    return output


# ──────────────────────────────────────────────────────────────────────────
# Already-packed path — cu_seqlens in, no pad/unpad; FA2/3 still explicit
# ──────────────────────────────────────────────────────────────────────────


def varlen_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    version: int | None = None,
    window_size: tuple[int, int] = (-1, -1),
    **kwargs: Any,
) -> torch.Tensor:
    """Run in-tree packed attention, or external FlashAttention 2 explicitly."""

    window_size = _validated_window_size(window_size)
    flash_attn_varlen_func = None
    flash_attn_varlen_func_v3 = None
    if version == 2:
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception as exc:
            logger.debug("flash_attn import failed: %s", exc)
    elif version == 3:
        try:
            from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
        except Exception as exc:
            logger.debug("flash_attn_interface import failed: %s", exc)
    capabilities = probe_attention_backends(query.device) if version in {2, 3} else {}
    if version == 3 and (
        flash_attn_varlen_func_v3 is None or not capabilities["flash_attention_3"].usable or dropout_p != 0.0
    ):
        warnings.warn("FlashAttention 3 is unavailable on this GPU/runtime; falling back to PyTorch SDPA.")
    if version == 2 and (flash_attn_varlen_func is None or not capabilities["flash_attention_2"].usable):
        warnings.warn("FlashAttention 2 is unavailable on this GPU/runtime; falling back to PyTorch SDPA.")
    if (
        flash_attn_varlen_func_v3 is not None
        and capabilities["flash_attention_3"].usable
        and dropout_p == 0.0
    ):
        resolved_max_q = _max_from_cumulative(cu_seqlens_q, max_seqlen_q, name="max_seqlen_q")
        resolved_max_k = _max_from_cumulative(cu_seqlens_k, max_seqlen_k, name="max_seqlen_k")
        output = flash_attn_varlen_func_v3(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=resolved_max_q,
            max_seqlen_k=resolved_max_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=bool(kwargs.pop("deterministic", False)),
        )
        return output[0] if isinstance(output, tuple) else output
    if (
        flash_attn_varlen_func is not None
        and capabilities["flash_attention_2"].usable
    ):
        resolved_max_q = _max_from_cumulative(cu_seqlens_q, max_seqlen_q, name="max_seqlen_q")
        resolved_max_k = _max_from_cumulative(cu_seqlens_k, max_seqlen_k, name="max_seqlen_k")
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=resolved_max_q,
            max_seqlen_k=resolved_max_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs,
        )
    return _varlen_attention_torch(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def _varlen_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Exact packed attention: jagged Flash SDPA, else a per-sample SDPA loop.

    The loop is the portable fallback for small batches, windowed
    attention, GQA, and CPU. CPU half tensors are promoted to fp32
    because math SDPA is unstable in fp16 on CPU. Rectangular causal
    windows that produce all-masked leading rows are ``nan_to_num``'d
    so backends that return NaN stay bit-compatible with math.
    """

    jagged_output = _varlen_attention_jagged_flash(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    if jagged_output is not None:
        return jagged_output

    q_offsets = _offsets(cu_seqlens_q)
    k_offsets = _offsets(cu_seqlens_k)
    outputs: list[torch.Tensor] = []
    compute_in_fp32 = query.device.type == "cpu" and query.dtype in {torch.float16, torch.bfloat16}

    for index in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[index], q_offsets[index + 1]
        k_start, k_end = k_offsets[index], k_offsets[index + 1]
        query_states = query[q_start:q_end]
        key_states = key[k_start:k_end]
        value_states = value[k_start:k_end]

        if compute_in_fp32:
            query_states = query_states.float()
            key_states = key_states.float()
            value_states = value_states.float()

        attn_mask = None
        if window_size != (-1, -1):
            attn_mask = _bottom_right_window_mask(
                int(query_states.shape[0]),
                int(key_states.shape[0]),
                query_states.device,
                window_size=window_size,
                causal=causal,
            )[None, None, :, :]
        elif causal:
            q_length = int(query_states.shape[0])
            k_length = int(key_states.shape[0])
            if _causal_lower_right is not None and query_states.device.type == "cuda":
                attn_mask = _causal_lower_right(q_length, k_length)
            else:
                attn_mask = _bottom_right_causal_mask(q_length, k_length, query_states.device)[None, None, :, :]

        sdpa_kwargs: dict[str, Any] = {}
        if query_states.device.type == "cpu":
            sdpa_kwargs["backend"] = "math"
        else:
            sdpa_kwargs["backends"] = native_sdpa_priority(
                query_states.device,
                has_mask=attn_mask is not None,
            )

        attn_output = scaled_dot_product_attention(
            query_states.transpose(0, 1).unsqueeze(0).contiguous(),
            key_states.transpose(0, 1).unsqueeze(0).contiguous(),
            value_states.transpose(0, 1).unsqueeze(0).contiguous(),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=softmax_scale,
            enable_gqa=query_states.shape[1] != key_states.shape[1],
            **sdpa_kwargs,
        )
        if causal and query_states.shape[0] > key_states.shape[0]:
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
        outputs.append(attn_output.squeeze(0).transpose(0, 1).to(dtype=query.dtype))

    if not outputs:
        return query.new_empty((0, query.shape[1], query.shape[2]))
    return torch.cat(outputs, dim=0)


def _varlen_attention_jagged_flash(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
) -> torch.Tensor | None:
    """Use PyTorch's packed jagged Flash SDPA without an external extension."""

    constructor = getattr(getattr(torch, "nested", None), "nested_tensor_from_jagged", None)
    batch = int(cu_seqlens_q.numel()) - 1
    try:
        min_batch = max(int(os.getenv("WORLDFOUNDRY_VARLEN_JAGGED_MIN_BATCH", "32")), 1)
    except ValueError:
        min_batch = 32
    if torch.compiler.is_compiling():
        min_batch = 1
    if (
        not callable(constructor)
        or batch < min_batch
        or query.device.type != "cuda"
        or query.dtype not in {torch.float16, torch.bfloat16}
        or window_size != (-1, -1)
        or query.shape[1] != key.shape[1]
        or key.shape[1] != value.shape[1]
    ):
        return None
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if torch.compiler.is_compiling():
        if max_seqlen_q is None or max_seqlen_k is None:
            return None
    elif (
        q_lengths.numel() == 0
        or int(q_lengths.min().item()) <= 0
        or int(k_lengths.min().item()) <= 0
    ):
        return None
    q_max = int(max_seqlen_q) if max_seqlen_q is not None else int(q_lengths.max().item())
    k_max = int(max_seqlen_k) if max_seqlen_k is not None else int(k_lengths.max().item())
    try:
        q_nested = constructor(
            query,
            offsets=cu_seqlens_q,
            min_seqlen=1,
            max_seqlen=q_max,
        ).transpose(1, 2)
        k_nested = constructor(
            key,
            offsets=cu_seqlens_k,
            min_seqlen=1,
            max_seqlen=k_max,
        ).transpose(1, 2)
        v_nested = constructor(
            value,
            offsets=cu_seqlens_k,
            min_seqlen=1,
            max_seqlen=k_max,
        ).transpose(1, 2)
        output = scaled_dot_product_attention(
            q_nested,
            k_nested,
            v_nested,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
            backend="flash",
        )
        packed = output.values().transpose(0, 1)
        return torch.nan_to_num(packed, nan=0.0) if causal else packed
    except RuntimeError as exc:
        message = str(exc).casefold()
        if "out of memory" in message or "alloc_failed" in message:
            raise
        # Older PyTorch builds and unsupported head dimensions fall through to
        # the exact per-sequence SDPA loop.
        return None


# ──────────────────────────────────────────────────────────────────────────
# Packed ABI helpers — keep host syncs off the dense / compile-hot path
# ──────────────────────────────────────────────────────────────────────────


def _offsets(cu_seqlens: torch.Tensor) -> list[int]:
    """Materialize cumulative offsets on CPU for the per-sample SDPA loop.

    A host copy is acceptable here because this path already decided
    jagged Flash was ineligible; the alternative is a Python index
    into a CUDA tensor per sample, which syncs once per batch item.
    """

    return [int(item) for item in cu_seqlens.detach().cpu().tolist()]


def _integer_max_seqlen(value: int, *, name: str, upper_bound: int | None = None) -> int:
    """Accept any ``operator.index`` integer; reject bools and out-of-range."""

    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if resolved < 0 or (upper_bound is not None and resolved > upper_bound):
        suffix = f" and at most {upper_bound}" if upper_bound is not None else ""
        raise ValueError(f"{name} must be non-negative{suffix}")
    return resolved


def _assert_lengths_fit_max(lengths: torch.Tensor, maximum: int, *, name: str) -> None:
    """Fail if any packed length exceeds the declared max (async on CUDA).

    ``torch._assert_async`` avoids a host sync on the hot varlen path;
    older PyTorch falls back to ``.item()``.
    """

    if lengths.numel() == 0:
        return
    valid = torch.all((lengths >= 0) & (lengths <= maximum))
    message = f"sequence lengths must be between 0 and {name}={maximum}"
    if lengths.device.type == "cpu":
        if not bool(valid.item()):
            raise ValueError(message)
        return
    assert_async = getattr(torch, "_assert_async", None)
    if callable(assert_async):
        assert_async(valid, message)
    elif not bool(valid.item()):  # pragma: no cover - compatibility with old torch
        raise ValueError(message)


def _resolve_max_seqlen(
    provided: int | None,
    *,
    lengths: torch.Tensor,
    padded_length: int,
    batch: int,
    name: str,
) -> int:
    """Use a caller-supplied max when present so launch avoids ``.item()``."""

    if provided is None:
        return int(lengths.max().item()) if batch else 0
    resolved = _integer_max_seqlen(provided, name=name, upper_bound=padded_length)
    _assert_lengths_fit_max(lengths, resolved, name=name)
    return resolved


def _max_from_cumulative(cumulative: torch.Tensor, provided: int | None, *, name: str) -> int:
    """Resolve max seqlen from ``cu_seqlens`` diffs, or trust a caller max."""

    lengths = cumulative[1:] - cumulative[:-1]
    if provided is None:
        return int(lengths.max().item()) if lengths.numel() else 0
    resolved = _integer_max_seqlen(provided, name=name)
    _assert_lengths_fit_max(lengths, resolved, name=name)
    return resolved


def _validated_lengths(
    lengths: torch.Tensor,
    *,
    batch: int,
    maximum: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Move a per-sample length vector to device int32 and bound-check it."""

    if lengths.ndim != 1 or lengths.numel() != batch:
        raise ValueError(f"{name} must contain exactly one length per batch item")
    lengths = lengths.to(device=device, dtype=torch.int32, non_blocking=True)
    _assert_lengths_fit_max(lengths, maximum, name=name)
    return lengths


def _validated_padding_mask(
    mask: torch.Tensor | None,
    *,
    batch: int,
    length: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    """Require a ``[B, L]`` keep-mask; ``None`` means the axis is dense."""

    if mask is None:
        return None
    if mask.shape != (batch, length):
        raise ValueError(f"{name} must have shape [{batch}, {length}]")
    return mask.to(device=device, dtype=torch.bool, non_blocking=True)


def _bottom_right_causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """Bottom-right causal keep-mask (FlashAttention rectangular convention)."""

    query_positions = torch.arange(q_len, device=device)[:, None]
    key_positions = torch.arange(k_len, device=device)[None, :]
    return key_positions <= query_positions + (k_len - q_len)


def _bottom_right_window_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    *,
    window_size: tuple[int, int],
    causal: bool,
) -> torch.Tensor:
    """Build FlashAttention-compatible rectangular local-attention semantics."""

    left, right = window_size
    query_centers = torch.arange(q_len, device=device)[:, None] + (k_len - q_len)
    key_positions = torch.arange(k_len, device=device)[None, :]
    allowed = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
    if left >= 0:
        allowed &= key_positions >= query_centers - left
    if right >= 0:
        allowed &= key_positions <= query_centers + right
    if causal:
        allowed &= key_positions <= query_centers
    return allowed


def _validated_window_size(window_size: tuple[int, int]) -> tuple[int, int]:
    """Require a ``(left, right)`` pair; ``-1`` means an unbounded side."""

    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        raise ValueError("window_size must be a (left, right) pair")
    left, right = (int(item) for item in window_size)
    if left < -1 or right < -1:
        raise ValueError("window_size entries must be -1 or non-negative")
    return left, right


__all__ = ["attention", "flash_attention", "varlen_scaled_dot_product_attention"]
