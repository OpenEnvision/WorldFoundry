# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""DreamZero flash-attention dispatcher (Wan 2.1 kernels).

Kernel adapter for DreamZero blocks.  No action math here.
"""

import os

import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name

    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention
from worldfoundry.core.attention.sequence.varlen import flash_attention as _worldfoundry_varlen_attention


def _gpu_supports_flash_attention():
    """FlashAttention requires Ampere (compute capability 8.0) or newer."""
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8
    except Exception:
        return False


def _sdpa_attention_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """PyTorch SDPA fallback for GPUs that don't support FlashAttention (e.g. pre-Ampere)."""
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention on this GPU. "
            "It can have a slight impact on quality.",
            stacklevel=1,
        )
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * softmax_scale
    out = _worldfoundry_scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous()


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
    window_size: tuple[int, int] | None = None,
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    version:        int. 2 for flash attention 2, 3 for flash attention 3.

    Returns:
        x:              [B, Lq, Nq, C2].
    """
    if window_size is None:
        window_size = (-1, -1)
    if version is None:
        version = 3

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # The shared dispatcher handles the current FlashAttention 3 Python ABI
    # (including seqused_q/seqused_k) and preserves variable lengths. The
    # older in-tree FA3 call below targets an obsolete ABI. It also supplies
    # the SDPA fallback when an optional FlashAttention package is absent.
    if version == 3 or not _gpu_supports_flash_attention() or (version == 2 and not FLASH_ATTN_2_AVAILABLE):
        return _worldfoundry_varlen_attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens, strict=False)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens, strict=False)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens, strict=False)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.", stacklevel=1)
    zeros = torch.zeros([1], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0).to(torch.int32)
    cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0).to(torch.int32)

    # apply attention
    if version == 3 and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        raise ValueError(f"Invalid version: {version}")

    # output
    return x.type(out_dtype)


class AttentionModule(torch.nn.Module):
    """DreamZero packed-sequence attention wrapper."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        q_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] | None = None,
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backend: str | None = None,
    ):
        super().__init__()
        if backend is None:
            backend = "torch"

        backend = os.getenv("ATTENTION_BACKEND") if os.getenv("ATTENTION_BACKEND") is not None else "FA2"

        # Check for TensorRT at runtime, not import time
        if os.getenv("ENABLE_TENSORRT", "False").lower() == "true":
            backend = "torch"

        # The Transformer Engine backend is not bundled; TE requests use FA2.
        if backend == "TE":
            print("Warning: Transformer Engine is not available. Falling back to FA2 backend.")
            backend = "FA2"

        assert backend in ["torch", "FA2", "FA3", "torch_onnx"]
        self.backend = backend

        if backend == "torch":

            def _torch_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                q = q.transpose(1, 2).to(dtype)
                k = k.transpose(1, 2).to(dtype)
                v = v.transpose(1, 2).to(dtype)

                out = _worldfoundry_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

                out = out.transpose(1, 2).contiguous()
                return out.to(out_dtype)

            self.attn_func = _torch_impl

        elif backend == "torch_onnx":

            def _torch_onnx_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                # use _worldfoundry_scaled_dot_product_attention for tensorrt export

                # The input is (s, n, d), but sdpa needs (b, h, s, d).
                # We add a batch dimension and transpose.
                q = q.unsqueeze(0).transpose(1, 2).to(dtype)
                k = k.unsqueeze(0).transpose(1, 2).to(dtype)
                v = v.unsqueeze(0).transpose(1, 2).to(dtype)

                # Fix for ONNX export: repeat k and v to match q's batch size in cross-attention
                if q.shape[0] != k.shape[0] and k.shape[0] == 1:
                    k = k.repeat(q.shape[0], 1, 1, 1)
                    v = v.repeat(q.shape[0], 1, 1, 1)

                out = _worldfoundry_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

                # Transpose back to (b, s, n, d) format.
                out = out.transpose(1, 2).contiguous()
                return out.to(out_dtype)

            self.attn_func = _torch_onnx_impl

        elif backend == "FA2" or backend == "FA3":

            def _flash_attn_impl(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                q_lens: torch.Tensor | None,
                k_lens: torch.Tensor | None,
            ) -> torch.Tensor:
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
                    version=3 if backend == "FA3" else 2,
                )

            self.attn_func = _flash_attn_impl

        else:
            raise ValueError(f"Invalid backend: {backend}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: torch.Tensor | None = None,
        k_lens: torch.Tensor | None = None,
    ):
        if self.backend == "torch" or self.backend == "torch_onnx":
            if q_lens is not None or k_lens is not None:
                warnings.warn(
                    "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.",
                    stacklevel=1,
                )
            return self.attn_func(q, k, v)  # type: ignore[call-arg]
        else:
            return self.attn_func(q, k, v, q_lens, k_lens)  # type: ignore[call-arg]
