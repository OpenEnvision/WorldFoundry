"""LTX multi-head attention on packed ``B T D`` tokens.

QK-RMSNorm self- or cross-attn with LTX-specific RoPE, optional STG
perturbation bypass (skip Q/K when the whole batch is perturbed), and
optional per-head gating.  Backends come from :class:`TransformerAttentionOps`.
"""

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from worldfoundry.core.attention.cache.context import ContextAttentionCache, prepend_history_mask
from worldfoundry.core.nn import TransformerAttentionOps as AttentionOps

from .rope import LTXRopeType, apply_rotary_emb

_CUDNN_FP8_ALLOWED_OPTIONS = frozenset({"descale_q", "descale_k", "descale_v", "scale_o", "dynamic_scale"})
_CUDNN_FP8_SCALE_KEYS = ("descale_q", "descale_k", "descale_v", "scale_o")
_APPROXIMATE_ATTENTION_BACKENDS = frozenset(
    {"sol_attn", "cudnn_fp8", "sage_attention", "sage_attention_3", "scaled_int8_fp8"}
)
_FP8_E4M3_MAX = 448.0


def _validate_min_tokens(raw: object) -> int:
    """Validate optional ``min_tokens`` sequence-length threshold for approximate attention."""
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"attention min_tokens must be a non-negative integer, got {raw!r}")
    return int(raw)


def _validate_cudnn_fp8_options(options: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize ``cudnn_fp8`` backend options upfront."""
    unknown = set(options) - _CUDNN_FP8_ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"unknown cudnn_fp8 options: {sorted(unknown)}")
    normalized: dict[str, object] = {}
    if "dynamic_scale" in options:
        raw_dynamic = options["dynamic_scale"]
        if not isinstance(raw_dynamic, bool):
            raise ValueError(f"cudnn_fp8 dynamic_scale must be a bool, got {type(raw_dynamic).__name__}")
        normalized["dynamic_scale"] = raw_dynamic
    limits = torch.finfo(torch.float32)
    has_static_scale = False
    for key in _CUDNN_FP8_SCALE_KEYS:
        if key not in options or options[key] is None:
            continue
        raw = options[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"cudnn_fp8 {key} must be a positive finite float, got {raw!r}")
        value = float(raw)
        if not math.isfinite(value) or value < limits.tiny or value > limits.max:
            raise ValueError(f"cudnn_fp8 {key} must be positive and finite in float32, got {raw!r}")
        normalized[key] = value
        has_static_scale = True
    if normalized.get("dynamic_scale", False) and has_static_scale:
        raise ValueError("cudnn_fp8 dynamic_scale=True cannot be combined with explicit static scales")
    return normalized


class Attention(torch.nn.Module):
    """Checkpoint-shaped LTX attention (self-attn when ``context_dim`` is ``None``)."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        ops: AttentionOps | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        if ops is None:
            ops = AttentionOps()
        self.rope_type = rope_type
        self.attention_function = ops.attention_function
        self.masked_attention_function = ops.masked_attention_function
        self.preattention_function = ops.preattention_function
        self.gated_attention_function = ops.gated_attention_function
        self.attention_backend: str | None = None
        self._attention_backend_options: dict[str, Any] | None = None
        self._attention_min_tokens: int = 0
        self._dense_attention_backend: str | None = None
        self._worldfoundry_force_dense: bool = False
        self._worldfoundry_sol_sink_tokens: int | None = None
        self._use_fused_gated_out: bool = False

        self.is_self_attention = context_dim is None
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.num_heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, object] | None = None,
    ) -> None:
        """Bind a request-scoped attention backend (``AttentionBackendConfigurable`` seam).

        ``None`` or ``"torch"`` keeps the configured ``TransformerAttentionOps`` callables.
        Explicit backends (``"cudnn_fp8"``, ``"sol_attn"``, ``"triton_tma"``,
        ``"scaled_int8_fp8"``, ``"flash_attention_3"``, etc.) route unmasked attention
        through :mod:`worldfoundry.core.attention.backends.dispatch`.
        """
        if backend is None:
            if backend_options:
                raise ValueError("backend_options require an explicit attention backend")
            self.attention_backend = None
            self._attention_backend_options = None
            self._attention_min_tokens = 0
            self._dense_attention_backend = None
            return

        from worldfoundry.core.attention.backends.probe import normalize_attention_backend

        normalized = normalize_attention_backend(str(backend))
        options: dict[str, Any] = dict(backend_options) if backend_options else {}
        min_tokens = 0
        dense_backend_specified = "dense_backend" in options
        dense_backend: str | None = None
        if "min_tokens" in options:
            if normalized not in {"sol_attn", "cudnn_fp8"}:
                raise ValueError(f"min_tokens is only supported for sol_attn and cudnn_fp8, got {normalized!r}")
            min_tokens = _validate_min_tokens(options.pop("min_tokens"))
        if dense_backend_specified:
            if normalized not in {"sol_attn", "cudnn_fp8"}:
                raise ValueError(f"dense_backend is only supported for sol_attn and cudnn_fp8, got {normalized!r}")
            raw_dense = options.pop("dense_backend")
            if raw_dense is not None:
                dense_backend = normalize_attention_backend(str(raw_dense))
                if dense_backend in _APPROXIMATE_ATTENTION_BACKENDS:
                    raise ValueError(f"dense_backend must be an exact attention backend, got {dense_backend!r}")
        if normalized == "sol_attn":
            if not self.is_self_attention:
                raise ValueError("Sol-Attn is supported only on LTX self-attention modules")
            if self.dim_head != 128:
                raise ValueError(f"Sol-Attn requires dim_head=128, got {self.dim_head}")
            from worldfoundry.core.attention.backends.sol import validate_sol_options

            options = validate_sol_options(**options)
        elif normalized == "cudnn_fp8":
            if self.dim_head not in {64, 128}:
                raise ValueError(f"cudnn_fp8 requires dim_head in {{64, 128}}, got {self.dim_head}")
            options = _validate_cudnn_fp8_options(options)
        elif options and normalized != "scaled_int8_fp8":
            raise ValueError(f"backend_options are not supported for attention backend {normalized!r}")

        if normalized in {"sol_attn", "cudnn_fp8"}:
            if dense_backend_specified:
                self._dense_attention_backend = dense_backend
            elif self.attention_backend in {"sol_attn", "cudnn_fp8"}:
                # Re-configuring approximate backend options keeps any previously bound exact dense fallback.
                pass
            elif self.attention_backend not in {None, "torch", "auto", *_APPROXIMATE_ATTENTION_BACKENDS}:
                # Preserve an already-bound exact dense backend (e.g. flash_attention_3)
                # for short sequences (Stage 1) and dense warmup steps.
                self._dense_attention_backend = self.attention_backend
            else:
                self._dense_attention_backend = None
        else:
            self._dense_attention_backend = None

        self.attention_backend = normalized
        self._attention_backend_options = options or None
        self._attention_min_tokens = min_tokens

    def fuse_qkv(self) -> bool:
        """Merge self-attention ``to_q``/``to_k``/``to_v`` into one ``to_qkv`` projection."""
        if not self.is_self_attention or "to_qkv" in self._modules:
            return False
        to_q = self._modules.get("to_q")
        to_k = self._modules.get("to_k")
        to_v = self._modules.get("to_v")
        if not (
            isinstance(to_q, torch.nn.Linear)
            and isinstance(to_k, torch.nn.Linear)
            and isinstance(to_v, torch.nn.Linear)
        ):
            return False
        if any(layer.weight.is_meta for layer in (to_q, to_k, to_v)):
            return False
        if (
            to_q.in_features != to_k.in_features
            or to_q.in_features != to_v.in_features
            or to_q.out_features != to_k.out_features
            or to_q.out_features != to_v.out_features
        ):
            return False
        has_bias = to_q.bias is not None
        if (to_k.bias is not None) != has_bias or (to_v.bias is not None) != has_bias:
            return False
        fused = torch.nn.Linear(
            to_q.in_features,
            3 * to_q.out_features,
            bias=has_bias,
            device=to_q.weight.device,
            dtype=to_q.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat((to_q.weight, to_k.weight, to_v.weight), dim=0))
            if has_bias:
                assert fused.bias is not None
                assert to_q.bias is not None and to_k.bias is not None and to_v.bias is not None
                fused.bias.copy_(torch.cat((to_q.bias, to_k.bias, to_v.bias), dim=0))
        self.add_module("to_qkv", fused)
        del self._modules["to_q"], self._modules["to_k"], self._modules["to_v"]
        return True

    def fuse_kv(self) -> bool:
        """Merge cross-attention ``to_k``/``to_v`` into one ``to_kv`` projection."""
        if self.is_self_attention or "to_kv" in self._modules:
            return False
        to_k = self._modules.get("to_k")
        to_v = self._modules.get("to_v")
        if not (isinstance(to_k, torch.nn.Linear) and isinstance(to_v, torch.nn.Linear)):
            return False
        if to_k.weight.is_meta or to_v.weight.is_meta:
            return False
        if to_k.in_features != to_v.in_features or to_k.out_features != to_v.out_features:
            return False
        has_bias = to_k.bias is not None
        if (to_v.bias is not None) != has_bias:
            return False
        fused = torch.nn.Linear(
            to_k.in_features,
            2 * to_k.out_features,
            bias=has_bias,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat((to_k.weight, to_v.weight), dim=0))
            if has_bias:
                assert fused.bias is not None and to_k.bias is not None and to_v.bias is not None
                fused.bias.copy_(torch.cat((to_k.bias, to_v.bias), dim=0))
        self.add_module("to_kv", fused)
        del self._modules["to_k"], self._modules["to_v"]
        return True

    def apply_rotary_embedding(self, value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Bind LTX's checkpoint-specific RoPE layout to the shared pre-attention policy."""

        return apply_rotary_emb(value, position, self.rope_type)

    def _project_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        is_self_call: bool,
        all_perturbed: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Project ``(q, k, v)`` through separate or post-load fused linear modules."""
        to_qkv = self._modules.get("to_qkv")
        if to_qkv is not None:
            inner_dim = self.heads * self.dim_head
            if all_perturbed:
                if isinstance(to_qkv, torch.nn.Linear):
                    bias = None if to_qkv.bias is None else to_qkv.bias[2 * inner_dim :]
                    return None, None, F.linear(context, to_qkv.weight[2 * inner_dim :], bias)
                return None, None, to_qkv(context).chunk(3, dim=-1)[2]
            if is_self_call:
                q, k, v = to_qkv(x).chunk(3, dim=-1)
                return q, k, v
            if isinstance(to_qkv, torch.nn.Linear):
                q_bias = None if to_qkv.bias is None else to_qkv.bias[:inner_dim]
                kv_bias = None if to_qkv.bias is None else to_qkv.bias[inner_dim:]
                q = F.linear(x, to_qkv.weight[:inner_dim], q_bias)
                k, v = F.linear(context, to_qkv.weight[inner_dim:], kv_bias).chunk(2, dim=-1)
                return q, k, v
            q = to_qkv(x).chunk(3, dim=-1)[0]
            _, k, v = to_qkv(context).chunk(3, dim=-1)
            return q, k, v

        to_kv = self._modules.get("to_kv")
        if to_kv is not None:
            inner_dim = self.heads * self.dim_head
            if all_perturbed:
                if isinstance(to_kv, torch.nn.Linear):
                    bias = None if to_kv.bias is None else to_kv.bias[inner_dim:]
                    return None, None, F.linear(context, to_kv.weight[inner_dim:], bias)
                return None, None, to_kv(context).chunk(2, dim=-1)[1]
            k, v = to_kv(context).chunk(2, dim=-1)
            q = self.to_q(x)
            return q, k, v

        v = self.to_v(context)
        if all_perturbed:
            return None, None, v
        return self.to_q(x), self.to_k(context), v

    def _run_cudnn_fp8_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize Q/K/V to E4M3 and invoke cuDNN Frontend FP8 SDPA."""
        batch, q_tokens, width = q.shape
        k_tokens = k.shape[1]
        q_bhld = q.view(batch, q_tokens, self.heads, self.dim_head).transpose(1, 2).contiguous()
        k_bhld = k.view(batch, k_tokens, self.heads, self.dim_head).transpose(1, 2).contiguous()
        v_bhld = v.view(batch, k_tokens, self.heads, self.dim_head).transpose(1, 2).contiguous()

        opts = self._attention_backend_options or {}
        dynamic_scale = bool(opts.get("dynamic_scale", False))
        has_custom_scales = any(key in opts for key in _CUDNN_FP8_SCALE_KEYS)

        if dynamic_scale:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "cudnn_fp8 with dynamic_scale=True requires host synchronization and cannot run "
                    "during CUDA Graph capture; use static calibrated descale_q/descale_k/descale_v "
                    "or default unit scaling"
                )
            from worldfoundry.core.attention.backends.dispatch import (
                _record_compiled_provider_graph_trace,
                _record_provider_runtime,
            )
            from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa

            tiny = float(torch.finfo(torch.float32).tiny)
            amax_q, amax_k, amax_v = (
                torch.stack((q_bhld.abs().amax(), k_bhld.abs().amax(), v_bhld.abs().amax())).float().tolist()
            )
            dq = max(amax_q / _FP8_E4M3_MAX, tiny)
            dk = max(amax_k / _FP8_E4M3_MAX, tiny)
            dv = max(amax_v / _FP8_E4M3_MAX, tiny)
            qf = (q_bhld.float() * (1.0 / dq)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            kf = (k_bhld.float() * (1.0 / dk)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            vf = (v_bhld.float() * (1.0 / dv)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            try:
                out = native_cudnn_fp8_sdpa(qf, kf, vf, descale_q=dq, descale_k=dk, descale_v=dv)
            except Exception:
                _record_provider_runtime("cudnn_fp8", attempts=1, errors=1)
                raise
            if torch.compiler.is_compiling():
                _record_compiled_provider_graph_trace("cudnn_fp8")
            else:
                _record_provider_runtime("cudnn_fp8", attempts=1, successes=1)
        elif has_custom_scales:
            from worldfoundry.core.attention.backends.dispatch import (
                _record_compiled_provider_graph_trace,
                _record_provider_runtime,
            )
            from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa

            dq = float(opts.get("descale_q", 1.0))
            dk = float(opts.get("descale_k", 1.0))
            dv = float(opts.get("descale_v", 1.0))
            raw_scale_o = opts.get("scale_o")
            scale_o = None if raw_scale_o is None else float(raw_scale_o)
            qf = (q_bhld.float() * (1.0 / dq)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            kf = (k_bhld.float() * (1.0 / dk)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            vf = (v_bhld.float() * (1.0 / dv)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            try:
                out = native_cudnn_fp8_sdpa(qf, kf, vf, descale_q=dq, descale_k=dk, descale_v=dv, scale_o=scale_o)
            except Exception:
                _record_provider_runtime("cudnn_fp8", attempts=1, errors=1)
                raise
            if torch.compiler.is_compiling():
                _record_compiled_provider_graph_trace("cudnn_fp8")
            else:
                _record_provider_runtime("cudnn_fp8", attempts=1, successes=1)
        else:
            from worldfoundry.core.attention.backends.dispatch import attention_forward

            qf = q_bhld.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            kf = k_bhld.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            vf = v_bhld.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            out = attention_forward(
                qf,
                kf,
                vf,
                q_pattern="b n s d",
                k_pattern="b n s d",
                v_pattern="b n s d",
                out_pattern="b n s d",
                backend="cudnn_fp8",
            )
        return out.transpose(1, 2).reshape(batch, q_tokens, width).to(dtype=v.dtype)

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch attention to the configured backend or default ``TransformerAttentionOps``."""
        backend = self.attention_backend
        use_dense_fallback = backend in _APPROXIMATE_ATTENTION_BACKENDS and (
            self._worldfoundry_force_dense
            or (self._attention_min_tokens > 0 and q.shape[1] < self._attention_min_tokens)
        )
        backend_options = self._attention_backend_options
        if use_dense_fallback:
            backend = self._dense_attention_backend
            backend_options = None

        if backend is None or backend in {"torch", "auto"}:
            if mask is None:
                if (q.device.type != "cuda" or v.dtype not in {torch.float16, torch.bfloat16}) and getattr(
                    self.attention_function, "label", ""
                ) in {"FlashAttention3", "FlashAttention4"}:
                    return self.masked_attention_function(q, k, v, self.heads, None)
                return self.attention_function(q, k, v, self.heads)
            return self.masked_attention_function(q, k, v, self.heads, mask)

        if mask is not None:
            if backend == "sol_attn":
                raise ValueError("Sol-Attn does not support attention masks")
            return self.masked_attention_function(q, k, v, self.heads, mask)

        if backend == "cudnn_fp8":
            return self._run_cudnn_fp8_attention(q, k, v)

        # External providers (Sol-Attn, Triton TMA, FA3/FA4, etc.) are not in
        # PyTorch's autocast dispatch table. Under CUDA autocast, align Q/K/V to
        # the active autocast dtype just as F.scaled_dot_product_attention does
        # (e.g., FP32 RoPE promotes Q/K to FP32 while V stays BF16).
        if q.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            if q.dtype != target_dtype:
                q = q.to(dtype=target_dtype)
            if k.dtype != target_dtype:
                k = k.to(dtype=target_dtype)
            if v.dtype != target_dtype:
                v = v.to(dtype=target_dtype)

        from worldfoundry.core.attention.backends.dispatch import attention_forward

        batch, q_tokens, width = q.shape
        k_tokens = k.shape[1]
        q4 = q.view(batch, q_tokens, self.heads, self.dim_head)
        k4 = k.view(batch, k_tokens, self.heads, self.dim_head)
        v4 = v.view(batch, k_tokens, self.heads, self.dim_head)
        if backend == "sol_attn" and self._worldfoundry_sol_sink_tokens is not None:
            backend_options = {**(backend_options or {}), "sink_tokens": int(self._worldfoundry_sol_sink_tokens)}
        out = attention_forward(
            q4,
            k4,
            v4,
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            backend=backend,
            backend_options=backend_options,
        )
        return out.reshape(batch, q_tokens, width)

    def _apply_gated_out(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Fuse per-head sigmoid gating with output projection (and rowwise FP8 quant when eligible)."""
        assert self.to_gate_logits is not None
        from worldfoundry.core.nn import PytorchGatedAttention

        if not self._use_fused_gated_out or type(self.gated_attention_function) is not PytorchGatedAttention:
            return self.to_out(self.gated_attention_function(x, out, self))

        from .fused_ops import fused_ltx_gated_attention, fused_ltx_gated_attention_fp8

        gate_logits = self.to_gate_logits(x)
        if isinstance(self.to_out, torch.nn.Sequential) and len(self.to_out) == 2:
            out_proj = self.to_out[0]
            tail = self.to_out[1]
            if isinstance(tail, torch.nn.Identity):
                from worldfoundry.core.acceleration.quantization.linear import Float8Linear, _fp8_linear_eligible

                if (
                    isinstance(out_proj, Float8Linear)
                    and out_proj.low_precision_enabled
                    and out_proj.scaling == "rowwise"
                    and out_proj.input_scale is None
                ):
                    proj_dtype = out.dtype
                    if proj_dtype == torch.float32 and out.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
                        ac_dtype = torch.get_autocast_dtype("cuda")
                        if ac_dtype in (torch.float16, torch.bfloat16):
                            out = out.to(dtype=ac_dtype)
                            proj_dtype = ac_dtype
                    if _fp8_linear_eligible(
                        out,
                        out_proj.out_features,
                        out_proj._hardware_eligible,
                        min_gemm_work=0.0 if out_proj.weight is None else None,
                    ):
                        quant = fused_ltx_gated_attention_fp8(
                            out,
                            gate_logits,
                            heads=self.heads,
                            dim_head=self.dim_head,
                            fp8_dtype=out_proj.fp8_dtype,
                        )
                        if quant is not None:
                            codes, scales = quant
                            return out_proj._forward_quantized(codes, scales, out.shape, proj_dtype)
        gated = fused_ltx_gated_attention(out, gate_logits, heads=self.heads, dim_head=self.dim_head)
        return self.to_out(gated)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        camera_control: object | None = None,
        cache: ContextAttentionCache | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA). A non-None
                ``mask`` routes to ``masked_attention_function``; ``None`` keeps
                the unmasked path.
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        is_self_call = context is None
        context = x if is_self_call else context
        use_attention = not all_perturbed

        q, k, v = self._project_qkv(
            x,
            context,
            is_self_call=is_self_call,
            all_perturbed=all_perturbed,
        )
        current_value = v
        camera_out = None

        if not use_attention:
            out = v
        else:
            assert q is not None and k is not None
            if camera_control is not None:
                camera_q, camera_k = self.preattention_function(q, k, self, mask, None, None)
                camera_out = self._camera_attention(
                    camera_q,
                    camera_k,
                    v,
                    x,
                    camera_control,
                    mask,
                    perturbation_mask,
                    cache,
                )
            q, k = self.preattention_function(q, k, self, mask, pe, k_pe)
            if cache is not None:
                k, v, _, history_tokens = cache.combine(self, k, v)
                mask = prepend_history_mask(mask, history_tokens)
            out = self._run_attention(q, k, v, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + current_value * (1 - perturbation_mask)

        # Apply per-head gating and output projection (fused when enabled)
        out = self._apply_gated_out(x, out) if self.to_gate_logits is not None else self.to_out(out)
        return out if camera_out is None else out + camera_out

    def _camera_attention(self, q, k, v, x, control, mask, perturbation_mask, cache):
        """Optional checkpoint-owned projective residual."""
        return None
