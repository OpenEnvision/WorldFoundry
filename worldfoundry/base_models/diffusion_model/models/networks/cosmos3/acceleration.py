# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for Cosmos3-Super & Cosmos3-Nano MoT transformers.

Implements the full Cosmos3 SOL-Engine recipe:

1. **Exact Lossless ``und_seq`` Static KV Cache**: Across all layers (36L Nano
   / 64L Super), the causal understanding prefix ``und_seq`` has zero timestep
   modulation and is bitwise-identical across denoising steps ``0..N-1``.
   Compute and cache per-layer ``(k_und, v_und)`` plus final ``und_out`` on
   step 0, then skip 100% of understanding projections, norms, causal attention,
   and SwiGLU MLPs on steps ``1..N-1`` (strict ``rel_l2 == 0.0`` parity).
2. **Calibrated TeaCache / FirstBlock Cross-Step Caching** on the ``gen``
   stream: evaluates layer 0 on ``gen_seq``, tracks the relative L1 change of
   the layer-0 residual (with TeaCache polynomial rescaling or direct
   FirstBlock threshold), protects boundary steps (warmup + dense tail + max
   consecutive skips), and skips layers ``1..L-1`` on cache hits.
3. **Step-Selective Quantization (Zero Weight Duplication)**:
   - Hopper SM90: :class:`StepSelectiveFloat8Linear` (rowwise FP8 W8A8,
     ``use_fast_accum=False``, ``keep_dense_fallback=False``). Interior steps
     run native Hopper ``torch._scaled_mm`` FP8 W8A8; sensitive early/late
     boundary steps run BF16 activations via zero-duplication on-the-fly weight
     dequantization (while edge blocks stay native BF16 ``nn.Linear``).
   - Blackwell SM100 hook: :class:`StepSelectiveNVFP4Linear` (NVFP4 1x16 packed
     weights, ``keep_dense_fallback=False``).
4. **Hopper FlashAttention-3 GQA + Triton Fused Kernels + Piecewise CUDA Graph**:
   - Native Hopper FA3 GQA dispatch (32Q/8KV & 64Q/8KV, zero transpose).
   - Lossless projection fusion (``gen_qkv_proj`` + SwiGLU ``gate_up_proj``).
   - Triton ``fused_rmsnorm_rope_gqa``, ``fused_residual_add_rmsnorm``, and
     ``silu_and_mul``.
   - :class:`PiecewiseBlockGraphRunner` capturing pure-tensor ``gen``-stream
     segments per ``(und_len, gen_len, precision_mode)`` outside Python cache
     gating.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    _autocast_linear_input,
    _fp8_hardware_eligible,
    _fp8_linear_eligible,
)
from worldfoundry.core.acceleration.quantization.nvfp4 import (
    NVFP4Linear,
    _nvfp4_hardware_eligible,
    dequantize_nvfp4,
    nvfp4_unswizzle_scales,
)
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

DEFAULT_COSMOS3_QUANT_EXCLUDES: tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    "proj_in",
    "proj_out",
    "time_proj",
    "time_embedder",
    "audio_proj_in",
    "audio_proj_out",
    "action_proj_in",
    "action_proj_out",
)


@dataclass(frozen=True, slots=True)
class Cosmos3AccelerationConfig:
    """Declarative SOL-Engine acceleration recipe for Cosmos3 Omni transformers."""

    und_static_kv_cache: bool = True
    attention_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3"
    fused_ops: bool = True
    fuse_projections: bool = True
    quantization: Literal["none", "fp8", "nvfp4", "auto"] = "none"
    fp8_scaling: Literal["rowwise", "tensorwise"] = "rowwise"
    fp8_use_fast_accum: bool = False
    keep_dense_fallback: bool = False
    quant_min_features: int = 1024
    quant_exclude_edge_blocks: bool = True
    quant_exclude: tuple[str, ...] = DEFAULT_COSMOS3_QUANT_EXCLUDES
    quant_und_pathway: bool = True
    # Step-selective boundary protection: keep early/late denoising steps in BF16.
    quant_start_step: int = 2
    quant_dense_last_steps: int = 2
    # Cross-step gen-stream caching (TeaCache or FirstBlock).
    cache_algorithm: Literal["none", "teacache", "firstblock"] = "none"
    cache_threshold: float = 0.12
    cache_warmup_steps: int = 4
    cache_dense_last: int = 2
    cache_max_consecutive_skips: int = 2
    cache_teacache_coefficients: tuple[float, ...] = (1.0, 0.0)
    cuda_graph: Literal["off", "piecewise"] = "off"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 8
    cuda_graph_clone_outputs: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quant_exclude"] = list(self.quant_exclude)
        payload["cache_teacache_coefficients"] = list(self.cache_teacache_coefficients)
        payload["notes"] = list(self.notes)
        return payload


class StepSelectiveFloat8Linear(Float8Linear):
    """Rowwise FP8 linear supporting BF16 boundary steps with ZERO weight duplication.

    When ``low_precision_enabled`` is True and running on eligible SM90+ CUDA
    hardware, executes native ``torch._scaled_mm`` FP8 W8A8 (``use_fast_accum=False``
    by default). When ``low_precision_enabled`` is False (early/late sensitive
    denoising steps) or on CPU/non-FP8 hardware, dequantizes ``weight_fp8`` on
    the fly into the input dtype and runs dense ``F.linear`` without holding a
    second dense weight buffer in VRAM.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = False,
        scaling: str = "rowwise",
        use_fast_accum: bool = False,
    ) -> None:
        super().__init__(
            weight,
            bias,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
            scaling=scaling,
            use_fast_accum=use_fast_accum,
        )
        self.step_bf16_calls: int = 0
        self.step_fp8_calls: int = 0
        self.request_step_bf16_calls: int = 0
        self.request_step_fp8_calls: int = 0

    def reset_request_window(self) -> None:
        super().reset_request_window()
        self.request_step_bf16_calls = 0
        self.request_step_fp8_calls = 0

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.weight is not None:
            return self.weight.to(dtype=dtype)
        w = self.weight_fp8.to(dtype=dtype)
        if self.scaling == "rowwise":
            return w * self.weight_scale.reshape(-1, 1).to(dtype=dtype)
        return w * self.weight_scale.to(dtype=dtype)

    def _forward_bf16_dequant(self, input: torch.Tensor) -> torch.Tensor:
        w = self._dequantized_weight(input.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        if not torch.compiler.is_compiling():
            self.step_bf16_calls += 1
            self.request_step_bf16_calls += 1
        return F.linear(input, w, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        input = _autocast_linear_input(input)
        # Boundary steps (low_precision_enabled=False) or non-FP8/grad/CPU/small-GEMM paths
        # run clean BF16/dense F.linear without requiring a duplicate weight copy.
        min_gemm_work = 0.0 if self.weight is None else None
        if not self.low_precision_enabled or not _fp8_linear_eligible(
            input,
            self.out_features,
            self._hardware_eligible,
            min_gemm_work=min_gemm_work,
        ):
            return self._forward_bf16_dequant(input)
        prev_lp = self.low_precision_kernel_calls
        out = super().forward(input)
        if not torch.compiler.is_compiling() and self.low_precision_kernel_calls > prev_lp:
            self.step_fp8_calls += 1
            self.request_step_fp8_calls += 1
        return out

    def runtime_report(self) -> dict[str, Any]:
        return {
            **super().runtime_report(),
            "provider": "cosmos3-step-selective-fp8",
            "keep_dense_fallback": self.weight is not None,
            "step_fp8_calls": self.step_fp8_calls,
            "step_bf16_calls": self.step_bf16_calls,
            "request_step_fp8_calls": self.request_step_fp8_calls,
            "request_step_bf16_calls": self.request_step_bf16_calls,
        }


class StepSelectiveNVFP4Linear(NVFP4Linear):
    """Blackwell SM100 NVFP4 linear hook with step-selective BF16 mode and zero weight duplication."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        keep_dense_fallback: bool = False,
    ) -> None:
        super().__init__(weight, bias, keep_dense_fallback=keep_dense_fallback)
        self.step_bf16_calls: int = 0
        self.step_nvfp4_calls: int = 0
        self.request_step_bf16_calls: int = 0
        self.request_step_nvfp4_calls: int = 0

    def reset_request_window(self) -> None:
        super().reset_request_window()
        self.request_step_bf16_calls = 0
        self.request_step_nvfp4_calls = 0

    def _forward_bf16_dequant(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            w = self.weight.to(dtype=input.dtype)
        else:
            raw_scales = nvfp4_unswizzle_scales(
                self.weight_scale,
                self.out_features,
                self.in_features // 16,
            )
            w = dequantize_nvfp4(
                self.weight_fp4,
                raw_scales,
                self.weight_global_scale,
                dtype=input.dtype,
            )
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        if not torch.compiler.is_compiling():
            self.step_bf16_calls += 1
            self.request_step_bf16_calls += 1
        return F.linear(input, w, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        input = _autocast_linear_input(input)
        if (
            not self.low_precision_enabled
            or input.device.type != "cuda"
            or torch.is_grad_enabled()
            or not self._hardware_eligible
            or input.dtype not in {torch.float16, torch.bfloat16}
            or self.in_features % 32 != 0
            or self.out_features % 16 != 0
        ):
            return self._forward_bf16_dequant(input)
        prev_lp = self.low_precision_kernel_calls
        out = super().forward(input)
        if not torch.compiler.is_compiling() and self.low_precision_kernel_calls > prev_lp:
            self.step_nvfp4_calls += 1
            self.request_step_nvfp4_calls += 1
        return out

    def runtime_report(self) -> dict[str, Any]:
        return {
            **super().runtime_report(),
            "provider": "cosmos3-step-selective-nvfp4",
            "keep_dense_fallback": self.weight is not None,
            "step_nvfp4_calls": self.step_nvfp4_calls,
            "step_bf16_calls": self.step_bf16_calls,
        }


@dataclass
class _UndBranchEntry:
    und_len: int
    und_seq_input: torch.Tensor
    cos_und: torch.Tensor
    sin_und: torch.Tensor
    layer_kv: list[tuple[torch.Tensor, torch.Tensor]]
    und_out: torch.Tensor
    tail_k: torch.Tensor | None = None
    tail_v: torch.Tensor | None = None


class Cosmos3UndStaticKVCache:
    """Exact bitwise-lossless static KV cache for Cosmos3's causal ``und_seq`` stream.

    Stores per-branch ``(k_und, v_und)`` across all layers plus final ``und_out``,
    verifying exact input equality on ``(und_seq_input, cos_und, sin_und)`` so
    prompt changes or CFG positive/negative branches never collide.
    """

    def __init__(self, *, max_branches: int = 4) -> None:
        self.max_branches = int(max_branches)
        self._entries: dict[str, _UndBranchEntry] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.layers_skipped: int = 0

    def reset(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.layers_skipped = 0

    def lookup(
        self,
        branch: str,
        und_seq_input: torch.Tensor,
        cos_und: torch.Tensor,
        sin_und: torch.Tensor,
    ) -> _UndBranchEntry | None:
        entry = self._entries.get(branch)
        if entry is None:
            return None
        if (
            entry.und_len == int(und_seq_input.shape[0])
            and entry.und_seq_input.shape == und_seq_input.shape
            and entry.und_seq_input.dtype == und_seq_input.dtype
            and entry.und_seq_input.device == und_seq_input.device
            and torch.equal(entry.und_seq_input, und_seq_input)
            and torch.equal(entry.cos_und, cos_und)
            and torch.equal(entry.sin_und, sin_und)
        ):
            self.hits += 1
            self.layers_skipped += len(entry.layer_kv)
            return entry
        return None

    def store(
        self,
        branch: str,
        und_seq_input: torch.Tensor,
        cos_und: torch.Tensor,
        sin_und: torch.Tensor,
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
        und_out: torch.Tensor,
    ) -> None:
        if len(self._entries) >= self.max_branches and branch not in self._entries:
            oldest = next(iter(self._entries))
            self._entries.pop(oldest, None)
        cloned_kv = [(k.detach().clone(), v.detach().clone()) for k, v in layer_kv]
        tail_k: torch.Tensor | None = None
        tail_v: torch.Tensor | None = None
        if len(cloned_kv) > 1:
            dev0 = cloned_kv[1][0].device
            if all(k.device == dev0 and v.device == dev0 for k, v in cloned_kv[1:]):
                tail_k = torch.stack([k for k, _ in cloned_kv[1:]], dim=0).contiguous()
                tail_v = torch.stack([v for _, v in cloned_kv[1:]], dim=0).contiguous()
        self._entries[branch] = _UndBranchEntry(
            und_len=int(und_seq_input.shape[0]),
            und_seq_input=und_seq_input.detach().clone(),
            cos_und=cos_und.detach().clone(),
            sin_und=sin_und.detach().clone(),
            layer_kv=cloned_kv,
            und_out=und_out.detach().clone(),
            tail_k=tail_k,
            tail_v=tail_v,
        )
        self.misses += 1

    def receipt(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": True,
            "branches_cached": sorted(self._entries.keys()),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "und_layers_skipped": self.layers_skipped,
        }


class Cosmos3GenStreamCache:
    """Calibrated TeaCache / FirstBlock cross-step cache for Cosmos3's ``gen`` stream.

    Always runs layer 0 on ``gen_seq`` to obtain ``r0 = gen_after_0 - gen_in``.
    Tracks the accumulated (optionally polynomial-rescaled) relative L1 change
    in ``r0`` across steps. On a hit, skips layers ``1..L-1`` and returns
    ``gen_after_0 + cached_tail_residual``.
    """

    def __init__(
        self,
        *,
        algorithm: Literal["teacache", "firstblock"] = "teacache",
        threshold: float = 0.12,
        total_steps: int = 35,
        warmup_steps: int = 4,
        dense_last: int = 2,
        max_consecutive_skips: int = 2,
        coefficients: tuple[float, ...] = (1.0, 0.0),
        eps: float = 1e-6,
    ) -> None:
        if algorithm not in {"teacache", "firstblock"}:
            raise ValueError(f"unsupported Cosmos3 gen cache algorithm {algorithm!r}")
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        self.algorithm = algorithm
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.coefficients = tuple(float(c) for c in coefficients)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._prev_probe: torch.Tensor | None = None
        self._tail_residual: torch.Tensor | None = None
        self._accumulated: float = 0.0
        self._consecutive_skips: int = 0
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device] | None = None
        self.events: list[BlockCacheEvent] = []

    def _rescale(self, raw_rel_l1: float) -> float:
        if self.algorithm != "teacache" or not self.coefficients:
            return raw_rel_l1
        val = 0.0
        for coeff in self.coefficients:
            val = val * raw_rel_l1 + coeff
        return max(0.0, float(val))

    def decide_and_run_tail(
        self,
        *,
        step: int,
        total_steps: int | None,
        gen_after_0: torch.Tensor,
        first_block_residual: torch.Tensor,
        num_layers: int,
        run_tail: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        count = self.total_steps if total_steps is None else int(total_steps)
        layout = (tuple(gen_after_0.shape), gen_after_0.dtype, gen_after_0.device)
        probe = first_block_residual.detach().float()
        reason: str | None = None
        if torch.is_grad_enabled():
            self._tail_residual = None
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._tail_residual = None
            self._prev_probe = None
            reason = "layout-change"
        elif step < self.warmup_steps:
            reason = "warmup"
        elif step >= max(count - self.dense_last, 0):
            reason = "dense-last"
        elif self._tail_residual is None or self._prev_probe is None:
            reason = "seed"
        elif self._consecutive_skips >= self.max_consecutive_skips:
            reason = "max-skip"

        if self._prev_probe is not None and self._prev_probe.shape == probe.shape:
            num = (probe - self._prev_probe).abs().mean()
            den = self._prev_probe.abs().mean().clamp_min(self.eps)
            raw_rel = float((num / den).item())
            scaled = self._rescale(raw_rel)
            if reason is None:
                self._accumulated += scaled
                if not math.isfinite(self._accumulated) or self._accumulated >= self.threshold:
                    reason = "threshold"

        self._prev_probe = probe.clone()
        hit = reason is None
        tail_indices = tuple(range(1, num_layers))
        if hit:
            out = gen_after_0 + cast(torch.Tensor, self._tail_residual)
            self._consecutive_skips += 1
            dense_blocks: tuple[int, ...] = (0,)
            skipped_blocks: tuple[int, ...] = tail_indices
        else:
            out = run_tail(gen_after_0)
            if not torch.is_grad_enabled():
                self._tail_residual = (out - gen_after_0).detach().clone()
                self._layout = layout
            self._accumulated = 0.0
            self._consecutive_skips = 0
            dense_blocks = tuple(range(num_layers))
            skipped_blocks = ()

        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                dense_blocks,
                skipped_blocks,
                self._accumulated,
            )
        )
        return out, hit

    def record_dense_step(
        self,
        *,
        step: int,
        gen_after_0: torch.Tensor,
        first_block_residual: torch.Tensor,
        gen_final: torch.Tensor,
        num_layers: int,
        reason: str = "und-cache-miss",
    ) -> None:
        """Record an already-computed full dense step (e.g. step 0 / prompt change) without skipping."""
        layout = (tuple(gen_after_0.shape), gen_after_0.dtype, gen_after_0.device)
        probe = first_block_residual.detach().float()
        if not torch.is_grad_enabled():
            self._prev_probe = probe.clone()
            self._tail_residual = (gen_final - gen_after_0).detach().clone()
            self._layout = layout
        self._accumulated = 0.0
        self._consecutive_skips = 0
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                False,
                "warmup" if int(step) < self.warmup_steps else reason,
                tuple(range(num_layers)),
                (),
                0.0,
            )
        )

    @property
    def hits(self) -> int:
        return sum(1 for ev in self.events if ev.hit)

    @property
    def dense_block_calls(self) -> int:
        return sum(ev.dense_block_calls for ev in self.events)

    @property
    def skipped_block_calls(self) -> int:
        return sum(ev.skipped_block_calls for ev in self.events)

    def receipt(self) -> dict[str, Any]:
        total = len(self.events)
        hits = self.hits
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_consecutive_skips": self.max_consecutive_skips,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "fallbacks": [],
        }


def _fuse_three_linears(q: nn.Linear, k: nn.Linear, v: nn.Linear) -> nn.Linear:
    any_bias = q.bias is not None or k.bias is not None or v.bias is not None
    with torch.no_grad():
        weight = torch.cat((q.weight.data, k.weight.data, v.weight.data), dim=0).contiguous()
        if any_bias:
            qb = (
                q.bias.data
                if q.bias is not None
                else torch.zeros(q.out_features, device=weight.device, dtype=weight.dtype)
            )
            kb = (
                k.bias.data
                if k.bias is not None
                else torch.zeros(k.out_features, device=weight.device, dtype=weight.dtype)
            )
            vb = (
                v.bias.data
                if v.bias is not None
                else torch.zeros(v.out_features, device=weight.device, dtype=weight.dtype)
            )
            bias: torch.Tensor | None = torch.cat((qb, kb, vb), dim=0).contiguous()
        else:
            bias = None
    fused = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    fused.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        fused.bias = nn.Parameter(bias, requires_grad=False)
    return fused


def _fuse_two_linears(gate: nn.Linear, up: nn.Linear) -> nn.Linear:
    any_bias = gate.bias is not None or up.bias is not None
    with torch.no_grad():
        weight = torch.cat((gate.weight.data, up.weight.data), dim=0).contiguous()
        if any_bias:
            gb = (
                gate.bias.data
                if gate.bias is not None
                else torch.zeros(gate.out_features, device=weight.device, dtype=weight.dtype)
            )
            ub = (
                up.bias.data
                if up.bias is not None
                else torch.zeros(up.out_features, device=weight.device, dtype=weight.dtype)
            )
            bias: torch.Tensor | None = torch.cat((gb, ub), dim=0).contiguous()
        else:
            bias = None
    fused = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    fused.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        fused.bias = nn.Parameter(bias, requires_grad=False)
    return fused


def fuse_cosmos3_projections(model: nn.Module, *, release_originals: bool = True) -> dict[str, int]:
    """Losslessly fuse Cosmos3 ``gen`` QKV and SwiGLU ``gate+up`` projections in-place.

    When ``release_originals=True``, original separate modules are replaced with
    ``None`` immediately per layer so peak VRAM does not double on 64B models.
    """
    qkv_fused = 0
    swiglu_fused = 0
    layers = getattr(model, "layers", ())
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if (
            attn is not None
            and getattr(attn, "gen_qkv_proj", None) is None
            and isinstance(getattr(attn, "add_q_proj", None), nn.Linear)
            and isinstance(getattr(attn, "add_k_proj", None), nn.Linear)
            and isinstance(getattr(attn, "add_v_proj", None), nn.Linear)
        ):
            attn.gen_qkv_proj = _fuse_three_linears(attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
            if release_originals:
                attn.add_q_proj = None
                attn.add_k_proj = None
                attn.add_v_proj = None
            qkv_fused += 1
        for mlp_attr in ("mlp_moe_gen", "mlp"):
            mlp = getattr(layer, mlp_attr, None)
            if (
                mlp is not None
                and getattr(mlp, "gate_up_proj", None) is None
                and isinstance(getattr(mlp, "gate_proj", None), nn.Linear)
                and isinstance(getattr(mlp, "up_proj", None), nn.Linear)
            ):
                mlp.gate_up_proj = _fuse_two_linears(mlp.gate_proj, mlp.up_proj)
                if release_originals:
                    mlp.gate_proj = None
                    mlp.up_proj = None
                swiglu_fused += 1
        if release_originals and (layer_idx + 1) % 8 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if release_originals and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    return {"gen_qkv_fused": qkv_fused, "swiglu_gate_up_fused": swiglu_fused}


def quantize_cosmos3_step_selective(
    model: nn.Module,
    *,
    mode: Literal["fp8", "nvfp4"] = "fp8",
    scaling: Literal["rowwise", "tensorwise"] = "rowwise",
    use_fast_accum: bool = False,
    keep_dense_fallback: bool = False,
    min_features: int = 1024,
    exclude_edge_blocks: bool = True,
    exclude: Sequence[str] = DEFAULT_COSMOS3_QUANT_EXCLUDES,
    quant_und_pathway: bool = True,
) -> dict[str, Any]:
    """Replace Cosmos3 heavy projections layer-by-layer with step-selective FP8 or NVFP4.

    Performs immediate per-module replacement so 64B Cosmos3-Super never
    materializes both BF16 and FP8 weights simultaneously on H200.
    """
    layers = getattr(model, "layers", ())
    num_layers = len(layers)
    excludes = list(exclude)
    if exclude_edge_blocks and num_layers >= 1:
        excludes.extend(["layers.0.", f"layers.{num_layers - 1}."])
    if not quant_und_pathway:
        excludes.extend([".self_attn.to_q", ".self_attn.to_k", ".self_attn.to_v", ".self_attn.to_out", ".mlp."])
    exclude_tuple = tuple(dict.fromkeys(excludes))

    replaced = 0
    skipped = 0

    def _is_excluded(full_name: str) -> bool:
        return any(token in full_name for token in exclude_tuple)

    def _visit(parent: nn.Module, prefix: str) -> None:
        nonlocal replaced, skipped
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if (
                    _is_excluded(full_name)
                    or min(child.in_features, child.out_features) < int(min_features)
                    or child.in_features % 16 != 0
                    or child.out_features % 16 != 0
                ):
                    skipped += 1
                    continue
                if mode == "nvfp4":
                    if child.in_features % 32 != 0:
                        skipped += 1
                        continue
                    q_mod: nn.Module = StepSelectiveNVFP4Linear(
                        child.weight,
                        child.bias,
                        keep_dense_fallback=keep_dense_fallback,
                    )
                else:
                    q_mod = StepSelectiveFloat8Linear(
                        child.weight,
                        child.bias,
                        keep_dense_fallback=keep_dense_fallback,
                        scaling=scaling,
                        use_fast_accum=use_fast_accum,
                    )
                setattr(parent, name, q_mod)
                del child
                replaced += 1
                if not keep_dense_fallback and replaced % 32 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif child is not None:
                _visit(child, full_name)

    _visit(model, "")
    if not keep_dense_fallback and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "mode": mode,
        "quantized_linears": replaced,
        "dense_bf16_linears": skipped,
        "scaling": scaling if mode == "fp8" else "nvfp4-1x16",
        "use_fast_accum": bool(use_fast_accum) if mode == "fp8" else None,
        "keep_dense_fallback": bool(keep_dense_fallback),
        "exclude_edge_blocks": bool(exclude_edge_blocks),
        "quant_und_pathway": bool(quant_und_pathway),
    }


def set_cosmos3_step_precision(model: nn.Module, *, use_low_precision: bool) -> int:
    """Toggle step-selective FP8/NVFP4 modules between low-precision and BF16 boundary mode."""
    toggled = 0
    flag = bool(use_low_precision)
    for module in model.modules():
        if isinstance(module, (StepSelectiveFloat8Linear, StepSelectiveNVFP4Linear)):
            module.low_precision_enabled = flag
            toggled += 1
    return toggled


def _resolve_cosmos3_fa3_usable(sm_major: int) -> bool:
    if int(sm_major) < 9:
        return False
    from worldfoundry.core.attention.backends.probe import attention_backend_capability

    return bool(attention_backend_capability("flash_attention_3").usable)


def cosmos3_sol_profile(
    tier: Literal["exact", "fp8_und_kv", "sol_high_fidelity", "sol_balanced", "sol_nvfp4"] | str,
    *,
    sm_major: int = 9,
    cuda_graph: Literal["off", "piecewise"] | None = None,
) -> Cosmos3AccelerationConfig:
    """Return a validated SOL-Engine preset for Cosmos3-Super / Cosmos3-Nano."""
    normalized = str(tier).strip().lower()
    fa3_ok = _resolve_cosmos3_fa3_usable(sm_major)
    attn_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3" if fa3_ok else "torch"

    if normalized == "exact":
        return Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend=attn_backend,
            fused_ops=True,
            fuse_projections=True,
            quantization="none",
            cache_algorithm="none",
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"lossless und static KV cache + {attn_backend} GQA + Triton fusion + piecewise graph",),
        )
    if normalized == "fp8_und_kv":
        return Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend=attn_backend,
            fused_ops=True,
            fuse_projections=True,
            quantization="fp8" if sm_major >= 9 else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_start_step=2,
            quant_dense_last_steps=2,
            cache_algorithm="none",
            cuda_graph=cuda_graph or "piecewise",
            notes=(
                f"und static KV cache + step-selective rowwise FP8(exact accum, zero-dup) + {attn_backend} + piecewise graph",
            ),
        )
    if normalized == "sol_high_fidelity":
        return Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend=attn_backend,
            fused_ops=True,
            fuse_projections=True,
            quantization="fp8" if sm_major >= 9 else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_start_step=2,
            quant_dense_last_steps=2,
            cache_algorithm="teacache",
            cache_threshold=0.08,
            cache_warmup_steps=5,
            cache_dense_last=3,
            cache_max_consecutive_skips=1,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"TeaCache(0.08) + step-selective FP8(exact accum) + und static KV + {attn_backend}",),
        )
    if normalized == "sol_balanced":
        return Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend=attn_backend,
            fused_ops=True,
            fuse_projections=True,
            quantization="fp8" if sm_major >= 9 else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_start_step=1,
            quant_dense_last_steps=1,
            cache_algorithm="teacache",
            cache_threshold=0.18,
            cache_warmup_steps=4,
            cache_dense_last=2,
            cache_max_consecutive_skips=2,
            cuda_graph=cuda_graph or "piecewise",
            notes=(
                f"TeaCache(0.18) + step-selective FP8(exact accum) + und static KV + {attn_backend} + piecewise graph",
            ),
        )
    if normalized == "sol_nvfp4":
        return Cosmos3AccelerationConfig(
            und_static_kv_cache=True,
            attention_backend=attn_backend,
            fused_ops=True,
            fuse_projections=True,
            quantization="nvfp4" if sm_major >= 10 else ("fp8" if sm_major >= 9 else "none"),
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            keep_dense_fallback=False,
            quant_exclude_edge_blocks=True,
            quant_start_step=2,
            quant_dense_last_steps=2,
            cache_algorithm="teacache",
            cache_threshold=0.14,
            cache_warmup_steps=4,
            cache_dense_last=2,
            cache_max_consecutive_skips=2,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"NVFP4(SM100)/FP8(SM90) step-selective + TeaCache + und static KV + {attn_backend}",),
        )
    raise ValueError(
        f"unsupported Cosmos3 acceleration tier {tier!r}; "
        "expected one of: exact, fp8_und_kv, sol_high_fidelity, sol_balanced, sol_nvfp4"
    )


def accelerate_cosmos3(
    model: nn.Module,
    config: Cosmos3AccelerationConfig | str,
    *,
    sm_major: int | None = None,
) -> dict[str, Any]:
    """Apply the complete Cosmos3-Super SOL-Engine optimization stack cleanly."""
    param = next(model.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    detected_sm = (
        torch.cuda.get_device_capability(param.device)[0]
        if is_cuda and param is not None
        else (sm_major if sm_major is not None else 0)
    )
    resolved = cosmos3_sol_profile(config, sm_major=detected_sm) if isinstance(config, str) else config
    fallbacks: list[str] = []

    # 1. Exact lossless und_seq static KV cache.
    if resolved.und_static_kv_cache:
        model._und_static_kv_cache = Cosmos3UndStaticKVCache()
    else:
        model._und_static_kv_cache = None

    # 2. Lossless projection fusion BEFORE quantization so FP8 quantizes single GEMMs.
    fusion_receipt: dict[str, int] = {"gen_qkv_fused": 0, "swiglu_gate_up_fused": 0}
    if resolved.fuse_projections:
        fusion_receipt = fuse_cosmos3_projections(model, release_originals=True)

    # 3. Attention backend + Triton fused ops across all layers.
    effective_attn = resolved.attention_backend
    if effective_attn == "flash_attention_3" and not (is_cuda and _resolve_cosmos3_fa3_usable(detected_sm)):
        effective_attn = "torch"
        if is_cuda:
            fallbacks.append("attention:flash_attention_3->torch")

    for layer in getattr(model, "layers", ()):
        layer._use_fused_ops = bool(resolved.fused_ops)
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.processor.attention_backend = effective_attn
            attn.processor.use_fused_ops = bool(resolved.fused_ops)
            attn.processor.fa3_calls = 0
            attn.processor.sdpa_calls = 0

    # 4. Step-selective quantization (SM90 FP8 use_fast_accum=False or SM100 NVFP4 hook).
    quant_mode = resolved.quantization
    if quant_mode == "auto":
        if is_cuda and param is not None and _nvfp4_hardware_eligible(param.device):
            quant_mode = "nvfp4"
        elif is_cuda and param is not None and _fp8_hardware_eligible(param.device):
            quant_mode = "fp8"
        else:
            quant_mode = "none"

    quant_receipt: dict[str, Any] = {"mode": "none", "quantized_linears": 0}
    if quant_mode in {"fp8", "nvfp4"}:
        quant_receipt = quantize_cosmos3_step_selective(
            model,
            mode=quant_mode,
            scaling=resolved.fp8_scaling,
            use_fast_accum=resolved.fp8_use_fast_accum,
            keep_dense_fallback=resolved.keep_dense_fallback,
            min_features=resolved.quant_min_features,
            exclude_edge_blocks=resolved.quant_exclude_edge_blocks,
            exclude=resolved.quant_exclude,
            quant_und_pathway=resolved.quant_und_pathway,
        )
        model._cosmos3_step_quant_policy = {
            "mode": quant_mode,
            "start_step": int(resolved.quant_start_step),
            "dense_last_steps": int(resolved.quant_dense_last_steps),
        }
    else:
        model._cosmos3_step_quant_policy = None

    # 5. Calibrated TeaCache / FirstBlock gen-stream caching policy.
    if resolved.cache_algorithm != "none":
        model._cosmos3_gen_cache_spec = {
            "algorithm": resolved.cache_algorithm,
            "threshold": float(resolved.cache_threshold),
            "warmup_steps": int(resolved.cache_warmup_steps),
            "dense_last": int(resolved.cache_dense_last),
            "max_consecutive_skips": int(resolved.cache_max_consecutive_skips),
            "coefficients": tuple(resolved.cache_teacache_coefficients),
        }
        model._cosmos3_gen_caches = {}
    else:
        model._cosmos3_gen_cache_spec = None
        model._cosmos3_gen_caches = {}

    # 6. Piecewise CUDA Graph runner for gen-stream tail segment.
    if resolved.cuda_graph == "piecewise":
        model._cosmos3_piecewise_graph = PiecewiseBlockGraphRunner(
            warmup=resolved.cuda_graph_warmup,
            max_graphs_per_segment=resolved.cuda_graph_max_graphs,
            clone_outputs=resolved.cuda_graph_clone_outputs,
            enabled=True,
        )
    else:
        model._cosmos3_piecewise_graph = None

    model._cosmos3_fa3_replay_calls = 0
    model._cosmos3_fp8_replay_calls = 0
    model._cosmos3_nvfp4_replay_calls = 0
    model._cosmos3_bf16_replay_calls = 0

    receipt: dict[str, Any] = {
        "engine": "worldfoundry-sol-cosmos3",
        "config": resolved.to_dict(),
        "und_static_kv_cache": {"enabled": bool(resolved.und_static_kv_cache)},
        "projection_fusion": fusion_receipt,
        "attention": {"backend": effective_attn, "fused_ops": bool(resolved.fused_ops)},
        "quantization": quant_receipt,
        "gen_cache": dict(model._cosmos3_gen_cache_spec) if model._cosmos3_gen_cache_spec else {"algorithm": "none"},
        "cuda_graph": {"mode": resolved.cuda_graph},
        "fallbacks": fallbacks,
    }
    model._cosmos3_acceleration_receipt = receipt
    return receipt


def cosmos3_runtime_receipt(model: nn.Module) -> dict[str, Any]:
    """Collect auditable live runtime receipts from an accelerated Cosmos3 model."""
    base = dict(
        getattr(model, "_cosmos3_acceleration_receipt", {"engine": "worldfoundry-sol-cosmos3", "fallbacks": []})
    )
    fallbacks = list(base.get("fallbacks", []))
    fa3_calls = 0
    sdpa_calls = 0
    for layer in getattr(model, "layers", ()):
        proc = getattr(getattr(layer, "self_attn", None), "processor", None)
        if proc is not None:
            fa3_calls += int(getattr(proc, "fa3_calls", 0))
            sdpa_calls += int(getattr(proc, "sdpa_calls", 0))

    fp8_calls = 0
    nvfp4_calls = 0
    bf16_boundary_calls = 0
    fp8_fallback_calls = 0
    for mod in model.modules():
        if isinstance(mod, StepSelectiveFloat8Linear):
            fp8_calls += mod.step_fp8_calls
            bf16_boundary_calls += mod.step_bf16_calls
            fp8_fallback_calls += mod.dense_fallback_calls
        elif isinstance(mod, StepSelectiveNVFP4Linear):
            nvfp4_calls += mod.step_nvfp4_calls
            bf16_boundary_calls += mod.step_bf16_calls
            fp8_fallback_calls += mod.dense_fallback_calls
    if fp8_fallback_calls > 0:
        fallbacks.append(f"quantization:dense_fallback_calls={fp8_fallback_calls}")

    und_cache = getattr(model, "_und_static_kv_cache", None)
    gen_caches = getattr(model, "_cosmos3_gen_caches", {}) or {}
    piecewise = getattr(model, "_cosmos3_piecewise_graph", None)
    graph_report = piecewise.report() if piecewise is not None else None
    if graph_report is not None:
        failed = sum(int(seg.get("capture_failed", 0)) for seg in graph_report.get("segments", {}).values())
        if failed > 0:
            fallbacks.append(f"cuda_graph:capture_failed={failed}")

    base["fallbacks"] = fallbacks
    base["runtime"] = {
        "attention_fa3_calls": fa3_calls + int(getattr(model, "_cosmos3_fa3_replay_calls", 0)),
        "attention_sdpa_calls": sdpa_calls,
        "quant_step_fp8_calls": fp8_calls + int(getattr(model, "_cosmos3_fp8_replay_calls", 0)),
        "quant_step_nvfp4_calls": nvfp4_calls + int(getattr(model, "_cosmos3_nvfp4_replay_calls", 0)),
        "quant_step_bf16_boundary_calls": bf16_boundary_calls + int(getattr(model, "_cosmos3_bf16_replay_calls", 0)),
        "quant_unexpected_fallbacks": fp8_fallback_calls,
        "und_static_kv_cache": und_cache.receipt() if und_cache is not None else {"enabled": False},
        "gen_cache_branches": {branch: cache.receipt() for branch, cache in gen_caches.items()},
        "piecewise_graph": graph_report,
    }
    return base


__all__ = [
    "DEFAULT_COSMOS3_QUANT_EXCLUDES",
    "Cosmos3AccelerationConfig",
    "Cosmos3GenStreamCache",
    "Cosmos3UndStaticKVCache",
    "StepSelectiveFloat8Linear",
    "StepSelectiveNVFP4Linear",
    "accelerate_cosmos3",
    "cosmos3_runtime_receipt",
    "cosmos3_sol_profile",
    "fuse_cosmos3_projections",
    "quantize_cosmos3_step_selective",
    "set_cosmos3_step_precision",
]
