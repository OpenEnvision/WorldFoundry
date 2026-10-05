"""Reversible native Wan cross-KV projection fusion with canonical weights.

Packed tensors are derived execution data, never checkpoint parameters. The
original K/V modules remain available for unloading and small-context bypass.
Fusion changes GEMM scheduling, so its numerical contract permits reduction
rounding differences; it does not quantize weights or attention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.core.acceleration.plugins import AccelerationHandle, PreparedAcceleration


@dataclass(frozen=True)
class CrossKVFusionConfig:
    min_tokens: int = 256
    include_image: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.min_tokens, bool) or not isinstance(self.min_tokens, int) or self.min_tokens < 0:
            raise ValueError("cross-KV min_tokens must be a nonnegative integer")
        if not isinstance(self.include_image, bool):
            raise TypeError("cross-KV include_image must be a bool")


def _source_stamp(key: nn.Linear, value: nn.Linear) -> tuple[Any, ...]:
    stamp = []
    for projection in (key, value):
        if (
            type(projection) is not nn.Linear
            or "forward" in projection.__dict__
            or any(
                getattr(projection, name)
                for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
            )
        ):
            raise ValueError("cross-KV fusion requires unhooked native nn.Linear projections")
        if projection.training:
            raise ValueError("cross-KV fusion requires eval projections")
        for parameter in (projection.weight, projection.bias):
            if parameter is None:
                stamp.append(None)
                continue
            try:
                version = parameter._version
            except RuntimeError as error:
                raise ValueError("cross-KV fusion requires versioned projection parameters") from error
            stamp.append((id(parameter), version, parameter.shape, parameter.dtype, parameter.device))
    if key.weight.shape != value.weight.shape or key.weight.device != value.weight.device:
        raise ValueError("cross-KV projections must have equal shapes and devices")
    if key.weight.dtype != value.weight.dtype or key.weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("cross-KV fusion requires matching FP16, BF16 or FP32 projections")
    if key.weight.device.type not in {"cpu", "cuda"}:
        raise ValueError("cross-KV fusion requires materialized CPU or CUDA weights")
    if (key.bias is None) != (value.bias is None):
        raise ValueError("cross-KV projection biases must match")
    for projection in (key, value):
        if projection.bias is not None and (
            projection.bias.dtype != projection.weight.dtype or projection.bias.device != projection.weight.device
        ):
            raise ValueError("cross-KV bias dtype/device must match its weight")
    return tuple(stamp)


class _PackedKV:
    def __init__(self, key: nn.Linear, value: nn.Linear) -> None:
        self.stamp: tuple[Any, ...] | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        # Keep source identities alive until refresh, so allocator reuse after
        # parameter replacement cannot impersonate the previous stamp.
        self.sources: tuple[torch.Tensor | None, ...] = ()
        self.refresh(key, value)

    def refresh(self, key: nn.Linear, value: nn.Linear) -> bool:
        stamp = _source_stamp(key, value)
        if self.stamp == stamp:
            return False
        # Publish both tensors only after successful allocation. Retaining the
        # canonical modules also keeps load_state_dict and state_dict unchanged.
        with torch.no_grad():
            weight = torch.cat((key.weight, value.weight), dim=0).detach()
            bias = None if key.bias is None else torch.cat((key.bias, value.bias), dim=0).detach()
        self.weight, self.bias, self.stamp = weight, bias, stamp
        self.sources = (key.weight, key.bias, value.weight, value.bias)
        return True


class CrossKVFusionState:
    """One attention module's derived tensors and live execution counters."""

    def __init__(self, attention: nn.Module, config: CrossKVFusionConfig, receipts: dict[str, int]) -> None:
        self.config = config
        self.receipts = receipts
        self.routes = {False: _PackedKV(attention.k, attention.v)}
        if config.include_image and attention.has_image_input:
            self.routes[True] = _PackedKV(attention.k_img, attention.v_img)

    def project(self, attention: nn.Module, context: torch.Tensor, *, image: bool = False):
        if torch.compiler.is_compiling():
            raise RuntimeError("cross-KV fusion has not been validated with torch.compile")
        if context.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cross-KV fusion has not been validated with CUDA Graph capture")
        if attention.training or torch.is_grad_enabled():
            raise RuntimeError("cross-KV fusion supports eval inference under no_grad/inference_mode only")
        key, value = (attention.k_img, attention.v_img) if image else (attention.k, attention.v)
        packed = self.routes.get(image)
        if packed is None or context.shape[-2] < self.config.min_tokens:
            self.receipts["dense_projection_pairs"] += 1
            return key(context), value(context)
        if packed.refresh(key, value):
            self.receipts["weight_refreshes"] += 1
        projected = F.linear(context, packed.weight, packed.bias)
        self.receipts["image_packed_calls" if image else "text_packed_calls"] += 1
        return projected.chunk(2, dim=-1)

    def clear(self) -> None:
        self.routes.clear()


def prepare_wan_cross_kv_fusion(model: Any, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    """Validate a native eager graph and prepare an atomic removable adapter.

    Supported state changes use normal parameter operations or checkpoint
    loading. Editing parameters through ``.data`` is outside this contract;
    reinstall after such edits. Existing static-KV caches still require their
    normal invalidation after any weight or conditioning change.
    """
    from ..models.networks.wan.model import CrossAttention, CrossAttentionProcessor, DiTBlock, WanModel
    from .static_cross_kv import StaticCrossKVProcessor

    config = CrossKVFusionConfig(**options)
    if type(model) not in {WanModel, DiTBlock, CrossAttention}:
        raise ValueError("cross-KV fusion requires a native WanModel, DiTBlock or CrossAttention")
    if model.training:
        raise ValueError("cross-KV fusion requires model.eval()")
    if getattr(model, "_worldfoundry_sequence_parallel", None) is not None:
        raise ValueError("cross-KV fusion has not been validated with sequence parallelism")
    if getattr(model, "_worldfoundry_layerwise_cpu_offload_handle", None) is not None:
        raise ValueError("cross-KV fusion requires resident weights")
    if getattr(model, "_worldfoundry_compile_runtime", None) is not None:
        raise ValueError("cross-KV fusion has not been validated with compile")
    if policy is not None:
        conflicts = [
            key
            for key in ("cuda_graph", "sequence_parallel", "sp_degree", "device_map", "approximate_attention")
            if policy.options.get(key)
        ]
        if policy.compile:
            conflicts.append("compile")
        if policy.offload.mode.value != "none":
            conflicts.append("offload")
        if conflicts:
            raise ValueError(f"cross-KV fusion conflicts with unvalidated options: {conflicts}")
    targets = [module for module in model.modules() if type(module) is CrossAttention]
    if not targets:
        raise ValueError("cross-KV fusion found no native Wan cross-attention modules")
    for attention in targets:
        if attention.training or getattr(attention, "_worldfoundry_cross_kv_fusion", None) is not None:
            raise ValueError("cross-KV fusion requires idle eval modules without an existing adapter")
        processor = attention.processor
        if type(processor) is StaticCrossKVProcessor:
            if processor._cache._entries or processor._cache._condition_entries:
                raise ValueError("invalidate existing static-KV entries before installing cross-KV fusion")
            processor = processor._inner
        if type(processor) is not CrossAttentionProcessor:
            raise ValueError("cross-KV fusion requires the native default or static-KV processor")
        pairs = [(attention.k, attention.v)]
        if config.include_image and attention.has_image_input:
            pairs.append((attention.k_img, attention.v_img))
        for key, value in pairs:
            _source_stamp(key, value)
            if key.in_features != attention.dim or key.out_features != attention.dim:
                raise ValueError("cross-KV fusion requires native square projection geometry")

    def activate() -> AccelerationHandle:
        receipts = {"text_packed_calls": 0, "image_packed_calls": 0, "dense_projection_pairs": 0, "weight_refreshes": 0}
        states = [(attention, CrossKVFusionState(attention, config, receipts)) for attention in targets]

        def undo() -> None:
            caches = {}
            for attention, state in states:
                if getattr(attention, "_worldfoundry_cross_kv_fusion", None) is state:
                    delattr(attention, "_worldfoundry_cross_kv_fusion")
                    processor = attention.processor
                    if type(processor) is StaticCrossKVProcessor:
                        caches[id(processor._cache)] = processor._cache
                state.clear()
            # Cached packed results can differ in their GEMM rounding from
            # independent projections. Removing the adapter restores dense
            # scheduling on the next call, including a fresh cache miss.
            for cache in caches.values():
                cache.invalidate()

        try:
            for attention, state in states:
                attention._worldfoundry_cross_kv_fusion = state
        except BaseException:
            undo()
            raise
        return AccelerationHandle(
            "wan_cross_kv_fusion",
            {
                "approximate": False,
                "numerical_contract": "native-gemm-reduction-tolerance",
                "configured_modules": len(targets),
                "min_tokens": config.min_tokens,
                "include_image": config.include_image,
                "canonical_parameters_retained": True,
                "runtime": receipts,
            },
            undo,
        )

    seams = {"wan.cross_attention.projections"}
    if config.include_image:
        seams.add("wan.cross_attention.image_projections")
    return PreparedAcceleration("wan_cross_kv_fusion", frozenset(seams), activate)


__all__ = ["CrossKVFusionConfig", "prepare_wan_cross_kv_fusion"]
