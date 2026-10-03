"""Reversible, explicitly selected FP8 projections for native diffusion graphs.

This adapter reuses the core scaled-mm provider. It retains original modules
and their dense storage for exact removal, so it is a compute optimization,
not a weight-memory optimization. Placement and weights remain fixed until
removal; quantized checkpoint loading and training belong to other workflows.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from worldfoundry.core.acceleration.plugins import AccelerationHandle, PreparedAcceleration
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    _autocast_linear_input,
    _fp8_fallback_reason,
    _fp8_linear_eligible,
)

from .projection_selection import ProjectionSelection, select_native_projections


@dataclass(frozen=True)
class SelectiveFP8Policy(ProjectionSelection):
    """Qualified module-path globs; no implicit whole-model conversion."""

    scaling: str = "rowwise"
    use_fast_accum: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scaling not in {"rowwise", "tensorwise", "auto"}:
            raise ValueError("selective_fp8 scaling must be rowwise, tensorwise or auto")
        if type(self.use_fast_accum) is not bool:
            raise TypeError("selective_fp8 use_fast_accum must be a boolean")


class _FixedPlacementFloat8Linear(Float8Linear):
    """The core provider with a retained source and explicit mutation guards."""

    def __init__(self, source: nn.Linear, config: SelectiveFP8Policy) -> None:
        super().__init__(
            source.weight,
            source.bias,
            keep_dense_fallback=False,
            scaling=config.scaling,
            use_fast_accum=config.use_fast_accum,
        )
        # Dense fallback aliases the retained module rather than allocating a
        # second dense weight. The original is deliberately not a child module.
        object.__setattr__(self, "_source", source)
        self.weight = source.weight.detach()
        self.bias = None if source.bias is None else source.bias.detach()
        self._source_fingerprint = self._fingerprint()
        self.eval()

    def _fingerprint(self):
        entries = []
        for value in (self._source.weight, self._source.bias):
            if value is None:
                entries.append(None)
                continue
            # Inference tensors have no version counter. Their weights, like
            # any raw .data writes, must obey the documented immutability rule.
            version = None if value.is_inference() else value._version
            entries.append((id(value), value.data_ptr(), value.device, value.dtype, version))
        return tuple(entries)

    def _apply(self, fn, recurse: bool = True):
        raise RuntimeError("uninstall selective_fp8 before changing model placement or dtype")

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("uninstall selective_fp8 before training")
        return super().train(False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            raise RuntimeError("selective_fp8 has not been validated with torch.compile")
        if input.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("selective_fp8 has not been validated with CUDA Graph capture")
        if torch.is_grad_enabled():
            raise RuntimeError("selective_fp8 requires no-grad inference; uninstall it before training")
        if self._fingerprint() != self._source_fingerprint:
            raise RuntimeError("selective_fp8 source weights changed; uninstall and reinstall after weight edits")
        effective_input = _autocast_linear_input(input)
        if input.numel() and not (
            self.low_precision_enabled
            and _fp8_linear_eligible(effective_input, self.out_features, self._hardware_eligible)
        ):
            # Keep the original operator, Parameter metadata and stride
            # behavior for quality-disabled, small or unsupported workloads.
            # Merely calling F.linear on detached buffers can choose a
            # different CUDA reduction and change a few BF16 output bits.
            output = self._source(input)
            self.dense_fallback_calls += 1
            self.request_dense_fallback_calls += 1
            self.last_fallback_reason = _fp8_fallback_reason(
                effective_input,
                self.out_features,
                self._hardware_eligible,
                enabled=self.low_precision_enabled,
            )
            return output
        return super().forward(input)


def prepare_selective_fp8(model: nn.Module, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    """Plan all replacements before mutations, then install atomically."""
    config = SelectiveFP8Policy(**options)
    modules, selected = select_native_projections(model, config, policy)

    seams = {
        "diffusion.approximation",
        "diffusion.precision_cache",
        *(f"diffusion.projection.{path}" for path, _ in selected),
    }
    if any(re.fullmatch(r"blocks\.\d+\.cross_attn\.[kv]", path) for path, _ in selected):
        seams.add("wan.cross_attention.projections")

    def activate() -> AccelerationHandle:
        replacements = [(path, original, _FixedPlacementFloat8Linear(original, config)) for path, original in selected]
        installed = []
        missing = object()
        original_apply = model.__dict__.get("_apply", missing)
        original_train = model.__dict__.get("train", missing)
        train_model = model.train

        def fixed_placement(*args, **kwargs):
            raise RuntimeError("uninstall selective_fp8 before changing model placement or dtype")

        def inference_train(mode: bool = True):
            if mode is True:
                raise RuntimeError("uninstall selective_fp8 before training")
            # The native method validates non-boolean arguments before
            # mutation. eval()/train(False) retain their usual behavior.
            return train_model(mode)

        def undo() -> None:
            for parent, name, original in reversed(installed):
                setattr(parent, name, original)
            installed.clear()
            if original_apply is missing:
                model.__dict__.pop("_apply", None)
            else:
                model._apply = original_apply
            if original_train is missing:
                model.__dict__.pop("train", None)
            else:
                model.train = original_train

        try:
            for path, original, replacement in replacements:
                parent_path, _, name = path.rpartition(".")
                parent = modules[parent_path]
                setattr(parent, name, replacement)
                installed.append((parent, name, original))
            model._apply = fixed_placement
            model.train = inference_train
        except BaseException:
            undo()
            raise
        return AccelerationHandle(
            "selective_fp8",
            {
                "approximate": True,
                "numerical_contract": "dynamic-rowwise-fp8" if config.scaling == "rowwise" else "dynamic-fp8",
                "execution": "runtime-pending",
                "provider": "torch._scaled_mm",
                "modules": [path for path, _ in selected],
                "scaling": config.scaling,
                "use_fast_accum": config.use_fast_accum,
                "dense_fallback_retained": True,
                "dense_storage": "shared-with-retained-original-modules",
                "placement": "fixed-until-uninstall",
            },
            undo,
        )

    return PreparedAcceleration("selective_fp8", frozenset(seams), activate)


__all__ = ["SelectiveFP8Policy", "prepare_selective_fp8"]
