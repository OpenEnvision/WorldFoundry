"""Reversible, explicitly selected FP8 projections for native diffusion graphs.

This adapter reuses the core scaled-mm provider. It retains original modules
and their dense storage for exact removal, so it is a compute optimization,
not a weight-memory optimization. Placement and weights remain fixed until
removal; quantized checkpoint loading and training belong to other workflows.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
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
from worldfoundry.core.model_loading.policy import OffloadMode, QuantizationMode


@dataclass(frozen=True)
class SelectiveFP8Policy:
    """Qualified module-path globs; no implicit whole-model conversion."""

    include: Sequence[str]
    exclude: Sequence[str] = ()
    min_features: int = 1024
    scaling: str = "rowwise"
    use_fast_accum: bool = False

    def __post_init__(self) -> None:
        for key in ("include", "exclude"):
            patterns = getattr(self, key)
            if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Sequence):
                raise TypeError(f"selective_fp8 {key} must be a sequence of module-path patterns")
            if any(not isinstance(value, str) or not value.strip() for value in patterns):
                raise ValueError(f"selective_fp8 {key} patterns must be nonempty strings")
            if len(set(patterns)) != len(patterns):
                raise ValueError(f"selective_fp8 {key} patterns must be unique")
            object.__setattr__(self, key, tuple(patterns))
        if not self.include:
            raise ValueError("selective_fp8 requires explicit include patterns")
        if type(self.min_features) is not int or self.min_features <= 0:
            raise ValueError("selective_fp8 min_features must be a positive integer")
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


def _validate_model(model: nn.Module, policy: Any) -> str:
    family = type(model).__module__.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    if family == "wan":
        from ..models.networks.wan.model import DiTBlock, WanModel

        supported_models, supported_blocks = {WanModel}, {DiTBlock}
    elif family == "sana":
        from ..models.networks.sana.sana import Sana, SanaBlock
        from ..models.networks.sana.sana_multi_scale import SanaMS, SanaMSBlock
        from ..models.networks.sana.sana_multi_scale_video import SanaMSVideo, SanaVideoMSBlock

        supported_models, supported_blocks = {Sana, SanaMS, SanaMSVideo}, {SanaBlock, SanaMSBlock, SanaVideoMSBlock}
    else:
        supported_models, supported_blocks = set(), set()
    if type(model) not in supported_models:
        raise ValueError(f"selective_fp8 does not support checkpoint graph {type(model).__name__}")
    if any(child.training for child in model.modules()):
        raise ValueError("selective_fp8 requires model.eval()")
    if any(type(block) not in supported_blocks for block in model.blocks):
        raise ValueError("selective_fp8 requires the native default block classes")
    if any(getattr(child, "_worldfoundry_quantization_layer", False) for child in model.modules()):
        raise ValueError("selective_fp8 cannot coexist with an existing quantization transform")
    if any("Cached" in type(child).__name__ or "Causal" in type(child).__name__ for child in model.modules()):
        raise ValueError("selective_fp8 has not been validated with stateful or causal graphs")
    if any(
        getattr(child, "attention_backend", None)
        in {"sol", "sol_attn", "sage", "sage_attention", "sage_attention_3", "cudnn_fp8"}
        for child in model.modules()
    ):
        raise ValueError("selective_fp8 has not been validated with approximate attention backends")
    for attribute in (
        "_worldfoundry_layerwise_cpu_offload_handle",
        "_worldfoundry_sequence_parallel",
        "_worldfoundry_static_cross_kv",
        "_worldfoundry_easycache_config",
    ):
        if getattr(model, attribute, None) is not None:
            raise ValueError(f"selective_fp8 has not been validated with {attribute}")
    if policy is not None:
        if policy.quantization.mode is not QuantizationMode.NONE:
            raise ValueError("selective_fp8 cannot coexist with the generic quantization policy")
        if policy.offload.mode is not OffloadMode.NONE:
            raise ValueError("selective_fp8 requires resident weights")
        conflicts = [
            key
            for key in (
                "cuda_graph",
                "sequence_parallel",
                "sp_degree",
                "device_map",
                "static_cross_kv",
                "feature_cache",
                "teacache",
                "magcache",
                "adacache",
                "taylorseer",
                "blocktaylorseer",
                "custom",
                "approximate_attention",
                "inplace_residual",
            )
            if policy.options.get(key)
        ]
        if policy.compile:
            conflicts.append("compile")
        plugins = policy.options.get("accelerations", {})
        if isinstance(plugins, Mapping):
            if plugins.get("easycache") is not None and plugins.get("easycache") is not False:
                conflicts.append("easycache")
            fusion = plugins.get("sana_block_fusion", {})
            if isinstance(fusion, Mapping) and (fusion.get("fuse_layer_norm") or fusion.get("repack_tokens")):
                conflicts.append("approximate sana_block_fusion")
            scoped_attention = plugins.get("attention_policy", {})
            if isinstance(scoped_attention, Mapping):
                for provider in scoped_attention.values():
                    backend = provider.get("backend") if isinstance(provider, Mapping) else provider
                    if backend in {
                        "sol",
                        "sol_attn",
                        "sol_attention",
                        "sage",
                        "sage_attention",
                        "sage3",
                        "sage_attention_3",
                        "cudnn_fp8",
                    }:
                        conflicts.append("approximate attention_policy")
        if conflicts:
            raise ValueError(f"selective_fp8 conflicts with unvalidated options: {conflicts}")
    return family


def _supported_path(path: str, family: str) -> bool:
    if family == "wan":
        return (
            re.fullmatch(r"blocks\.\d+\.(?:(?:self_attn|cross_attn)\.(?:q|k|v|o|qkv)|ffn\.(?:0|2))", path) is not None
        )
    return (
        re.fullmatch(
            r"blocks\.\d+\.(?:(?:attn|cross_attn)\.(?:qkv|q_linear|kv_linear|proj)|"
            r"mlp\.(?:fc1|fc2|(?:inverted_conv|point_conv)\.linear))",
            path,
        )
        is not None
    )


def prepare_selective_fp8(model: nn.Module, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    """Plan all replacements before mutations, then install atomically."""
    config = SelectiveFP8Policy(**options)
    family = _validate_model(model, policy)
    modules = dict(model.named_modules())
    matches = [
        (path, child)
        for path, child in modules.items()
        if any(fnmatchcase(path, pattern) for pattern in config.include)
        and not any(fnmatchcase(path, pattern) for pattern in config.exclude)
        and isinstance(child, nn.Linear)
    ]
    selected = []
    for path, child in matches:
        if not _supported_path(path, family):
            raise ValueError(f"selective_fp8 projection {path!r} is outside supported native blocks")
        if type(child) is not nn.Linear or hasattr(child, "parametrizations"):
            raise ValueError(f"selective_fp8 projection {path!r} is not a plain nn.Linear")
        if min(child.in_features, child.out_features) < config.min_features or (
            child.in_features % 16 or child.out_features % 16
        ):
            continue
        if family == "wan" and path.split(".")[2] in {"self_attn", "cross_attn"}:
            from ..models.networks.wan.model import CrossAttentionProcessor, SelfAttentionProcessor

            owner = modules[".".join(path.split(".")[:3])]
            expected_processor = (
                SelfAttentionProcessor if path.split(".")[2] == "self_attn" else CrossAttentionProcessor
            )
            if type(owner.processor) is not expected_processor:
                raise ValueError(f"selective_fp8 projection {path!r} requires the native default attention processor")
        for ancestor_path in ("", *[".".join(path.split(".")[:index]) for index in range(1, len(path.split(".")) + 1)]):
            ancestor = modules[ancestor_path]
            if ancestor._forward_hooks or ancestor._forward_pre_hooks or ancestor._backward_hooks:
                raise ValueError(f"selective_fp8 projection {path!r} has hooks in its call path")
            if any(token in type(ancestor).__name__ for token in ("Triton", "Cached", "Causal", "PAG")):
                raise ValueError(f"selective_fp8 projection {path!r} belongs to an unsupported fused/stateful graph")
            if "forward" in ancestor.__dict__:
                raise ValueError(f"selective_fp8 projection {path!r} has a customized forward in its call path")
        if child.weight.device.type == "meta" or child.weight.dtype not in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }:
            raise ValueError(f"selective_fp8 projection {path!r} requires materialized floating-point weights")
        if not bool(torch.isfinite(child.weight).all()) or (
            child.bias is not None and not bool(torch.isfinite(child.bias).all())
        ):
            raise ValueError(f"selective_fp8 projection {path!r} has nonfinite weights")
        selected.append((path, child))
    for pattern in config.include:
        if not any(fnmatchcase(path, pattern) for path, _ in selected):
            raise ValueError(f"selective_fp8 include pattern {pattern!r} matched no eligible projections")

    # Canonical Parameters, module aliases and distinct views of the same
    # allocation are all unsafe to replace independently.
    aliases: dict[int, int] = {}
    for _, child in model.named_modules(remove_duplicate=False):
        aliases[id(child)] = aliases.get(id(child), 0) + 1
    storages: dict[tuple[str, int], int] = {}
    for _, value in model.named_parameters(remove_duplicate=False):
        if value.device.type != "meta":
            key = (str(value.device), value.untyped_storage().data_ptr())
            storages[key] = storages.get(key, 0) + 1
    for path, child in selected:
        tensors = (child.weight,) if child.bias is None else (child.weight, child.bias)
        if aliases[id(child)] > 1 or any(
            storages[(str(value.device), value.untyped_storage().data_ptr())] > 1 for value in tensors
        ):
            raise ValueError(f"selective_fp8 projection {path!r} has tied modules or shared parameter storage")

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
