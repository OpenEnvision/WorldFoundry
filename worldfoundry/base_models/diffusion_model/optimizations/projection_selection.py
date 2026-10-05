"""Shared native projection admission; validation never mutates the model."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import nn

from worldfoundry.core.model_loading.policy import OffloadMode, QuantizationMode


@dataclass(frozen=True)
class ProjectionSelection:
    include: Sequence[str]
    exclude: Sequence[str] = ()
    min_features: int = 1024

    def __post_init__(self):
        for key in ("include", "exclude"):
            patterns = getattr(self, key)
            if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Sequence):
                raise TypeError(f"projection {key} must be a sequence of module-path patterns")
            if any(not isinstance(value, str) or not value.strip() for value in patterns):
                raise ValueError(f"projection {key} patterns must be nonempty strings")
            if len(set(patterns)) != len(patterns):
                raise ValueError(f"projection {key} patterns must be unique")
            object.__setattr__(self, key, tuple(patterns))
        if not self.include:
            raise ValueError("projection selection requires explicit include patterns")
        if type(self.min_features) is not int or self.min_features <= 0:
            raise ValueError("projection min_features must be a positive integer")


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
        raise ValueError(f"projection quantization does not support checkpoint graph {type(model).__name__}")
    if any(child.training for child in model.modules()):
        raise ValueError("projection quantization requires model.eval()")
    if any(type(block) not in supported_blocks for block in model.blocks):
        raise ValueError("projection quantization requires the native default block classes")
    if any(getattr(child, "_worldfoundry_quantization_layer", False) for child in model.modules()):
        raise ValueError("projection quantization cannot coexist with an existing quantization transform")
    if any("Cached" in type(child).__name__ or "Causal" in type(child).__name__ for child in model.modules()):
        raise ValueError("projection quantization has not been validated with stateful or causal graphs")
    if any(
        getattr(child, "attention_backend", None)
        in {"sol", "sol_attn", "sage", "sage_attention", "sage_attention_3", "cudnn_fp8"}
        for child in model.modules()
    ):
        raise ValueError("projection quantization has not been validated with approximate attention backends")
    for attribute in (
        "_worldfoundry_layerwise_cpu_offload_handle",
        "_worldfoundry_sequence_parallel",
        "_worldfoundry_static_cross_kv",
        "_worldfoundry_easycache_config",
    ):
        if getattr(model, attribute, None) is not None:
            raise ValueError(f"projection quantization has not been validated with {attribute}")
    if policy is not None:
        if policy.quantization.mode is not QuantizationMode.NONE:
            raise ValueError("projection quantization cannot coexist with the generic quantization policy")
        if policy.offload.mode is not OffloadMode.NONE:
            raise ValueError("projection quantization requires resident weights")
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
            raise ValueError(f"projection quantization conflicts with unvalidated options: {conflicts}")
    return family


def _supported_path(path: str, family: str) -> bool:
    if family == "wan":
        return (
            re.fullmatch(r"blocks\.\d+\.(?:(?:self_attn|cross_attn)\.(?:q|k|v|o|qkv|k_img|v_img)|ffn\.(?:0|2))", path)
            is not None
        )
    return (
        re.fullmatch(
            r"blocks\.\d+\.(?:(?:attn|cross_attn)\.(?:qkv|q_linear|kv_linear|proj)|"
            r"mlp\.(?:fc1|fc2|(?:inverted_conv|point_conv)\.linear))",
            path,
        )
        is not None
    )


def select_native_projections(model, config, policy):
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
            raise ValueError(f"projection quantization projection {path!r} is outside supported native blocks")
        if type(child) is not nn.Linear or hasattr(child, "parametrizations"):
            raise ValueError(f"projection quantization projection {path!r} is not a plain nn.Linear")
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
                raise ValueError(
                    f"projection quantization projection {path!r} requires the native default attention processor"
                )
        # Replacing or bypassing this Linear would drop its hooks. Parent
        # modules remain in the call path, so their observation hooks survive.
        if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks:
            raise ValueError(f"projection quantization projection {path!r} has hooks on the replaced projection")
        for ancestor_path in ("", *[".".join(path.split(".")[:index]) for index in range(1, len(path.split(".")) + 1)]):
            ancestor = modules[ancestor_path]
            if any(token in type(ancestor).__name__ for token in ("Triton", "Cached", "Causal", "PAG")):
                raise ValueError(
                    f"projection quantization projection {path!r} belongs to an unsupported fused/stateful graph"
                )
            if "forward" in ancestor.__dict__:
                raise ValueError(
                    f"projection quantization projection {path!r} has a customized forward in its call path"
                )
        if child.weight.device.type == "meta" or child.weight.dtype not in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }:
            raise ValueError(
                f"projection quantization projection {path!r} requires materialized floating-point weights"
            )
        if not bool(torch.isfinite(child.weight).all()) or (
            child.bias is not None and not bool(torch.isfinite(child.bias).all())
        ):
            raise ValueError(f"projection quantization projection {path!r} has nonfinite weights")
        selected.append((path, child))
    for pattern in config.include:
        if not any(fnmatchcase(path, pattern) for path, _ in selected):
            raise ValueError(f"projection quantization include pattern {pattern!r} matched no eligible projections")

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
            raise ValueError(
                f"projection quantization projection {path!r} has tied modules or shared parameter storage"
            )

    return modules, selected
