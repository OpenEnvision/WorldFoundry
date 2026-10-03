"""Native diffusion plugin adapters; no kernels or request state live here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from worldfoundry.core.acceleration.plugins import (
    AccelerationHandle,
    AccelerationRegistry,
    PreparedAcceleration,
)


def _attribute_plan(
    name: str, targets: list[Any], attribute: str, value: Any, details: Mapping[str, Any], seams: frozenset[str]
) -> PreparedAcceleration:
    if not targets:
        raise ValueError(f"{name} has no supported modules in this checkpoint graph")
    missing = object()
    previous = [(target, target.__dict__.get(attribute, missing)) for target in targets]

    def undo() -> None:
        for target, original in previous:
            if original is missing:
                target.__dict__.pop(attribute, None)
            else:
                setattr(target, attribute, original)

    def activate() -> AccelerationHandle:
        try:
            for target in targets:
                setattr(target, attribute, value)
        except BaseException:
            undo()
            raise
        return AccelerationHandle(name, details, undo)

    return PreparedAcceleration(name, seams, activate)


def _sana_block_fusion(model: Any, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    from ..models.networks.sana.block_ops import SanaBlockFusionPolicy

    config = SanaBlockFusionPolicy(**options)
    targets = [child for child in model.modules() if hasattr(type(child), "_worldfoundry_block_fusion")]
    return _attribute_plan(
        "sana_block_fusion",
        targets,
        "_worldfoundry_block_fusion",
        config,
        {
            "configured_blocks": len(targets),
            "backend": config.backend,
            "min_elements": config.min_elements,
            "fuse_layer_norm": config.fuse_layer_norm,
            "repack_tokens": config.repack_tokens,
            "approximate": config.fuse_layer_norm or config.repack_tokens,
            "numerical_contract": "reduction-tolerance"
            if config.fuse_layer_norm or config.repack_tokens
            else "eager-pointwise-rounding",
        },
        frozenset(
            {"sana.block_glue"}
            | ({"diffusion.approximation"} if config.fuse_layer_norm or config.repack_tokens else set())
        ),
    )


def _easycache(model: Any, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    from worldfoundry.core.acceleration.easycache import EasyCacheConfig

    config = EasyCacheConfig(**options)
    if getattr(model, "training", False):
        raise ValueError("EasyCache requires model.eval()")
    if getattr(model, "_worldfoundry_sequence_parallel", None) is not None:
        raise ValueError("EasyCache has not been validated with sequence parallelism")
    if getattr(model, "_worldfoundry_layerwise_cpu_offload_handle", None) is not None or (
        policy is not None and policy.offload.mode.value != "none"
    ):
        raise ValueError("EasyCache has not been validated with offload; use resident weights")
    if policy is not None:
        incompatible = [
            key
            for key in (
                "cuda_graph",
                "sequence_parallel",
                "sp_degree",
                "device_map",
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
            incompatible.append("compile")
        if incompatible:
            raise ValueError(f"EasyCache conflicts with unvalidated options: {incompatible}")
    # Streaming/stateful/camera/control graphs need their own cache lifecycle.
    supported = type(model).__name__ in {"SanaMS", "SanaMSVideo", "WanModel"}
    if not supported or not type(model).__module__.startswith(
        "worldfoundry.base_models.diffusion_model.models.networks."
    ):
        raise ValueError(f"EasyCache does not support checkpoint graph {type(model).__name__}")
    if any(
        "Cached" in type(getattr(block, "attn", None)).__name__
        or "Cached" in type(getattr(block, "mlp", None)).__name__
        for block in getattr(model, "blocks", ())
    ):
        raise ValueError("EasyCache cannot skip blocks that update autoregressive state")
    return _attribute_plan(
        "easycache",
        [model],
        "_worldfoundry_easycache_config",
        config,
        {"approximate": config.threshold > 0, "seam": "embedded-token-block-stack", **config.options},
        frozenset({"diffusion.block_stack", "diffusion.precision_cache", "diffusion.approximation"}),
    )


def diffusion_acceleration_registry() -> AccelerationRegistry:
    """Return an extensible registry without loading optional provider packages."""
    from .kv_fusion import prepare_wan_cross_kv_fusion
    from .precision import prepare_selective_fp8

    registry = AccelerationRegistry()
    registry.register("sana_block_fusion", _sana_block_fusion)
    registry.register("easycache", _easycache)
    registry.register("attention_policy", _attention_policy)
    registry.register("selective_fp8", prepare_selective_fp8)
    registry.register("wan_cross_kv_fusion", prepare_wan_cross_kv_fusion)
    return registry


def install_diffusion_accelerations(model: Any, options: Mapping[str, Any], policy: Any = None):
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    if not isinstance(options, Mapping):
        raise TypeError("accelerations must be a mapping of plugin names to options")
    if policy is None:
        policy = RuntimePolicy()
    if not isinstance(policy, RuntimePolicy):
        raise TypeError("diffusion acceleration adapters require a RuntimePolicy context")
    policy = replace(policy, options={**policy.options, "accelerations": dict(options)})
    return diffusion_acceleration_registry().install(model, options, policy)


def acceleration_policy(context: Any):
    """Honor component overrides and validate inference-only plugin builds."""
    from ..components import validate_runtime_policy_for_purpose

    options = dict(context.policy.options)
    if "accelerations" in context.component_options or options.get("accelerations"):
        options.update(context.component_options)
        policy = replace(context.policy, options=options)
        validate_runtime_policy_for_purpose(policy, context.purpose)
        return policy
    return context.policy


def validate_acceleration_installation(component: Any, context: Any) -> None:
    """Reject accepted options when a denoiser did not install their adapters.

    Reports are configuration evidence only. Nested expert reports use the
    same schema, so the assembler needs no model-class knowledge.
    """
    policy = acceleration_policy(context)
    options = policy.options.get("accelerations")
    if not options:
        return
    if not isinstance(options, Mapping):
        raise TypeError("accelerations must be a mapping of plugin names to options")
    active = {name for name, value in options.items() if value is not None and value is not False}
    if not active:
        return
    installed = set()

    def inspect(value):
        if isinstance(value, Mapping):
            manifest = value.get("accelerations")
            if isinstance(manifest, Mapping):
                installed.update(
                    row["name"] for row in manifest.get("installed", ()) if isinstance(row, Mapping) and "name" in row
                )
            for child in value.values():
                inspect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                inspect(child)

    reporter = getattr(component, "runtime_optimization_report", None)
    if callable(reporter):
        inspect(reporter())
    if missing := active - installed:
        raise ValueError(
            f"denoiser {type(component).__name__} did not install requested acceleration plugins: {sorted(missing)}"
        )


def _attention_policy(model: Any, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    from worldfoundry.core.attention.backends.probe import normalize_attention_backend, resolve_attention_backend

    if set(options) - {"self", "cross"} or not options:
        raise ValueError("attention_policy requires self and/or cross options")
    if policy is not None and any(
        policy.options.get(key) for key in ("sequence_parallel", "sp_degree", "approximate_attention", "cuda_graph")
    ):
        raise ValueError("scoped attention_policy has not been validated with SP, approximate processors or CUDA Graph")
    plans = []
    reports = {}
    for scope, raw in options.items():
        if not isinstance(raw, (str, Mapping)):
            raise TypeError("attention_policy scopes require a backend string or mapping")
        values = {"backend": raw} if isinstance(raw, str) else dict(raw)
        if "backend" not in values:
            raise ValueError("attention_policy scopes require a backend")
        requested = normalize_attention_backend(values.pop("backend"))
        resolved = resolve_attention_backend(requested, None if policy is None else policy.device)
        if values and resolved != "sol_attn":
            raise ValueError("provider options are currently supported only for Sol-Attn")
        if set(values) - {"tau", "thresh_type", "kv_splits", "compress_kv"}:
            raise ValueError("unknown Sol-Attn provider options")
        if resolved == "sol_attn" and scope != "self":
            raise ValueError("Sol-Attn is supported only for unmasked self-attention")
        if resolved == "sol_attn":
            from worldfoundry.core.attention.backends.sol import validate_sol_options

            values = validate_sol_options(**values)
            if policy is not None and (policy.compile or policy.dtype != torch.bfloat16):
                raise ValueError("Wan Sol-Attn requires BF16 eager inference; compile is not yet validated")
        class_name = "SelfAttention" if scope == "self" else "CrossAttention"
        parents = [
            child
            for child in model.modules()
            if type(child).__name__ == class_name and type(child).__module__.endswith(".wan.model")
        ]
        if not parents:
            raise ValueError(f"scoped attention_policy found no native Wan {scope}-attention modules")
        targets = [parent.attn for parent in parents]
        if scope == "self":
            from ..models.networks.wan.model import SelfAttentionProcessor

            if any(type(parent.processor) is not SelfAttentionProcessor for parent in parents):
                raise ValueError("scoped self-attention requires the default Wan processor")
        if resolved == "sol_attn" and any(parent.dim // parent.num_heads != 128 for parent in parents):
            raise ValueError("Sol-Attn requires head dimension 128 in every selected layer")
        reports[scope] = {
            "requested": requested,
            "resolved": resolved,
            "configured_modules": len(targets),
            "options": values,
        }
        for attribute, value in (("attention_backend", resolved), ("_worldfoundry_attention_options", values)):
            plans.append(_attribute_plan("attention_policy", targets, attribute, value, {}, frozenset()))

    def activate():
        handles = []
        try:
            for plan in plans:
                handles.append(plan.activate())
        except BaseException:
            for handle in reversed(handles):
                handle.undo()
            raise

        def undo():
            for handle in reversed(handles):
                handle.undo()

        approximate = any(
            row["resolved"] in {"sol_attn", "sage_attention", "sage_attention_3", "cudnn_fp8"}
            for row in reports.values()
        )
        return AccelerationHandle("attention_policy", {"approximate": approximate, "scopes": reports}, undo)

    seams = {f"wan.{scope}_attention.backend" for scope in options}
    if any(
        row["resolved"] in {"sol_attn", "sage_attention", "sage_attention_3", "cudnn_fp8"} for row in reports.values()
    ):
        seams.add("diffusion.approximation")
    return PreparedAcceleration("attention_policy", frozenset(seams), activate)
