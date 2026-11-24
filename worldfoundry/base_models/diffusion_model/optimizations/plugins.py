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


def _cuda_graph_blocks_dynamic_plugin(raw_graph: object) -> bool:
    """Return True when ``cuda_graph`` is enabled in a non-piecewise mode."""

    if raw_graph is None or raw_graph is False:
        return False
    if isinstance(raw_graph, str):
        return raw_graph.strip().casefold() not in {"", "off", "none", "false", "disabled", "piecewise"}
    return bool(raw_graph)


def _cuda_graph_enabled(raw_graph: object) -> bool:
    """Return True when ``cuda_graph`` enables whole or piecewise capture."""

    if raw_graph is None or raw_graph is False:
        return False
    if isinstance(raw_graph, str):
        return raw_graph.strip().casefold() not in {"", "off", "none", "false", "disabled"}
    return bool(raw_graph)


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
        raw_graph = policy.options.get("cuda_graph")
        graph_blocks_easycache = _cuda_graph_blocks_dynamic_plugin(raw_graph)
        incompatible = [
            key
            for key in (
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
        if graph_blocks_easycache:
            incompatible.append("cuda_graph")
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
    allow_compose = bool(policy is not None and policy.options.get("allow_approximate_composition", False))
    ec_seams = {"diffusion.block_stack"}
    if not allow_compose:
        ec_seams.update({"diffusion.precision_cache", "diffusion.approximation"})
    return _attribute_plan(
        "easycache",
        [model],
        "_worldfoundry_easycache_config",
        config,
        {"approximate": config.threshold > 0, "seam": "embedded-token-block-stack", **config.options},
        frozenset(ec_seams),
    )


def diffusion_acceleration_registry() -> AccelerationRegistry:
    """Return an extensible registry without loading optional provider packages."""
    from .kv_fusion import prepare_wan_cross_kv_fusion
    from .mha import prepare_optimized_mha
    from .precision import prepare_selective_fp8
    from .svdquant import prepare_svdquant

    registry = AccelerationRegistry()
    registry.register("sana_block_fusion", _sana_block_fusion)
    registry.register("easycache", _easycache)
    registry.register("attention_policy", _attention_policy)
    registry.register("token_pruning", _token_pruning)
    registry.register("selective_fp8", prepare_selective_fp8)
    registry.register("svdquant", prepare_svdquant)
    registry.register("wan_cross_kv_fusion", prepare_wan_cross_kv_fusion)
    registry.register("optimized_mha", prepare_optimized_mha)
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


def _token_pruning(model: Any, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    """Install reversible Sol-Engine feature-norm token pruning on Wan block stacks."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TokenPruningPolicy:
        keep_ratio: float = 0.75
        method: str = "feat_norm"
        start_block: int = 2
        end_block: int = -2
        dense_first_steps: int = 5
        dense_last_steps: int = 5

        def __post_init__(self) -> None:
            if isinstance(self.keep_ratio, bool) or not (0.0 < float(self.keep_ratio) <= 1.0):
                raise ValueError("token_pruning keep_ratio must be in (0, 1]")
            if not isinstance(self.method, str) or not self.method.strip():
                raise ValueError("token_pruning method must be a non-empty string")
            if isinstance(self.start_block, bool) or not isinstance(self.start_block, int) or self.start_block < 0:
                raise ValueError("token_pruning start_block must be a non-negative integer")
            if isinstance(self.end_block, bool) or not isinstance(self.end_block, int):
                raise ValueError("token_pruning end_block must be an integer")
            if self.dense_first_steps < 0 or self.dense_last_steps < 0:
                raise ValueError("token_pruning dense step boundaries must be non-negative")

    config = TokenPruningPolicy(**options)
    if getattr(model, "training", False):
        raise ValueError("token_pruning requires model.eval()")
    if getattr(model, "_worldfoundry_sequence_parallel", None) is not None or (
        policy is not None and any(policy.options.get(k) for k in ("sequence_parallel", "sp_degree"))
    ):
        raise ValueError("token_pruning is currently supported on single-rank sequence layouts only")
    if policy is not None and policy.options.get("approximate_attention"):
        raise ValueError("token_pruning conflicts with grid-indexed approximate_attention processors")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) < 4:
        raise ValueError("token_pruning requires a transformer model with at least 4 blocks")
    num_blocks = len(blocks)
    resolved_end = num_blocks + config.end_block if config.end_block <= 0 else config.end_block
    if not (0 <= config.start_block < resolved_end <= num_blocks):
        raise ValueError(
            f"token_pruning block span [{config.start_block}, {resolved_end}) is invalid for {num_blocks} blocks"
        )
    if policy is not None:
        raw_graph = policy.options.get("cuda_graph")
        if _cuda_graph_blocks_dynamic_plugin(raw_graph):
            raise ValueError(
                "token_pruning requires cuda_graph='piecewise' or 'off' (shape changes inside middle blocks)"
            )
    runtime: dict[str, Any] = {
        "pruned_calls": 0,
        "dense_calls": 0,
        "keep_ratio": float(config.keep_ratio),
        "method": config.method,
        "block_span": [config.start_block, resolved_end],
        "pruners": {},
    }

    def activate() -> AccelerationHandle:
        prev_cfg = getattr(model, "_worldfoundry_token_pruning_config", None)
        prev_rt = getattr(model, "_worldfoundry_token_pruning_runtime", None)
        model._worldfoundry_token_pruning_config = config
        model._worldfoundry_token_pruning_runtime = runtime

        def undo() -> None:
            if prev_cfg is None:
                model.__dict__.pop("_worldfoundry_token_pruning_config", None)
            else:
                model._worldfoundry_token_pruning_config = prev_cfg
            if prev_rt is None:
                model.__dict__.pop("_worldfoundry_token_pruning_runtime", None)
            else:
                model._worldfoundry_token_pruning_runtime = prev_rt
            runtime["pruners"].clear()

        def reset_window() -> None:
            runtime["pruned_calls"] = 0
            runtime["dense_calls"] = 0
            runtime["pruners"].clear()

        return AccelerationHandle(
            "token_pruning",
            {
                "approximate": config.keep_ratio < 1.0,
                "keep_ratio": float(config.keep_ratio),
                "method": config.method,
                "start_block": config.start_block,
                "end_block": resolved_end,
                "dense_first_steps": config.dense_first_steps,
                "dense_last_steps": config.dense_last_steps,
            },
            undo,
            runtime_report=lambda: {
                "pruned_calls": runtime["pruned_calls"],
                "dense_calls": runtime["dense_calls"],
                "keep_ratio": runtime["keep_ratio"],
                "block_span": runtime["block_span"],
            },
            reset_request_window=reset_window,
        )

    allow_compose = bool(policy is not None and policy.options.get("allow_approximate_composition", False))
    tp_seams = {"diffusion.token_pruning"}
    if not allow_compose:
        tp_seams.add("diffusion.approximation")
    return PreparedAcceleration("token_pruning", frozenset(tp_seams), activate)


def _attention_policy(model: Any, options: Mapping[str, Any], policy: Any) -> PreparedAcceleration:
    from worldfoundry.core.attention.backends.probe import normalize_attention_backend, resolve_attention_backend

    if set(options) - {"self", "cross"} or not options:
        raise ValueError("attention_policy requires self and/or cross options")
    if policy is not None and policy.options.get("approximate_attention"):
        raise ValueError("scoped attention_policy conflicts with approximate_attention processors")
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
        _SOL_SCHEDULE_KEYS = {"dense_steps", "dense_blocks", "tile_3d"}
        _SOL_PROVIDER_KEYS = {
            "tau",
            "thresh_type",
            "kv_splits",
            "compress_kv",
            "sink_tokens",
            "sink_start",
            "allow_multi_split",
        }
        if set(values) - (_SOL_PROVIDER_KEYS | _SOL_SCHEDULE_KEYS):
            raise ValueError("unknown Sol-Attn provider options")
        if resolved == "sol_attn" and scope != "self":
            raise ValueError("Sol-Attn is supported only for unmasked self-attention")
        sol_schedule: dict[str, Any] | None = None
        dense_block_indices: set[int] = set()
        if resolved == "sol_attn":
            from worldfoundry.core.attention.backends.sol import validate_sol_options

            raw_dense_steps = values.pop("dense_steps", 0)
            raw_dense_blocks = values.pop("dense_blocks", ())
            raw_tile_3d = values.pop("tile_3d", None)
            if isinstance(raw_dense_steps, bool) or not isinstance(raw_dense_steps, int) or raw_dense_steps < 0:
                raise ValueError("Sol-Attn dense_steps must be a non-negative integer")
            if isinstance(raw_dense_blocks, (str, bytes)) or not hasattr(raw_dense_blocks, "__iter__"):
                raise ValueError("Sol-Attn dense_blocks must be a sequence of block indices")
            parsed_dense_blocks: list[int] = []
            for item in raw_dense_blocks:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError("Sol-Attn dense_blocks entries must be integers")
                parsed_dense_blocks.append(int(item))
            parsed_tile_3d: tuple[int, int, int] | None = None
            if raw_tile_3d is not None:
                if isinstance(raw_tile_3d, (str, bytes)) or not hasattr(raw_tile_3d, "__iter__"):
                    raise ValueError("Sol-Attn tile_3d must be a 3-tuple of positive integers")
                tile_vals = list(raw_tile_3d)
                if len(tile_vals) != 3 or any(
                    isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in tile_vals
                ):
                    raise ValueError("Sol-Attn tile_3d must contain three positive integers (tf, th, tw)")
                parsed_tile_3d = (int(tile_vals[0]), int(tile_vals[1]), int(tile_vals[2]))

            values = validate_sol_options(**values)
            if policy is not None and (policy.compile or policy.dtype != torch.bfloat16):
                raise ValueError("Wan Sol-Attn requires BF16 eager/graph inference; compile is not yet validated")
            if raw_dense_steps > 0 or parsed_tile_3d is not None or parsed_dense_blocks:
                sol_schedule = {
                    "dense_steps": int(raw_dense_steps),
                    "dense_blocks": tuple(parsed_dense_blocks),
                    "tile_3d": parsed_tile_3d,
                }
        if (
            resolved == "scaled_int8_fp8"
            and policy is not None
            and _cuda_graph_enabled(policy.options.get("cuda_graph"))
        ):
            raise ValueError("scaled_int8_fp8 attention does not support CUDA Graph capture")
        class_name = "SelfAttention" if scope == "self" else "CrossAttention"
        parents = [
            child
            for child in model.modules()
            if type(child).__name__ == class_name and type(child).__module__.endswith(".wan.model")
        ]
        if not parents:
            raise ValueError(f"scoped attention_policy found no native Wan {scope}-attention modules")
        num_parents = len(parents)
        if sol_schedule is not None and sol_schedule["dense_blocks"]:
            for idx in sol_schedule["dense_blocks"]:
                norm_idx = idx + num_parents if idx < 0 else idx
                if 0 <= norm_idx < num_parents:
                    dense_block_indices.add(norm_idx)
        targets = [parent.attn for idx, parent in enumerate(parents) if idx not in dense_block_indices]
        dense_targets = [parent.attn for idx, parent in enumerate(parents) if idx in dense_block_indices]
        if scope == "self":
            from ..models.networks.wan.model import SelfAttentionProcessor
            from .sequence_parallel import SequenceParallelSelfAttentionProcessor

            allowed_procs = (SelfAttentionProcessor, SequenceParallelSelfAttentionProcessor)
            if any(not isinstance(parent.processor, allowed_procs) for parent in parents):
                raise ValueError("scoped self-attention requires the default or Ulysses SP Wan processor")
        if resolved == "sol_attn" and any(parent.dim // parent.num_heads != 128 for parent in parents):
            raise ValueError("Sol-Attn requires head dimension 128 in every selected layer")
        report_entry: dict[str, Any] = {
            "requested": requested,
            "resolved": resolved,
            "configured_modules": len(targets),
            "options": values,
        }
        if sol_schedule is not None:
            report_entry["schedule"] = {
                "dense_steps": sol_schedule["dense_steps"],
                "dense_blocks": sorted(dense_block_indices),
                "tile_3d": sol_schedule["tile_3d"],
            }
        reports[scope] = report_entry
        if targets:
            for attribute, value in (("attention_backend", resolved), ("_worldfoundry_attention_options", values)):
                plans.append(_attribute_plan("attention_policy", targets, attribute, value, {}, frozenset()))
        if dense_targets:
            plans.append(
                _attribute_plan("attention_policy", dense_targets, "attention_backend", "torch", {}, frozenset())
            )
            plans.append(
                _attribute_plan(
                    "attention_policy", dense_targets, "_worldfoundry_attention_options", None, {}, frozenset()
                )
            )
        if sol_schedule is not None and (sol_schedule["dense_steps"] > 0 or sol_schedule["tile_3d"] is not None):
            plans.append(
                _attribute_plan(
                    "attention_policy", [model], "_worldfoundry_sol_schedule", sol_schedule, {}, frozenset()
                )
            )

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

    allow_compose = bool(policy is not None and policy.options.get("allow_approximate_composition", False))
    seams = {f"wan.{scope}_attention.backend" for scope in options}
    if not allow_compose and any(
        row["resolved"] in {"sol_attn", "sage_attention", "sage_attention_3", "cudnn_fp8"} for row in reports.values()
    ):
        seams.add("diffusion.approximation")
    return PreparedAcceleration("attention_policy", frozenset(seams), activate)
