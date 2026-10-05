"""Native PyTorch module loading shared by every diffusion family.

This is the diffusion-facing adapter over ``worldfoundry.core.model_loading``.
A recipe factory builds a :class:`ModuleLoadSpec` plus a :class:`CheckpointSpec`;
:class:`NativeModuleLoader.load` materializes weights, applies
:class:`~..optimizations.policy.RuntimePolicy` (offload, quantization, compile,
approximate attention, QKV fusion, static cross-KV), and returns a concrete
``torch.nn.Module``.  The runner never loads weights; it only receives the
already-constructed ``Denoiser`` / encoder / VAE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from ..optimizations import OffloadMode, QuantizationMode, RuntimePolicy, parse_device_map
from .checkpoints import CheckpointSpec
from .materialize import MaterializedCheckpoint, NativeCheckpointResolver

StateDictConverter = Callable[[Mapping[str, object]], Mapping[str, object]]
CheckpointConfigResolver = Callable[[MaterializedCheckpoint], Mapping[str, object]]
ApproximateAttentionStateResolver = Callable[[], Mapping[str, object]]
PostLoadHook = Callable[[torch.nn.Module], None]


def _compile_policy_from_runtime(policy: RuntimePolicy):
    """Resolve the public compile knobs and external CUDA-Graph interaction."""

    from worldfoundry.core.execution.compile_cache import CompilePolicy

    backend = str(policy.options.get("compile_backend", "inductor")).strip()
    if not backend:
        raise ValueError("compile_backend cannot be empty")
    raw_options = policy.options.get("compile_options", {})
    if not isinstance(raw_options, Mapping):
        raise TypeError("compile_options must be a mapping")
    compile_options = dict(raw_options)
    configured_mode = policy.options.get("compile_mode")
    cuda_graph = bool(policy.options.get("cuda_graph", False))
    if compile_options and configured_mode is not None:
        raise ValueError(
            "compile_mode and compile_options are mutually exclusive in torch.compile"
        )
    if compile_options and cuda_graph and backend == "inductor":
        raise ValueError(
            "external cuda_graph cannot be combined with custom Inductor compile_options; "
            "use compile_mode='max-autotune-no-cudagraphs'"
        )
    if compile_options:
        mode = None
    elif configured_mode is None:
        mode = "max-autotune-no-cudagraphs" if cuda_graph and backend == "inductor" else "default"
    else:
        mode = str(configured_mode).strip()
    allowed_modes = {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    if mode is not None and mode not in allowed_modes:
        raise ValueError(
            f"unsupported compile_mode {mode!r}; expected one of {sorted(allowed_modes)}"
        )
    if cuda_graph and backend == "inductor" and mode != "max-autotune-no-cudagraphs":
        raise ValueError(
            "external cuda_graph with the Inductor backend requires "
            "compile_mode='max-autotune-no-cudagraphs' to avoid nested CUDA Graph capture"
        )
    dynamic_value = policy.options.get("compile_dynamic")
    if dynamic_value is not None and not isinstance(dynamic_value, bool):
        raise TypeError("compile_dynamic must be a bool or None")
    fullgraph_value = policy.options.get("compile_fullgraph", False)
    if not isinstance(fullgraph_value, bool):
        raise TypeError("compile_fullgraph must be a bool")
    return (
        CompilePolicy(
            backend=backend,
            mode=mode,
            fullgraph=fullgraph_value,
            dynamic=dynamic_value,
        ),
        compile_options,
    )


@dataclass(frozen=True, slots=True)
class ModuleLoadSpec:
    """Construction and weight-mapping rules for one PyTorch component.

    Attributes:
        module_class: Concrete ``nn.Module`` to construct.
        config: Static constructor kwargs.  Must not overlap keys returned by
            ``config_resolver`` (that collision raises :exc:`ValueError`).
        config_resolver: Optional callback that reads JSON / safetensors
            metadata from the materialized checkpoint.
        state_dict_converter: Optional key remapper applied before load.
        approximate_attention_state_resolver: Optional zero-argument callback
            returning auxiliary checkpoint tensors required by a learned
            approximate-attention provider. It is evaluated only when
            ``approximate_attention`` is requested and after checkpoint
            conversion has populated the callback-owned state.
        supports_approximate_attention: Whether this component owns the Wan
            self-attention blocks targeted by the approximate-attention policy.
            Runtime policies are shared by every component in a recipe, so this
            explicit capability prevents a DiT-only option from being applied
            to text encoders and VAEs.
        supports_acceleration_plugins: Apply model-scoped acceleration plugins
            to a native DiT, after transforms and before compilation. Encoders
            and VAEs sharing the runtime policy leave this capability disabled.
        vram_module_map: Required for ``OffloadMode.DISK`` and component
            offload; maps module types to AutoWrapped replacements.
        layer_container: Attribute name of the block list used by
            layerwise CPU offload and ``device_map=balanced``.
        vram_limit_gib: Optional per-module VRAM budget forwarded to core.
        post_load_hook: Mutation after restore; rejected with disk offload.
    """

    module_class: type[torch.nn.Module]
    config: Mapping[str, object] = field(default_factory=dict)
    config_resolver: CheckpointConfigResolver | None = None
    state_dict_converter: StateDictConverter | None = None
    approximate_attention_state_resolver: ApproximateAttentionStateResolver | None = None
    supports_approximate_attention: bool = False
    vram_module_map: Mapping[type[torch.nn.Module], type[torch.nn.Module]] | None = None
    layer_container: str | None = None
    vram_limit_gib: float | None = None
    post_load_hook: PostLoadHook | None = None
    supports_acceleration_plugins: bool = False


class NativeModuleLoader:
    """Thin diffusion-facing adapter over WorldFoundry's shared model loader.

    Failure conditions:
        ValueError: Static config overlaps checkpoint-derived config;
            ``device_map=balanced`` with any offload; disk offload with a
            post-load hook or without ``vram_module_map``; component offload
            without ``vram_module_map``.
        NotImplementedError: Quantization combined with disk offload (the
            load-time transform needs materialized parameters).
        TypeError: The core loader returned a non-``nn.Module`` object.
    """

    def load(
        self,
        spec: ModuleLoadSpec,
        checkpoint: CheckpointSpec,
        policy: RuntimePolicy,
    ) -> torch.nn.Module:
        materialized = NativeCheckpointResolver().materialize(checkpoint)
        sources = tuple(str(path) for path in materialized.paths)
        config = dict(spec.config)
        if spec.config_resolver is not None:
            resolved_config = dict(spec.config_resolver(materialized))
            overlap = sorted(set(config) & set(resolved_config))
            if overlap:
                raise ValueError(f"static and checkpoint-derived module config overlap: {overlap}")
            config.update(resolved_config)

        if policy.offload.mode is OffloadMode.DISK and policy.quantization.mode is not QuantizationMode.NONE:
            raise NotImplementedError(
                "quantization with disk offload is not supported because the "
                "load-time weight transform requires materialized parameters"
            )
        if (
            policy.offload.mode is OffloadMode.BLOCK
            and policy.quantization.mode is not QuantizationMode.NONE
        ):
            raise NotImplementedError(
                "quantization with block offload is not certified: quantized "
                "weights are registered buffers, while the asynchronous "
                "two-layer offloader currently owns Parameters only; use "
                "offload_mode=none or quantization=none"
            )
        if (
            policy.offload.mode is OffloadMode.DISK
            and policy.options.get("approximate_attention")
            and spec.supports_approximate_attention
        ):
            raise NotImplementedError(
                "approximate attention with disk offload is not supported because "
                "its Wan processor/provider installation requires materialized blocks"
            )

        device_map = parse_device_map(policy.options.get("device_map"), owner="runtime policy")
        shard_across_gpus = bool(device_map) and bool(spec.layer_container)
        if shard_across_gpus and policy.offload.mode is not OffloadMode.NONE:
            raise ValueError("device_map=balanced requires offload_mode=none so weights stay resident in VRAM")

        # Lazy imports keep the canonical package importable without optional
        # Transformers/VRAM dependencies until a real module is constructed.
        from worldfoundry.core.model_loading import (
            load_model,
            load_model_with_disk_offload,
        )

        source: str | list[str]
        source = sources[0] if len(sources) == 1 else list(sources)
        layerwise_cpu_offload = (
            policy.offload.mode is OffloadMode.BLOCK
            and bool(spec.layer_container)
        )
        legacy_block_offload = (
            policy.offload.mode is OffloadMode.BLOCK
            and not layerwise_cpu_offload
            and spec.vram_module_map is not None
        )
        if (
            policy.offload.mode is OffloadMode.BLOCK
            and not layerwise_cpu_offload
            and not legacy_block_offload
        ):
            raise ValueError(
                "block offload requires ModuleLoadSpec.layer_container for "
                "asynchronous prefetch, or vram_module_map for the explicitly "
                "reported legacy on-demand path"
            )
        # Layerwise offload must be installed before the full checkpoint ever
        # reaches CUDA. Loading on CUDA and attaching hooks afterwards defeats
        # the low-VRAM contract and can OOM during checkpoint restoration.
        load_device = (
            torch.device("cpu")
            if shard_across_gpus or layerwise_cpu_offload
            else policy.device
        )
        if policy.offload.mode is OffloadMode.DISK:
            if spec.post_load_hook is not None:
                raise ValueError("post-load mutations are not supported with disk offload")
            if spec.vram_module_map is None:
                raise ValueError("disk offload requires ModuleLoadSpec.vram_module_map")
            module = load_model_with_disk_offload(
                spec.module_class,
                source,
                config=config,
                torch_dtype=policy.dtype,
                device=policy.device,
                state_dict_converter=spec.state_dict_converter,
                module_map=dict(spec.vram_module_map),
            )
        else:
            vram_config = None
            module_map = None
            if spec.vram_module_map is not None and (
                policy.offload.mode is OffloadMode.COMPONENT
                or legacy_block_offload
            ):
                module_map = dict(spec.vram_module_map)
                vram_config = {
                    "offload_dtype": policy.dtype,
                    "offload_device": policy.offload.target,
                    "onload_dtype": policy.dtype,
                    "onload_device": policy.device,
                    "preparing_dtype": policy.dtype,
                    "preparing_device": policy.device,
                    "computation_dtype": policy.dtype,
                    "computation_device": policy.device,
                }
            module = load_model(
                spec.module_class,
                source,
                config=config,
                torch_dtype=policy.dtype,
                device=load_device,
                state_dict_converter=spec.state_dict_converter,
                module_map=module_map,
                vram_config=vram_config,
                vram_limit=spec.vram_limit_gib,
                post_load_hook=spec.post_load_hook,
            )

            # Load-time transforms run after checkpoint restoration (and any
            # adapter merge) but before offload hooks replace live parameters.
            from worldfoundry.core.model_loading.optimize import (
                AppliedOptimizations,
                apply_attention_policy,
                apply_quantization_policy,
            )

            applied = AppliedOptimizations()
            attention_report = apply_attention_policy(
                module,
                policy.attention,
                device=policy.device,
            )
            applied.record_attention(attention_report)

            fuse_qkv = bool(policy.options.get("fuse_qkv", False))
            qkv_strategy = str(policy.options.get("qkv_strategy", "auto"))
            qkv_split_threshold = int(
                policy.options.get("qkv_split_threshold", 8192)
            )
            fused_blocks = 0
            if fuse_qkv:
                from ..optimizations.qkv_fusion import fuse_qkv_projections

                fused_blocks = fuse_qkv_projections(
                    module,
                    strategy=qkv_strategy,
                    split_threshold=qkv_split_threshold,
                )
            applied.record_fusion(
                requested=fuse_qkv,
                fused_blocks=fused_blocks,
                strategy=qkv_strategy,
                split_threshold=qkv_split_threshold,
            )

            sequence_parallel_value = policy.options.get(
                "sequence_parallel",
                policy.options.get("sp_degree", False),
            )
            if sequence_parallel_value:
                if isinstance(sequence_parallel_value, bool):
                    import torch.distributed as dist

                    if not dist.is_initialized():
                        raise RuntimeError(
                            "sequence_parallel=True cannot infer a degree before "
                            "torchrun/NCCL initialization"
                        )
                    sequence_parallel_degree = int(dist.get_world_size())
                else:
                    sequence_parallel_degree = int(sequence_parallel_value)
                # Do not use a blanket incompatibility list. Fused 3D RoPE is
                # safe because its kernel accepts the SP rank's global
                # sequence_offset and padded-token boundary. The remaining
                # combinations each lack a distinct correctness mechanism.
                conflict_reasons = {
                    "adacache": (
                        "cache hit/miss decisions must be synchronized across ranks before "
                        "any rank may skip Ulysses collectives"
                    ),
                    "feature_cache": (
                        "cache hit/miss decisions must be synchronized across ranks and "
                        "cached residuals need a shard-consistent lifecycle"
                    ),
                    "magcache": (
                        "cache hit/miss decisions must be synchronized across ranks before "
                        "any rank may skip Ulysses collectives"
                    ),
                    "taylorseer": (
                        "cache/extrapolation decisions must be synchronized across ranks and "
                        "history must remain shard-consistent"
                    ),
                    "teacache": (
                        "cache hit/miss decisions must be synchronized across ranks before "
                        "any rank may skip Ulysses collectives"
                    ),
                    "approximate_attention": (
                        "both features replace the Wan SelfAttention processor; sparse "
                        "metadata/routing must be rebuilt after Ulysses head exchange"
                    ),
                    "cuda_graph": (
                        "NCCL collective capture needs a rank-coordinated graph lifecycle "
                        "and a graph-safe communicator path not provided by the current runner"
                    ),
                }
                active_conflicts = [
                    (name, reason)
                    for name, reason in conflict_reasons.items()
                    if bool(policy.options.get(name))
                ]
                if active_conflicts:
                    details = "; ".join(
                        f"{name}: {reason}" for name, reason in active_conflicts
                    )
                    raise ValueError(
                        "sequence_parallel combination is not yet correctness-safe; " + details
                    )
                from ..optimizations.sequence_parallel import (
                    enable_sequence_parallel,
                    require_sequence_parallel_runtime,
                )

                require_sequence_parallel_runtime(sequence_parallel_degree)
                sequence_parallel_state = enable_sequence_parallel(
                    module,
                    sequence_parallel_degree,
                )
                module._worldfoundry_sequence_parallel = sequence_parallel_state
                applied.record_sequence_parallel(
                    requested=True,
                    sp_degree=sequence_parallel_degree,
                    wrapped_blocks=sequence_parallel_state.wrapped_blocks,
                    backend=sequence_parallel_state.backend,
                    head_parallel=sequence_parallel_state.head_parallel,
                )
            else:
                applied.record_sequence_parallel(
                    requested=False,
                    sp_degree=1,
                    wrapped_blocks=0,
                    backend="single-gpu",
                )

            static_cross_kv = bool(policy.options.get("static_cross_kv", False))
            wrapped_cross_kv = 0
            if static_cross_kv:
                from ..optimizations.static_cross_kv import install_static_cross_kv_cache

                cross_kv_cache = install_static_cross_kv_cache(module)
                wrapped_cross_kv = cross_kv_cache.wrapped_blocks
                module._worldfoundry_static_cross_kv = cross_kv_cache
            applied.record_cross_kv(requested=static_cross_kv, wrapped_blocks=wrapped_cross_kv)

            quantization_report = apply_quantization_policy(module, policy.quantization)
            applied.record_quantization(quantization_report)

            # Install model-specific sparse processors only after every
            # weight/projection transform.  In particular, FastVideo learned
            # SLA owns an FP32 ``proj_l`` in each provider; creating those
            # modules before generic quantization would let a permissive
            # ``min_features`` policy replace them and silently break parity.
            # QKV fusion is already complete, and the sparse processor has an
            # explicit fused-QKV contract.
            approximate_option = (
                policy.options.get("approximate_attention")
                if spec.supports_approximate_attention
                else None
            )
            if approximate_option:
                from ..optimizations.approximate_attention import (
                    install_approximate_attention,
                    parse_approximate_attention,
                )

                approximate_config = parse_approximate_attention(approximate_option)
                approximate_checkpoint_state = (
                    spec.approximate_attention_state_resolver()
                    if spec.approximate_attention_state_resolver is not None
                    else None
                )
                approximate_state = install_approximate_attention(
                    module,
                    approximate_config,
                    checkpoint_state_dict=approximate_checkpoint_state,
                )
                module._worldfoundry_approximate_attention = approximate_state
                applied.record_approximate_attention(
                    requested=True,
                    kind=approximate_config.kind,
                    wrapped_blocks=approximate_state.wrapped_blocks,
                    effective_kernel=approximate_state.effective_kernel,
                )
            else:
                applied.record_approximate_attention(
                    requested=False,
                    kind="",
                    wrapped_blocks=0,
                    effective_kernel="exact",
                )
            module._worldfoundry_applied_optimizations = applied

            if layerwise_cpu_offload:
                from worldfoundry.core.vram import enable_layerwise_cpu_offload

                offload_handle = enable_layerwise_cpu_offload(
                    module,
                    layer_container=spec.layer_container,
                    device=policy.device,
                    pin_memory=policy.offload.pin_memory,
                )
                if not offload_handle.enabled:
                    raise RuntimeError(
                        "asynchronous block offload was requested but could not "
                        f"be installed: {offload_handle.reason or 'unknown reason'}"
                    )
                module._worldfoundry_layerwise_cpu_offload_handle = offload_handle
                # The hook installer replaces managed layer parameters with
                # zero-sized CUDA placeholders while retaining CPU masters.
                # Moving the remaining module skeleton afterwards keeps input,
                # output, norm, and buffer tensors on the computation device
                # without materializing every managed layer there at once.
                module = module.to(dtype=policy.dtype, device=policy.device)
            elif policy.offload.mode is OffloadMode.COMPONENT and module_map is None:
                raise ValueError("component offload requires ModuleLoadSpec.vram_module_map")

            if layerwise_cpu_offload:
                applied.record_offload(
                    requested_mode=policy.offload.mode.value,
                    effective="async-double-buffer-installed (runtime-pending)",
                )
            elif legacy_block_offload:
                applied.record_offload(
                    requested_mode=policy.offload.mode.value,
                    effective="legacy-on-demand-wrapper",
                    reason=(
                        "no layer_container was declared, so copy/compute "
                        "overlap cannot be proven"
                    ),
                )
            elif policy.offload.mode is OffloadMode.COMPONENT:
                applied.record_offload(
                    requested_mode=policy.offload.mode.value,
                    effective="component-wrapper",
                )
            else:
                applied.record_offload(
                    requested_mode=policy.offload.mode.value,
                    effective="resident",
                )

        if spec.supports_acceleration_plugins and policy.options.get("accelerations"):
            from ..optimizations.plugins import install_diffusion_accelerations

            session = install_diffusion_accelerations(module, policy.options["accelerations"], policy)
            applied = getattr(module, "_worldfoundry_applied_optimizations", None)
            if applied is not None:
                applied.requested["accelerations"] = dict(policy.options["accelerations"])
                applied.effective["accelerations"] = session.report()
                if any(handle.details.get("approximate") for handle in session.handles):
                    applied.quality_tier = "approximate"

        if shard_across_gpus:
            from worldfoundry.core.vram import enable_balanced_device_map

            module._worldfoundry_device_map_handle = enable_balanced_device_map(
                module,
                layer_container=spec.layer_container,
                home_device=policy.device,
            )

        approximate_state = getattr(
            module,
            "_worldfoundry_approximate_attention",
            None,
        )
        if approximate_state is not None:
            from ..optimizations.approximate_attention import (
                place_fastvideo_sla_providers,
                prepare_lightx2v_providers,
            )

            place_fastvideo_sla_providers(module, approximate_state)
            prepare_lightx2v_providers(
                module,
                approximate_state,
                fallback_device=policy.device,
            )

        if policy.compile:
            from worldfoundry.core.attention.backends.dispatch import (
                attention_compile_receipt_scope,
            )
            from worldfoundry.runtime.compile_cache import compile_callable_cached

            # Compile the bound forward callable and retain the concrete model
            # instance. Compiling an entire module returns OptimizedModule,
            # which breaks the checkpoint-compatible type contract enforced by
            # every component factory (and hides model-specific seams such as
            # Wan's cache lifecycle). A compiled bound method gives the same
            # lazy Dynamo/Inductor execution path without changing the loader's
            # return type.
            compile_policy, compile_options = _compile_policy_from_runtime(policy)
            eager_forward = module.forward
            compiled_forward = compile_callable_cached(
                eager_forward,
                policy=compile_policy,
                namespace="diffusion-module-forward",
                options=compile_options,
            )
            wrapper_installed = compiled_forward is not eager_forward
            compile_runtime = {
                "wrapper_installed": wrapper_installed,
                # ``calls``/``failures`` are lifetime totals retained for
                # diagnostics.  The denoiser resets the request counters at
                # step zero and exposes those under the legacy report keys so
                # a warmup or previous prompt cannot certify a later request.
                "calls": 0,
                "failures": 0,
                "last_error": None,
                "request_calls": 0,
                "request_failures": 0,
                "request_last_error": None,
                "attention_provider_graph_traces": {},
            }
            if wrapper_installed:
                def audited_compiled_forward(*args, **kwargs):
                    try:
                        with attention_compile_receipt_scope(
                            compile_runtime["attention_provider_graph_traces"]
                        ):
                            output = compiled_forward(*args, **kwargs)
                    except Exception as error:
                        compile_runtime["failures"] += 1
                        compile_runtime["request_failures"] += 1
                        error_text = (
                            f"{type(error).__name__}: {error}"
                        )
                        compile_runtime["last_error"] = error_text
                        compile_runtime["request_last_error"] = error_text
                        raise
                    compile_runtime["calls"] += 1
                    compile_runtime["request_calls"] += 1
                    return output

                module.forward = audited_compiled_forward
            else:
                module.forward = eager_forward
            module._worldfoundry_compile_runtime = compile_runtime
            module._worldfoundry_compile_config = {
                "backend": compile_policy.backend,
                "mode": compile_policy.mode,
                "fullgraph": compile_policy.fullgraph,
                "dynamic": compile_policy.dynamic,
                "options": compile_options,
            }
            applied = getattr(module, "_worldfoundry_applied_optimizations", None)
            if applied is not None:
                applied.record_compile(
                    requested=True,
                    compiled=wrapper_installed,
                    details=module._worldfoundry_compile_config,
                )
        else:
            applied = getattr(module, "_worldfoundry_applied_optimizations", None)
            if applied is not None:
                applied.record_compile(requested=False)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"module loader returned {type(module).__name__}, expected torch.nn.Module")
        return module


__all__ = [
    "CheckpointConfigResolver",
    "ModuleLoadSpec",
    "NativeModuleLoader",
    "PostLoadHook",
    "StateDictConverter",
    "_compile_policy_from_runtime",
]
