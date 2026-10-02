"""Load-time optimization transform: apply a typed policy to a built model.

This is the single place that turns a declarative :class:`QuantizationPolicy`
into an in-place model transform (weight replacement), mirroring the
"transform stage at load time, not per-request patching" pattern used by
FastVideo (``convert_model_to_fp8``) and LightX2V (registry-driven weight
classes). Keeping it here — next to the policy definition and after the dense
checkpoint is loaded/placed — means model code never imports quantization
kernels directly, and the exact set of replaced modules is reported for the
performance manifest.

Order matters and follows ``plan/inference_operator_optimization_plan.md``
§4.1: load weights -> ``apply_quantization_policy`` -> install offload hooks.
Attention backend selection and CUDA Graph capture are separate, later stages
(the graph captures the already-quantized module), so this function only owns
weight-level precision transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from worldfoundry.core.model_loading.policy import (
    AttentionBackend,
    QuantizationMode,
    QuantizationPolicy,
)

if TYPE_CHECKING:
    from torch import nn

# ──────────────────────────────────────────────────────────────────────────
# Manifest rows — requested vs applied; never hide a silent downgrade
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    """What ``apply_quantization_policy`` actually did, for the manifest."""

    requested_mode: str
    applied: bool
    replaced_modules: int = 0
    scaling: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten for the performance manifest; ``extra`` keys merge last."""
        return {
            "requested_mode": self.requested_mode,
            "applied": self.applied,
            "replaced_modules": self.replaced_modules,
            "scaling": self.scaling,
            "reason": self.reason,
            **self.extra,
        }


@dataclass(frozen=True, slots=True)
class AttentionPolicyReport:
    """Requested vs resolved request-scoped attention provider."""

    requested_backend: str
    effective_backend: str
    configured_modules: int
    approximate: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flatten requested vs effective attention for the manifest."""
        return {
            "requested_backend": self.requested_backend,
            "effective_backend": self.effective_backend,
            "configured_modules": self.configured_modules,
            "approximate": self.approximate,
            "reason": self.reason,
        }


# ──────────────────────────────────────────────────────────────────────────
# Apply policies — bind seams / replace linears; kernels imported lazily
# ──────────────────────────────────────────────────────────────────────────


def apply_attention_policy(
    model: "nn.Module",
    backend: AttentionBackend | str,
    *,
    device: object = None,
) -> AttentionPolicyReport:
    """Bind an attention provider to compatible model seams in ``model``.

    The capability resolver runs once at build time. Compatible attention
    modules receive the resolved provider through ``set_attention_backend``;
    no process environment variable is mutated, so two pipelines can use
    different providers in one worker.
    """

    from worldfoundry.core.attention.backends.probe import (
        normalize_attention_backend,
        probe_attention_backends,
        require_generic_attention_backend,
        resolve_attention_backend,
    )

    raw_backend = backend.value if isinstance(backend, AttentionBackend) else str(backend)
    requested = normalize_attention_backend(raw_backend)
    # A generic load policy owns no per-layer sparse metadata or extra
    # checkpoint projections. Model-specific systems must be installed by the
    # corresponding graph factory, never bound to every set_attention_backend
    # seam as if they were drop-in Q/K/V kernels.
    require_generic_attention_backend(requested)
    effective = resolve_attention_backend(requested, device)
    configured = 0
    for child in model.modules():
        setter = getattr(child, "set_attention_backend", None)
        if callable(setter):
            # Diffusers' same-named API has a different backend vocabulary
            # ("native" rather than "torch") and changes a process-global
            # registry. It is not our per-module attention seam. Inherited
            # ModelMixin methods may also control none of a model's custom
            # attention layers, so leave those to their owning runtime.
            if getattr(setter, "__module__", "").startswith("diffusers."):
                continue
            setter(effective)
            configured += 1

    reason = None
    if requested == "flash_attention" and effective == "torch":
        capabilities = probe_attention_backends(device)
        reason = "; ".join(
            f"{name}: {capabilities[name].reason or 'not usable'}"
            for name in ("flash_attention_3", "flash_attention_2")
        )
    elif effective != requested and requested not in {"auto", "flash_attention"}:
        capabilities = probe_attention_backends(device)
        capability = capabilities.get(requested)
        reason = (
            capability.reason
            if capability is not None
            else f"provider {requested!r} did not resolve to an executable backend"
        )
    if configured == 0:
        seam_reason = "model exposes no compatible set_attention_backend seam"
        reason = seam_reason if reason is None else f"{reason}; {seam_reason}"
    return AttentionPolicyReport(
        requested_backend=requested,
        effective_backend=effective,
        configured_modules=configured,
        approximate=effective in {"sage_attention", "sage_attention_3"},
        reason=reason,
    )


def _option(policy: QuantizationPolicy, key: str, default: Any) -> Any:
    """Read ``policy.options[key]``; treat explicit ``None`` as "use default"."""
    value = policy.options.get(key, default)
    return default if value is None else value


def apply_quantization_policy(model: "nn.Module", policy: QuantizationPolicy) -> QuantizationReport:
    """Apply ``policy`` to ``model`` in place; return what was done.

    - ``NONE``: no-op (dense stays dense), reported as not applied.
    - ``FP8``: replace eligible ``nn.Linear`` with :class:`Float8Linear`
      (row-wise on PyTorch >= 2.5, tensor-wise fallback otherwise). ``options``
      may carry ``min_features`` (int), ``scaling`` ("auto"/"rowwise"/
      "tensorwise"), and ``keep_dense_fallback`` (bool). ``policy.exclude``
      names are matched against child module names.
    - ``NVFP4``: replace aligned ``nn.Linear`` layers with Blackwell FP4
      storage/execution and a portable dense fallback.
    - ``INT8``: install per-channel packed weights and a dynamic per-token W8A8
      Triton path by default. Explicit positive ``group_size`` values retain
      the portable groupwise dequantize+dense path.
    - ``INT4``: replace eligible linears with real groupwise packed weight-only
      storage. The portable path dequantizes before a dense GEMM and reports
      that fact; it never masquerades as a low-bit kernel.
    - ``GGUF``: consume a real GGUF checkpoint through the shared tensor loader,
      then repack eligible linears as groupwise INT4 (or ``runtime_bits=8``).
      Execution remains the honest portable dequantize+dense path until a
      qualified GGUF low-bit GEMM is installed.

    FP8/NVFP4 retain a dense fallback by default. Dynamic per-channel INT8 also
    retains BF16 weights for calibrated small-shape dispatch. Groupwise INT8,
    INT4, and GGUF remain compressed-only unless the caller explicitly asks to
    retain dense weights.
    """

    mode = QuantizationMode(policy.mode)
    if _option(policy, "fuse_fp8_ffn", False) and mode is not QuantizationMode.FP8:
        raise ValueError("fuse_fp8_ffn requires FP8 quantization")
    if mode == QuantizationMode.NONE:
        return QuantizationReport(requested_mode=mode.value, applied=False, reason="policy mode is none")

    # Import the kernel layer lazily so importing this module (and the policy
    # vocabulary) never pulls in torch-heavy acceleration code on a CPU/doc path.
    min_features = int(_option(policy, "min_features", 1024))
    keep_dense_default = mode in {QuantizationMode.FP8, QuantizationMode.NVFP4}
    keep_dense = bool(_option(policy, "keep_dense_fallback", keep_dense_default))
    fused_fp8_ffn = 0
    if mode is QuantizationMode.FP8:
        from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8

        scaling = str(_option(policy, "scaling", "auto"))
        use_fast_accum = bool(_option(policy, "use_fast_accum", True))
        replaced = replace_linear_with_float8(
            model,
            min_features=min_features,
            keep_dense_fallback=keep_dense,
            exclude=tuple(policy.exclude),
            scaling=scaling,
            use_fast_accum=use_fast_accum,
        )
        storage = f"fp8-{scaling}"
        execution = "fp8-wrapper-installed (runtime-pending)"
        if bool(_option(policy, "fuse_fp8_ffn", False)):
            from worldfoundry.core.acceleration.quantization.fused_ffn import fuse_fp8_gelu_feed_forwards

            fused_fp8_ffn = fuse_fp8_gelu_feed_forwards(model)
    elif mode is QuantizationMode.NVFP4:
        from worldfoundry.core.acceleration.quantization.nvfp4 import replace_linear_with_nvfp4

        scaling = "nvfp4-1x16"
        replaced = replace_linear_with_nvfp4(
            model,
            min_features=min_features,
            keep_dense_fallback=keep_dense,
            exclude=tuple(policy.exclude),
        )
        storage = "nvfp4-1x16"
        execution = "nvfp4-wrapper-installed (runtime-pending)"
    else:
        from worldfoundry.core.acceleration.quantization.linear import (
            replace_linear_with_weight_only,
        )

        if mode is QuantizationMode.GGUF:
            bits = int(_option(policy, "runtime_bits", 4))
            if bits not in {4, 8}:
                raise ValueError("GGUF runtime_bits must be 4 or 8")
        else:
            bits = 8 if mode is QuantizationMode.INT8 else 4
        group_size = int(
            _option(
                policy,
                "group_size",
                0 if mode is QuantizationMode.INT8 else 64,
            )
        )
        use_kernel = bool(
            _option(policy, "use_kernel", mode is QuantizationMode.INT8)
        )
        if "keep_dense_fallback" not in policy.options:
            # Dynamic W8A8 is profitable only for sufficiently large GEMMs.
            # Retain BF16 weights for the calibrated small-shape dispatch, but
            # preserve compressed-only storage for explicit groupwise INT8,
            # INT4, and GGUF modes that have no qualified kernel today.
            keep_dense = bool(
                mode is QuantizationMode.INT8
                and bits == 8
                and group_size == 0
                and use_kernel
            )
        scaling = (
            "symmetric-per-output-channel"
            if group_size == 0
            else f"symmetric-groupwise-{group_size}"
        )
        replaced = replace_linear_with_weight_only(
            model,
            bits=bits,
            min_features=min_features,
            group_size=group_size,
            keep_dense_fallback=keep_dense,
            use_kernel=use_kernel,
            exclude=tuple(policy.exclude),
        )
        if mode is QuantizationMode.GGUF:
            storage = f"gguf-loaded+groupwise-int{bits}"
        else:
            storage = (
                "per-channel-int8"
                if bits == 8 and group_size == 0
                else f"groupwise-int{bits}"
            )
        execution = (
            "int8-w8a8-wrapper-installed (runtime-pending)"
            if mode is QuantizationMode.INT8
            and bits == 8
            and group_size == 0
            and use_kernel
            else f"int{bits}-packed-weight-installed (runtime-pending)"
        )
    return QuantizationReport(
        requested_mode=mode.value,
        applied=replaced > 0,
        replaced_modules=replaced,
        scaling=scaling,
        reason=None if replaced > 0 else "no eligible nn.Linear modules matched the policy",
        extra={
            "min_features": min_features,
            "exclude": list(policy.exclude),
            "storage": storage,
            "execution": execution,
            "dense_fallback_retained": keep_dense,
            "fused_fp8_ffn_blocks": fused_fp8_ffn,
            "fuse_fp8_ffn_requested": bool(_option(policy, "fuse_fp8_ffn", False)),
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Audit accumulator — requested vs effective vs fallback for OptimizationSnapshot
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AppliedOptimizations:
    """Auditable record of what optimizations a loader actually applied.

    A loader fills this as it runs each transform, then can turn it into a
    :class:`worldfoundry.core.contracts.OptimizationSnapshot` for the
    performance manifest. This is the "requested vs effective vs fallback"
    visibility the optimization plan requires so a manifest never hides a
    silent downgrade (e.g. "requested FP8, effective dense").
    """

    requested: dict[str, Any] = field(default_factory=dict)
    effective: dict[str, Any] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    # "exact" until an opt-in lossy transform (e.g. approximate attention)
    # engages; then downgraded so the manifest surfaces that a lossy path ran.
    quality_tier: str = "exact"

    def record_attention(self, report: AttentionPolicyReport) -> None:
        """Record requested vs resolved backend; mark approximate Sage paths."""
        self.requested["attention"] = report.requested_backend
        self.effective["attention"] = report.effective_backend
        self.effective["attention_modules"] = report.configured_modules
        if report.approximate and self.quality_tier == "exact":
            self.quality_tier = "numerically-approximate"
        if report.reason:
            self.fallbacks.append(
                f"attention {report.requested_backend}: {report.reason}; "
                f"effective={report.effective_backend}"
            )

    def record_fusion(
        self,
        *,
        requested: bool,
        fused_blocks: int,
        strategy: str | None = None,
        split_threshold: int | None = None,
    ) -> None:
        """Record QKV fusion; a zero match after request is a fallback, not a no-op success."""
        self.requested["fuse_qkv"] = bool(requested)
        if requested:
            self.effective["fuse_qkv_blocks"] = int(fused_blocks)
            if strategy is not None:
                self.effective["qkv_strategy"] = str(strategy)
            if split_threshold is not None:
                self.effective["qkv_split_threshold"] = int(split_threshold)
            if fused_blocks == 0:
                self.fallbacks.append("fuse_qkv: no separate-projection attention blocks matched")

    def record_quantization(self, report: QuantizationReport) -> None:
        """Record weight transform; ``applied`` means a wrapper exists, not a kernel ran."""
        self.requested["quantization"] = report.requested_mode
        if report.extra.get("fuse_fp8_ffn_requested"):
            self.requested["fuse_fp8_ffn"] = True
            self.effective["fused_fp8_ffn_blocks"] = report.extra["fused_fp8_ffn_blocks"]
            if not report.extra["fused_fp8_ffn_blocks"]:
                self.fallbacks.append("fuse_fp8_ffn: no eligible Linear/GELU/Linear patterns matched")
        if report.applied:
            # Replacing a module proves that a wrapper/packed representation
            # exists, not that a hardware low-precision kernel has executed.
            # Request-time telemetry replaces this pending state after the
            # first forward call.
            execution = report.extra.get(
                "execution",
                f"{report.requested_mode}-installed (runtime-pending)",
            )
            self.effective["quantization"] = execution
            self.effective["quantization_replaced"] = report.replaced_modules
            if "storage" in report.extra:
                self.effective["quantization_storage"] = report.extra["storage"]
            self.effective["quantization_dense_fallback_retained"] = bool(
                report.extra.get("dense_fallback_retained", False)
            )
            if self.quality_tier == "exact":
                self.quality_tier = "numerically-approximate"
            if report.scaling is not None:
                self.effective["quantization_scaling"] = report.scaling
        elif report.requested_mode != QuantizationMode.NONE.value:
            self.effective["quantization"] = "dense"
            self.fallbacks.append(f"quantization {report.requested_mode}: {report.reason or 'not applied'}")

    def record_offload(
        self,
        *,
        requested_mode: str,
        effective: str,
        reason: str | None = None,
    ) -> None:
        """Record the concrete memory path instead of only the policy name.

        In particular, ``block`` is not considered asynchronous merely because
        it was accepted by the parser.  The loader records either the native
        two-layer prefetch window, a legacy on-demand wrapper, or resident
        placement.  Runtime counters later prove whether prefetch executed.
        """

        mode = str(requested_mode)
        self.requested["offload"] = mode
        self.effective["offload"] = str(effective)
        if reason:
            self.fallbacks.append(
                f"offload {mode}: {reason}; effective={effective}"
            )

    def record_compile(
        self,
        *,
        requested: bool,
        compiled: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record compile request vs wrapper/eager outcome; ``None`` means lazy install."""
        self.requested["compile"] = bool(requested)
        if not requested:
            return
        # torch.compile attaches _orig_mod to the wrapper; its absence means the
        # compile island fell back to eager (e.g. dynamo could not trace the
        # graph). Record what actually happened, not merely what was requested,
        # so the audit snapshot never claims a compile that silently no-op'd.
        if compiled is None:
            self.effective["compile"] = "compile-wrapper-installed (lazy)"
        elif compiled:
            self.effective["compile"] = "compile-wrapper-installed (lazy)"
        else:
            self.effective["compile"] = "eager (compile fallback)"
            self.fallbacks.append("compile: requested but wrapper fell back to eager")
        if details:
            self.effective["compile_config"] = dict(details)

    def record_cross_kv(self, *, requested: bool, wrapped_blocks: int) -> None:
        """Record static cross-KV wrapping; zero matches after request are a fallback."""
        self.requested["static_cross_kv"] = bool(requested)
        if requested:
            self.effective["static_cross_kv_blocks"] = int(wrapped_blocks)
            if wrapped_blocks == 0:
                self.fallbacks.append("static_cross_kv: no cross-attention blocks matched")

    def record_approximate_attention(
        self, *, requested: bool, kind: str, wrapped_blocks: int, effective_kernel: str
    ) -> None:
        """Record an opt-in *lossy* sparse-attention lane (STA/VSA).

        Engaging it downgrades ``quality_tier`` to ``"approximate"`` so the
        manifest never hides that a user opted into a lossy path. Records the
        effective kernel (``sta``/``vsa`` vs an ``exact (...)`` fallback string)
        and flags the case where no self-attention block matched.
        """
        self.requested["approximate_attention"] = kind if requested else False
        if not requested:
            return
        installed_blocks = int(wrapped_blocks)
        # ``install_approximate_attention`` cannot claim a sparse kernel until
        # the first provider forward has completed.  Its initial ``exact``
        # value is therefore installation-time pending evidence, not a dense
        # fallback.  Request-local runtime telemetry replaces this value after
        # execution.  Keep every other ``exact (...)`` value as a real
        # fallback: those values describe a missing prerequisite or a provider
        # failure rather than a wrapper waiting for its first call.
        runtime_pending = (
            installed_blocks > 0
            and effective_kernel == "exact (sparse provider not executed)"
        )
        self.effective["approximate_attention_blocks"] = installed_blocks
        self.effective["approximate_attention_kernel"] = (
            f"{kind}-wrapper-installed (runtime-pending)"
            if runtime_pending
            else effective_kernel
        )
        self.quality_tier = "approximate"
        if installed_blocks == 0:
            self.fallbacks.append("approximate_attention: no self-attention blocks matched")
        if effective_kernel.startswith("exact") and not runtime_pending:
            self.fallbacks.append(f"approximate_attention {kind}: {effective_kernel}")

    def record_sequence_parallel(
        self, *, requested: bool, sp_degree: int, wrapped_blocks: int, backend: str, head_parallel: bool = True
    ) -> None:
        """Record multi-GPU sequence-parallel (SP) attention.

        SP is numerically equivalent to single-GPU attention (not lossy), so it
        does NOT change ``quality_tier``. Records the degree, backend, wrapped
        block count, and whether the head-parallel (divisible) path was used.
        """
        self.requested["sequence_parallel"] = int(sp_degree) if requested else False
        if not requested:
            return
        self.effective["sequence_parallel_degree"] = int(sp_degree)
        self.effective["sequence_parallel_backend"] = backend
        self.effective["sequence_parallel_blocks"] = int(wrapped_blocks)
        if wrapped_blocks == 0:
            self.fallbacks.append("sequence_parallel: no self-attention blocks matched")
        if not head_parallel:
            self.fallbacks.append(
                f"sequence_parallel: num_heads not divisible by sp_degree={sp_degree}; using all-gather fallback"
            )

    def record_model_kernel(
        self,
        name: str,
        *,
        requested: bool,
        effective: str,
        approximate: bool = False,
        reason: str | None = None,
    ) -> None:
        """Record a model-specific optional kernel without hiding fallback."""

        key = str(name)
        self.requested[key] = bool(requested)
        if not requested:
            return
        self.effective[key] = str(effective)
        if approximate and self.quality_tier == "exact":
            self.quality_tier = "numerically-approximate"
        if reason:
            self.fallbacks.append(f"{key}: {reason}; effective={effective}")

    def to_optimization_snapshot(self, *, quality_tier: str | None = None) -> Any:
        """Build a JSON-safe :class:`~worldfoundry.core.contracts.OptimizationSnapshot`.

        *quality_tier* overrides the accumulated tier when a caller already
        knows a later stage changed it; otherwise the recorded tier is used.
        """
        from worldfoundry.core.contracts import OptimizationSnapshot

        return OptimizationSnapshot(
            requested=dict(self.requested),
            effective=dict(self.effective),
            fallbacks=tuple(self.fallbacks),
            quality_tier=self.quality_tier if quality_tier is None else quality_tier,
        )


__all__ = [
    "AppliedOptimizations",
    "AttentionPolicyReport",
    "QuantizationReport",
    "apply_attention_policy",
    "apply_quantization_policy",
]
