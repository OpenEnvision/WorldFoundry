"""Tests for the auditable applied-optimizations record.

Locks the "requested vs effective vs fallback" contract so a manifest never
hides a silent downgrade (e.g. requested FP8, effective dense).
"""

from __future__ import annotations

import json

from worldfoundry.core.model_loading.optimize import AppliedOptimizations, QuantizationReport


def test_records_effective_optimizations() -> None:
    applied = AppliedOptimizations()
    applied.record_fusion(requested=True, fused_blocks=3)
    applied.record_quantization(
        QuantizationReport(requested_mode="fp8", applied=True, replaced_modules=30, scaling="rowwise")
    )
    applied.record_compile(requested=True)
    snap = applied.to_optimization_snapshot()
    assert snap.requested["fuse_qkv"] is True
    assert snap.requested["quantization"] == "fp8"
    assert snap.effective["fuse_qkv_blocks"] == 3
    assert snap.effective["quantization"] == "fp8-installed (runtime-pending)"
    assert snap.effective["quantization_replaced"] == 30
    assert snap.effective["compile"] == "compile-wrapper-installed (lazy)"
    assert snap.fallbacks == ()


def test_records_compile_effective_when_wrapper_present() -> None:
    applied = AppliedOptimizations()
    applied.record_compile(requested=True, compiled=True)
    snap = applied.to_optimization_snapshot()
    assert snap.effective["compile"] == "compile-wrapper-installed (lazy)"
    assert snap.fallbacks == ()


def test_records_eager_fallback_when_compile_no_ops() -> None:
    # compile_module_cached returns the original module (no _orig_mod) when
    # dynamo cannot trace the graph; the audit must not claim a compile happened.
    applied = AppliedOptimizations()
    applied.record_compile(requested=True, compiled=False)
    snap = applied.to_optimization_snapshot()
    assert snap.requested["compile"] is True
    assert snap.effective["compile"] == "eager (compile fallback)"
    assert any("compile" in f for f in snap.fallbacks)


def test_records_no_compile_when_not_requested() -> None:
    applied = AppliedOptimizations()
    applied.record_compile(requested=False)
    snap = applied.to_optimization_snapshot()
    assert snap.requested["compile"] is False
    assert "compile" not in snap.effective


def test_records_approximate_attention_downgrades_quality_tier() -> None:
    applied = AppliedOptimizations()
    applied.record_approximate_attention(
        requested=True, kind="vsa", wrapped_blocks=30, effective_kernel="vsa"
    )
    snap = applied.to_optimization_snapshot()
    assert snap.quality_tier == "approximate"
    assert snap.requested["approximate_attention"] == "vsa"
    assert snap.effective["approximate_attention_blocks"] == 30
    assert snap.effective["approximate_attention_kernel"] == "vsa"
    assert snap.fallbacks == ()


def test_records_approximate_attention_kernel_fallback() -> None:
    applied = AppliedOptimizations()
    applied.record_approximate_attention(
        requested=True, kind="sta", wrapped_blocks=30, effective_kernel="exact (no fastvideo_kernel)"
    )
    snap = applied.to_optimization_snapshot()
    assert snap.quality_tier == "approximate"
    assert any("approximate_attention" in f for f in snap.fallbacks)


def test_records_installed_approximate_attention_as_runtime_pending() -> None:
    applied = AppliedOptimizations()
    applied.record_approximate_attention(
        requested=True,
        kind="vsa",
        wrapped_blocks=30,
        effective_kernel="exact (sparse provider not executed)",
    )

    snap = applied.to_optimization_snapshot()

    assert (
        snap.effective["approximate_attention_kernel"]
        == "vsa-wrapper-installed (runtime-pending)"
    )
    assert snap.fallbacks == ()


def test_pending_kernel_without_installed_blocks_is_still_a_fallback() -> None:
    applied = AppliedOptimizations()
    applied.record_approximate_attention(
        requested=True,
        kind="vsa",
        wrapped_blocks=0,
        effective_kernel="exact (sparse provider not executed)",
    )

    snap = applied.to_optimization_snapshot()

    assert snap.effective["approximate_attention_kernel"].startswith("exact")
    assert any("no self-attention blocks matched" in reason for reason in snap.fallbacks)
    assert any("sparse provider not executed" in reason for reason in snap.fallbacks)


def test_approximate_attention_not_requested_keeps_exact() -> None:
    applied = AppliedOptimizations()
    applied.record_approximate_attention(
        requested=False, kind="vsa", wrapped_blocks=0, effective_kernel="pending"
    )
    snap = applied.to_optimization_snapshot()
    assert snap.quality_tier == "exact"
    assert snap.requested["approximate_attention"] is False
    assert "approximate_attention_blocks" not in snap.effective


def test_records_dense_fallback_when_fp8_skipped() -> None:
    applied = AppliedOptimizations()
    applied.record_quantization(
        QuantizationReport(requested_mode="fp8", applied=False, reason="no eligible nn.Linear modules matched")
    )
    snap = applied.to_optimization_snapshot()
    assert snap.effective["quantization"] == "dense"
    assert any("fp8" in f for f in snap.fallbacks)


def test_records_fusion_fallback_when_no_blocks() -> None:
    applied = AppliedOptimizations()
    applied.record_fusion(requested=True, fused_blocks=0)
    snap = applied.to_optimization_snapshot()
    assert any("fuse_qkv" in f for f in snap.fallbacks)


def test_none_quantization_is_not_a_fallback() -> None:
    applied = AppliedOptimizations()
    applied.record_quantization(QuantizationReport(requested_mode="none", applied=False))
    snap = applied.to_optimization_snapshot()
    assert snap.fallbacks == ()


def test_snapshot_is_manifest_serializable() -> None:
    applied = AppliedOptimizations()
    applied.record_fusion(requested=True, fused_blocks=2)
    applied.record_quantization(QuantizationReport(requested_mode="fp8", applied=True, replaced_modules=10))
    snap = applied.to_optimization_snapshot()
    # Must round-trip through the manifest JSON layer.
    assert json.dumps(snap.to_dict())


def test_records_exact_fused_rope_kernel() -> None:
    applied = AppliedOptimizations()
    applied.record_model_kernel(
        "fused_rope",
        requested=True,
        effective="hidden_qk_rmsnorm_rope_3d:fp64",
    )
    snap = applied.to_optimization_snapshot()
    assert snap.requested["fused_rope"] is True
    assert snap.effective["fused_rope"] == "hidden_qk_rmsnorm_rope_3d:fp64"
    assert snap.quality_tier == "exact"
    assert snap.fallbacks == ()


def test_records_fp32_fused_rope_as_numerically_approximate() -> None:
    applied = AppliedOptimizations()
    applied.record_model_kernel(
        "fused_rope",
        requested=True,
        effective="hidden_qk_rmsnorm_rope_3d:fp32",
        approximate=True,
    )
    snap = applied.to_optimization_snapshot()
    assert snap.quality_tier == "numerically-approximate"


def test_records_model_kernel_fallback_reason() -> None:
    applied = AppliedOptimizations()
    applied.record_model_kernel(
        "fused_rope",
        requested=True,
        effective="complex_rope",
        reason="unsupported head geometry",
    )
    snap = applied.to_optimization_snapshot()
    assert any("unsupported head geometry" in reason for reason in snap.fallbacks)
