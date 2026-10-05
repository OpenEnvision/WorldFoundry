"""Contract tests for WanDenoiser cross-step feature cache wiring (CPU-only).

The cache reuse policy itself is tested in the AdaptiveResidualCache unit tests;
here we lock the denoiser-level wiring that must hold on any device: disabled by
default (dense every step), opt-in threshold skips interior steps while keeping
warmup/boundary dense, per-CFG-branch isolation (positive and negative keep
separate residual trajectories), and autograd falls back to dense.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WanDenoiser
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    QKVFusionState,
)
from worldfoundry.base_models.diffusion_model.optimizations.wan.feature_cache import (
    WanFeatureCacheConfig,
)


class _CountingModel:
    """Minimal WanModel stand-in that mirrors the real block-residual cache path.

    The real WanModel.forward applies the feature cache around its transformer
    block stack: it caches the residual (blocks(x) - x) keyed on the timestep
    modulation, and on a hit reuses that residual on the current x. This stub
    reproduces that exact wiring so the denoiser-level tests exercise the same
    hit/dense accounting, counting only genuine dense "block stack" computes.
    """

    per_token_timestep = False
    inject_sample_info = False
    has_image_input = False
    patch_size = (1, 2, 2)

    def __init__(self) -> None:
        self.calls = 0

    def _blocks(self, x):
        self.calls += 1
        # Deterministic per-call residual so a replay differs from a fresh compute.
        return x + float(self.calls)

    def __call__(self, *, x, timestep, context, **kwargs):
        cache = kwargs.get("feature_cache")
        if cache is None:
            return self._blocks(x)
        # Signal stands in for t_mod; constant here so change stays below any
        # positive threshold once warmed up (mirrors the real stable signal).
        signal = torch.zeros(1, 4)
        residual = cache.run(
            int(kwargs.get("feature_cache_step", 0)),
            signal,
            lambda: self._blocks(x) - x,
            total_steps=kwargs.get("feature_cache_total_steps"),
        )
        return x + residual


class _BranchCountingBlockModel:
    """Block-cache stand-in whose residual trajectory depends on CFG context."""

    per_token_timestep = False
    inject_sample_info = False
    has_image_input = False
    patch_size = (1, 2, 2)

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, x, timestep, context, **kwargs):
        del timestep
        cache = kwargs["feature_cache"]
        context_scale = context.flatten()[0].to(device=x.device, dtype=x.dtype)

        def run_block(block_id: int, hidden: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return hidden + context_scale * float(block_id + 1)

        return cache.run_blocks(
            int(kwargs.get("feature_cache_step", 0)),
            x,
            run_block,
            block_count=3,
            total_steps=kwargs.get("feature_cache_total_steps"),
        )


class _BranchTaylorPhaseModel:
    """Wan stand-in exposing the exact BlockTaylorSeer phase callback seam."""

    per_token_timestep = False
    inject_sample_info = False
    has_image_input = False
    patch_size = (1, 2, 2)

    def __init__(self) -> None:
        self.dense_phase_calls = 0

    def __call__(self, *, x, timestep, context, **kwargs):
        del timestep
        cache = kwargs["feature_cache"]
        context_scale = context.flatten()[0].to(device=x.device, dtype=x.dtype)

        def run_block(block_id: int, hidden: torch.Tensor) -> torch.Tensor:
            return hidden + context_scale * float(3 * (block_id + 1))

        def run_phase_block(block_id, hidden, cached_phases):
            if cached_phases is None:
                self.dense_phase_calls += 1
                value = context_scale * float(block_id + 1)
                phases = {
                    "self_attn_out": torch.full_like(hidden, value),
                    "cross_attn_out": torch.full_like(hidden, value),
                    "ffn_out": torch.full_like(hidden, value),
                }
            else:
                phases = cached_phases
            return hidden + sum(phases.values()), phases

        return cache.run_phase_blocks(
            int(kwargs.get("feature_cache_step", 0)),
            x,
            run_block,
            run_phase_block,
            block_count=2,
            total_steps=kwargs.get("feature_cache_total_steps"),
        )


def _denoiser(threshold=None) -> WanDenoiser:
    d = WanDenoiser.__new__(WanDenoiser)
    d.model = _CountingModel()
    d.compute_dtype = torch.float32
    d.reference_condition_key = None
    d.manage_autocast = False
    d._graph_runner = None
    d._teacache_threshold = threshold
    d._feature_cache_requests = {}
    d._feature_cache_epoch_counter = 0
    d._optimization_request_windows = {}
    return d


def _block_cache_denoiser() -> WanDenoiser:
    d = _denoiser(threshold=None)
    d.model = _BranchCountingBlockModel()
    d._feature_cache_config = WanFeatureCacheConfig(
        "dynamicblock",
        {"residual_diff_threshold": 0.1},
    )
    return d


def _block_taylor_denoiser() -> WanDenoiser:
    d = _denoiser(threshold=None)
    d.model = _BranchTaylorPhaseModel()
    d._feature_cache_config = WanFeatureCacheConfig("blocktaylorseer")
    return d


def _input(
    latents,
    step_index,
    total_steps,
    branch="positive",
    request_id="request-a",
) -> DenoiserInput:
    return DenoiserInput(
        latents=latents,
        timestep=torch.zeros(latents.shape[0]),
        next_timestep=torch.zeros(latents.shape[0]),
        conditioning={"context": torch.zeros(1, 4, 8)},
        step_index=step_index,
        total_steps=total_steps,
        branch=branch,
        request_id=request_id,
    )


def _branch_input(
    latents: torch.Tensor,
    step_index: int,
    total_steps: int,
    branch: str,
    context_scale: float,
    request_id: str = "request-a",
) -> DenoiserInput:
    return DenoiserInput(
        latents=latents,
        timestep=torch.zeros(latents.shape[0]),
        next_timestep=torch.zeros(latents.shape[0]),
        conditioning={"context": torch.full((1, 4, 8), context_scale)},
        step_index=step_index,
        total_steps=total_steps,
        branch=branch,
        request_id=request_id,
    )


def _request_caches(d: WanDenoiser, request_id: str) -> dict[str, object]:
    return d._feature_cache_requests[request_id]["caches"]


def test_disabled_by_default_is_dense_every_step() -> None:
    d = _denoiser(threshold=None)
    latents = torch.randn(1, 16, 3, 4, 4)
    total = 6
    with torch.no_grad():
        for step in range(total):
            d(_input(latents, step, total))
    assert d.model.calls == total  # no caching, every step dense
    report = d.feature_cache_report()
    assert report["enabled"] is False
    assert report["effective"] == "disabled"


def test_threshold_skips_interior_steps_but_keeps_boundaries_dense() -> None:
    d = _denoiser(threshold=1e9)  # huge threshold => always eligible to cache
    latents = torch.randn(1, 16, 3, 4, 4)
    total = 6
    with torch.no_grad():
        for step in range(total):
            d(_input(latents, step, total))
    # warmup step 0 (seed) + final boundary step 5 are dense; the interior can
    # be cached, capped by AdaptiveResidualCache.max_consecutive_hits (3).
    assert d.model.calls < total
    assert d.model.calls >= 2  # at least seed + boundary
    report = d.feature_cache_report()
    assert report["enabled"] is True
    assert report["effective"] == "residual-reuse"
    assert report["events"] == total
    assert report["hits"] == total - d.model.calls
    assert report["branches"]["positive"]["events"] == total


def test_cfg_branches_keep_separate_caches() -> None:
    d = _denoiser(threshold=1e9)
    latents = torch.randn(1, 16, 3, 4, 4)
    total = 4
    with torch.no_grad():
        for step in range(total):
            d(_input(latents, step, total, branch="positive"))
            d(_input(latents, step, total, branch="negative"))
    caches = _request_caches(d, "request-a")
    assert set(caches) == {"positive", "negative"}
    # Each branch warms up independently: neither branch's step-0 seed is
    # served from the other's cache.
    for cache in caches.values():
        assert cache.events[0].reason == "seed"


def test_dynamicblock_cfg_and_new_request_cannot_reuse_another_trajectory() -> None:
    d = _block_cache_denoiser()
    latents = torch.zeros(1, 2, 1, 2, 2)
    total = 4

    with torch.no_grad():
        positive_seed = d(
            _branch_input(latents, 0, total, "positive", 1.0)
        ).sample
        negative_seed = d(
            _branch_input(latents, 0, total, "negative", 10.0)
        ).sample
        positive_hit = d(
            _branch_input(latents, 1, total, "positive", 1.0)
        ).sample
        negative_hit = d(
            _branch_input(latents, 1, total, "negative", 10.0)
        ).sample

    torch.testing.assert_close(positive_hit, positive_seed)
    torch.testing.assert_close(negative_hit, negative_seed)
    torch.testing.assert_close(positive_hit, torch.full_like(latents, 6.0))
    torch.testing.assert_close(negative_hit, torch.full_like(latents, 60.0))
    assert d.model.calls == 6
    first_caches = _request_caches(d, "request-a")
    old_positive = first_caches["positive"]
    old_negative = first_caches["negative"]
    assert old_positive is not old_negative
    assert old_positive.receipt()["skipped_block_calls"] == 3
    assert old_negative.receipt()["skipped_block_calls"] == 3
    first_report = d.feature_cache_report()
    assert first_report["request_epoch"] == 1
    assert first_report["request_local"] is True
    assert first_report["skipped_block_calls"] == 6
    for branch in ("positive", "negative"):
        branch_report = first_report["branches"][branch]
        assert branch_report["algorithm"] == "dynamicblock"
        assert branch_report["request_epoch"] == 1
        assert branch_report["request_local"] is True
        assert branch_report["skipped_block_calls"] == 3
        assert branch_report["receipt"]["branch"] == branch
        assert branch_report["receipt"]["request_epoch"] == 1

    # An explicit new identity owns a fresh branch map; the old request stays
    # available for audit and cannot be selected by the new request.
    with torch.no_grad():
        new_seed = d(
            _branch_input(
                latents,
                0,
                total,
                "positive",
                5.0,
                request_id="request-b",
            )
        ).sample
    torch.testing.assert_close(new_seed, torch.full_like(latents, 30.0))
    second_caches = _request_caches(d, "request-b")
    assert set(second_caches) == {"positive"}
    assert second_caches["positive"] is not old_positive
    assert second_caches["positive"].events[0].reason == "seed"
    assert second_caches["positive"].skipped_block_calls == 0
    second_report = d.feature_cache_report()
    assert second_report["request_epoch"] == 2
    assert set(second_report["branches"]) == {"positive"}
    assert second_report["branches"]["positive"]["receipt"]["request_epoch"] == 2
    # Keeping a Python reference to the old cache cannot make its receipt part
    # of the current report or update it to the new request epoch.
    assert old_positive.receipt()["request_epoch"] == 1
    assert old_negative.receipt()["request_epoch"] == 1


def test_feature_cache_a0_b0_a1_interleaving_cannot_cross_request_history() -> None:
    d = _block_cache_denoiser()
    latents = torch.zeros(1, 2, 1, 2, 2)

    with torch.no_grad():
        a0 = d(
            _branch_input(
                latents,
                0,
                4,
                "positive",
                1.0,
                request_id="request-a",
            )
        ).sample
        b0 = d(
            _branch_input(
                latents,
                0,
                4,
                "positive",
                10.0,
                request_id="request-b",
            )
        ).sample
        a1 = d(
            _branch_input(
                latents,
                1,
                4,
                "positive",
                1.0,
                request_id="request-a",
            )
        ).sample

    torch.testing.assert_close(a0, torch.full_like(latents, 6.0))
    torch.testing.assert_close(b0, torch.full_like(latents, 60.0))
    torch.testing.assert_close(a1, a0)
    assert _request_caches(d, "request-a")["positive"] is not (
        _request_caches(d, "request-b")["positive"]
    )
    a_report = d.feature_cache_report("request-a")
    b_report = d.feature_cache_report("request-b")
    assert a_report["request_id"] == "request-a"
    assert a_report["hits"] == 1
    assert b_report["request_id"] == "request-b"
    assert b_report["hits"] == 0


def test_negative_first_new_request_gets_a_fresh_receipt_and_cache() -> None:
    d = _block_cache_denoiser()
    latents = torch.zeros(1, 2, 1, 2, 2)
    with torch.no_grad():
        d(
            _branch_input(
                latents,
                0,
                4,
                "negative",
                10.0,
                request_id="request-a",
            )
        )
        d(
            _branch_input(
                latents,
                1,
                4,
                "negative",
                10.0,
                request_id="request-a",
            )
        )
        new_negative = d(
            _branch_input(
                latents,
                0,
                4,
                "negative",
                5.0,
                request_id="request-b",
            )
        ).sample

    torch.testing.assert_close(new_negative, torch.full_like(latents, 30.0))
    report = d.feature_cache_report("request-b")
    assert report["request_epoch"] == 2
    assert set(report["branches"]) == {"negative"}
    negative = report["branches"]["negative"]
    assert negative["events"] == 1
    assert negative["hits"] == 0
    assert negative["receipt"]["request_id"] == "request-b"
    assert negative["receipt"]["event_receipts"][0]["reason"] == "seed"


def test_feature_cache_refuses_missing_request_identity() -> None:
    d = _block_cache_denoiser()
    model_input = _branch_input(
        torch.zeros(1, 2, 1, 2, 2),
        0,
        4,
        "negative",
        1.0,
        request_id="request-a",
    ).with_updates(request_id=None)
    with pytest.raises(ValueError, match="explicit non-empty request_id"):
        d(model_input)


def test_end_request_snapshots_receipt_and_releases_live_tensor_cache() -> None:
    d = _block_cache_denoiser()
    latents = torch.zeros(1, 2, 1, 2, 2)
    with torch.no_grad():
        d(
            _branch_input(
                latents,
                0,
                2,
                "positive",
                1.0,
                request_id="release-me",
            )
        )
        d(
            _branch_input(
                latents,
                1,
                2,
                "positive",
                1.0,
                request_id="release-me",
            )
        )
    cache = _request_caches(d, "release-me")["positive"]
    assert cache.events
    d.end_request("release-me")

    assert d.feature_cache_lifecycle_report() == {
        "live_requests": 0,
        "receipt_snapshots": 1,
        "max_receipt_snapshots": 32,
    }
    assert cache.events == []
    assert cache._block_inputs == {}
    assert cache._block_residuals == {}
    snapshot = d.feature_cache_report("release-me")
    assert snapshot["finalized"] is True
    assert snapshot["release_reason"] == "completed"
    assert snapshot["hits"] == 1
    # Returned reports are decoded copies; callers cannot mutate storage.
    snapshot["hits"] = 999
    assert d.feature_cache_report("release-me")["hits"] == 1


def test_end_request_error_path_releases_cache_and_snapshot_history_is_bounded() -> None:
    d = _block_cache_denoiser()
    latents = torch.zeros(1, 2, 1, 2, 2)
    for index in range(36):
        request_id = f"bounded-{index}"
        with torch.no_grad():
            d(
                _branch_input(
                    latents,
                    0,
                    1,
                    "negative",
                    float(index + 1),
                    request_id=request_id,
                )
            )
        error = RuntimeError("abort") if index == 35 else None
        d.end_request(request_id, error=error)

    lifecycle = d.feature_cache_lifecycle_report()
    assert lifecycle["live_requests"] == 0
    assert lifecycle["receipt_snapshots"] == 32
    assert len(d._feature_cache_receipt_snapshots) == 32
    assert "bounded-0" not in d._feature_cache_receipt_snapshots
    failed = d.feature_cache_report("bounded-35")
    assert failed["release_reason"] == "error"
    assert failed["error_type"] == "RuntimeError"


def test_block_taylor_cfg_receipts_and_phase_history_are_request_local() -> None:
    d = _block_taylor_denoiser()
    latents = torch.zeros(1, 2, 1, 2, 2)
    total = 4

    with torch.no_grad():
        positive_seed = d(
            _branch_input(latents, 0, total, "positive", 1.0)
        ).sample
        negative_seed = d(
            _branch_input(latents, 0, total, "negative", 10.0)
        ).sample
        positive_hit = d(
            _branch_input(latents, 1, total, "positive", 1.0)
        ).sample
        negative_hit = d(
            _branch_input(latents, 1, total, "negative", 10.0)
        ).sample

    torch.testing.assert_close(positive_seed, torch.full_like(latents, 9.0))
    torch.testing.assert_close(negative_seed, torch.full_like(latents, 90.0))
    torch.testing.assert_close(positive_hit, positive_seed)
    torch.testing.assert_close(negative_hit, negative_seed)
    assert d.model.dense_phase_calls == 4
    first_report = d.feature_cache_report()
    assert first_report["algorithm"] == "blocktaylorseer"
    assert first_report["request_epoch"] == 1
    assert first_report["request_local"] is True
    assert first_report["skipped_block_calls"] == 4
    first_caches = _request_caches(d, "request-a")
    old_positive = first_caches["positive"]
    old_negative = first_caches["negative"]
    for branch in ("positive", "negative"):
        branch_report = first_report["branches"][branch]
        assert branch_report["request_local"] is True
        assert branch_report["receipt"]["prediction_scope"] == (
            "per-block-self-cross-ffn"
        )
        assert branch_report["receipt"]["branch"] == branch
        assert branch_report["receipt"]["request_epoch"] == 1

    with torch.no_grad():
        new_seed = d(
            _branch_input(
                latents,
                0,
                total,
                "positive",
                5.0,
                request_id="request-b",
            )
        ).sample
    torch.testing.assert_close(new_seed, torch.full_like(latents, 45.0))
    second_caches = _request_caches(d, "request-b")
    assert set(second_caches) == {"positive"}
    assert second_caches["positive"] is not old_positive
    assert second_caches["positive"].events[0].reason == "seed"
    assert old_positive.receipt()["request_epoch"] == 1
    assert old_negative.receipt()["request_epoch"] == 1
    assert d.feature_cache_report()["request_epoch"] == 2


def test_autograd_forces_dense() -> None:
    d = _denoiser(threshold=1e9)
    latents = torch.randn(1, 16, 3, 4, 4, requires_grad=True)
    total = 5
    with torch.enable_grad():
        for step in range(total):
            d(_input(latents, step, total))
    assert d.model.calls == total  # grad enabled => never cache


def test_runtime_report_merges_actual_teacache_and_graph_state() -> None:
    d = _denoiser(threshold=1e9)
    latents = torch.randn(1, 16, 3, 4, 4)
    with torch.no_grad():
        for step in range(4):
            d(_input(latents, step, 4))

    report = d.runtime_optimization_report()
    assert report["requested"]["cuda_graph"] is False
    assert report["requested"]["teacache"] == 1e9
    assert report["effective"]["teacache"] == "residual-reuse"
    runtime = report["runtime"]
    assert runtime["cuda_graph"] is None
    assert runtime["teacache"]["hits"] > 0
    assert "kernel_dispatch" in runtime
    assert "attention_dispatch" in runtime


def test_request_reset_preserves_lifetime_compile_trace_but_clears_execution() -> None:
    d = _denoiser(threshold=None)
    d.model._worldfoundry_compile_runtime = {
        "wrapper_installed": True,
        "calls": 7,
        "failures": 1,
        "last_error": "old request",
        "request_calls": 3,
        "request_failures": 1,
        "request_last_error": "old request",
        "attention_provider_graph_traces": {"flash_attention_3": 2},
    }
    d.model._worldfoundry_qkv_fusion = QKVFusionState(
        fused_blocks=1,
        eager_projection_calls=5,
        compiled_graph_traces=2,
        request_eager_projection_calls=3,
        request_compiled_graph_traces=1,
    )

    d._reset_request_optimization_state()

    stored = d.model._worldfoundry_compile_runtime
    assert stored["calls"] == 7
    assert stored["failures"] == 1
    assert stored["request_calls"] == 0
    assert stored["request_failures"] == 0
    assert stored["request_last_error"] is None
    report = d.runtime_optimization_report()
    compile_report = report["runtime"]["compile"]
    assert compile_report["calls"] == 0
    assert compile_report["failures"] == 0
    assert compile_report["lifetime_calls"] == 7
    assert compile_report["lifetime_failures"] == 1
    assert compile_report["attention_provider_graph_traces"] == {
        "flash_attention_3": 2
    }
    qkv_report = report["runtime"]["qkv_fusion"]
    assert qkv_report["eager_projection_calls"] == 0
    assert qkv_report["compiled_graph_traces"] == 2
    assert qkv_report["request_compiled_graph_traces"] == 0
