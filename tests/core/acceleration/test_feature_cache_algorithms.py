from __future__ import annotations

import pytest
import torch

from worldfoundry.core.acceleration.cache import (
    AdaCacheResidualCache,
    BlockTaylorSeerCache,
    CustomTaylorResidualCache,
    DualBlockFeatureCache,
    DynamicBlockFeatureCache,
    FirstBlockFeatureCache,
    MagCacheResidualCache,
    TaylorSeerResidualCache,
    TeaCacheResidualCache,
)


def _computed(counter: list[int], value: float = 1.0) -> torch.Tensor:
    counter[0] += 1
    return torch.full((1, 2), value * counter[0])


def _counting_blocks(calls: list[int]):
    def run_block(block_id: int, hidden: torch.Tensor) -> torch.Tensor:
        calls[block_id] += 1
        return hidden + float(block_id + 1)

    return run_block


def test_teacache_uses_polynomial_threshold_and_dense_boundaries() -> None:
    counter = [0]
    cache = TeaCacheResidualCache(
        threshold=0.5,
        coefficients=(1.0, 0.0),
        warmup_steps=1,
        dense_last=1,
        total_steps=4,
    )
    signals = (
        torch.ones(1, 2),
        torch.full((1, 2), 1.1),
        torch.full((1, 2), 1.2),
        torch.full((1, 2), 1.3),
    )
    with torch.no_grad():
        outputs = [
            cache.run(step, signal, lambda: _computed(counter), total_steps=4)
            for step, signal in enumerate(signals)
        ]
    assert counter[0] == 2
    assert [event.hit for event in cache.events] == [False, True, True, False]
    torch.testing.assert_close(outputs[1], outputs[0])
    assert cache.events[1].reason == "polynomial-below-threshold"


def test_magcache_replays_with_calibrated_ratio_budget() -> None:
    counter = [0]
    cache = MagCacheResidualCache(
        (1.0, 0.99, 0.99, 0.5),
        threshold=0.1,
        max_skip_steps=2,
        retention_ratio=0.0,
        total_steps=4,
    )
    with torch.no_grad():
        for step in range(4):
            cache.run(step, torch.zeros(1), lambda: _computed(counter), total_steps=4)
    assert [event.hit for event in cache.events] == [False, True, True, False]
    assert counter[0] == 2


def test_adacache_uses_mid_stack_probe_to_schedule_future_hits() -> None:
    counter = [0]
    cache = AdaCacheResidualCache(total_steps=6, dense_last=1)

    def compute() -> torch.Tensor:
        cache.observe_probe(torch.full((1, 2), 1.0 + counter[0] * 0.001))
        return _computed(counter)

    with torch.no_grad():
        for step in range(6):
            cache.run(step, torch.zeros(1), compute, total_steps=6)
    assert any(event.hit for event in cache.events)
    assert any(event.reason.startswith("adaptive-rate-") for event in cache.events)
    assert cache.events[-1].reason == "dense-boundary"


def test_taylorseer_extrapolates_between_dense_steps() -> None:
    counter = [0]
    cache = TaylorSeerResidualCache(interval=2, dense_last=0, total_steps=5)
    with torch.no_grad():
        outputs = [
            cache.run(step, torch.zeros(1), lambda: _computed(counter), total_steps=5)
            for step in range(5)
        ]
    assert [event.hit for event in cache.events] == [False, True, False, True, False]
    assert counter[0] == 3
    # The first hit has no derivative yet and therefore reuses the seed.
    torch.testing.assert_close(outputs[1], outputs[0])
    # After two dense points the next hit advances by the learned derivative.
    torch.testing.assert_close(outputs[3], torch.full((1, 2), 2.5))


def test_block_taylorseer_predicts_each_phase_on_lightx2v_dense_pattern() -> None:
    cache = BlockTaylorSeerCache(total_steps=6)
    block_input = torch.zeros(1, 2, 3)
    current_step = [0]
    dense_phase_calls = [0, 0]
    cached_phase_calls = [0, 0]

    def run_block(block_id: int, hidden: torch.Tensor) -> torch.Tensor:
        raise AssertionError(f"unexpected generic dense block {block_id}: {hidden.shape}")

    def run_phase_block(block_id, hidden, cached_phases):
        if cached_phases is None:
            dense_phase_calls[block_id] += 1
            base = float(current_step[0] * 2 + block_id + 1)
            phases = {
                "self_attn_out": torch.full_like(hidden, base),
                "cross_attn_out": torch.full_like(hidden, base + 1.0),
                "ffn_out": torch.full_like(hidden, base + 2.0),
            }
        else:
            cached_phase_calls[block_id] += 1
            phases = cached_phases
        return hidden + sum(phases.values()), phases

    outputs = []
    with torch.no_grad():
        for step in range(6):
            current_step[0] = step
            outputs.append(
                cache.run_phase_blocks(
                    step,
                    block_input,
                    run_block,
                    run_phase_block,
                    block_count=2,
                    total_steps=6,
                )
            )

    assert [event.hit for event in cache.events] == [
        False,
        True,
        True,
        True,
        False,
        True,
    ]
    assert dense_phase_calls == [2, 2]
    assert cached_phase_calls == [4, 4]
    # Step 1 has no learned slope yet and replays step 0 phase values.
    torch.testing.assert_close(outputs[1], outputs[0])
    # Dense steps 0 and 4 differ by 8 per phase, so the learned per-step
    # derivative is 2 and step 5 advances every phase by exactly 2.
    torch.testing.assert_close(outputs[5] - outputs[4], torch.full_like(outputs[4], 12.0))
    receipt = cache.receipt()
    assert receipt["algorithm"] == "blocktaylorseer"
    assert receipt["prediction_scope"] == "per-block-self-cross-ffn"
    assert receipt["dense_pattern"] == [True, False, False, False]
    assert receipt["dense_block_calls"] == 4
    assert receipt["skipped_block_calls"] == 8


def test_block_taylorseer_autograd_is_dense_and_does_not_mutate_phase_history() -> None:
    cache = BlockTaylorSeerCache()
    block_input = torch.zeros(1, 2, 3)
    phase_calls = [0]
    dense_calls = [0]

    def run_block(_block_id: int, hidden: torch.Tensor) -> torch.Tensor:
        dense_calls[0] += 1
        return hidden + 10.0

    def run_phase_block(_block_id, hidden, cached_phases):
        phase_calls[0] += 1
        phases = cached_phases or {
            "self_attn_out": torch.ones_like(hidden),
            "cross_attn_out": torch.ones_like(hidden),
            "ffn_out": torch.ones_like(hidden),
        }
        return hidden + sum(phases.values()), phases

    with torch.no_grad():
        cache.run_phase_blocks(
            0,
            block_input,
            run_block,
            run_phase_block,
            block_count=1,
        )
    grad_input = block_input.clone().requires_grad_()
    cache.run_phase_blocks(
        1,
        grad_input,
        run_block,
        run_phase_block,
        block_count=1,
    ).sum().backward()
    with torch.no_grad():
        hit = cache.run_phase_blocks(
            2,
            block_input,
            run_block,
            run_phase_block,
            block_count=1,
        )

    assert dense_calls == [1]
    assert phase_calls == [2]
    assert [event.reason for event in cache.events] == [
        "seed",
        "autograd",
        "taylor-extrapolation",
    ]
    torch.testing.assert_close(hit, torch.full_like(hit, 3.0))


def test_custom_cache_combines_tea_decision_with_taylor_residual() -> None:
    cache = CustomTaylorResidualCache(
        threshold=0.15,
        coefficients=(1.0, 0.0),
        warmup_steps=1,
        dense_last=0,
        total_steps=4,
    )
    dense_calls = [0]
    current_step = [0]

    def compute() -> torch.Tensor:
        dense_calls[0] += 1
        return torch.full((1, 2), 10.0 + 2.0 * current_step[0])

    signals = [torch.full((1, 2), value) for value in (1.0, 1.1, 1.2, 1.3)]
    outputs = []
    with torch.no_grad():
        for step, signal in enumerate(signals):
            current_step[0] = step
            outputs.append(cache.run(step, signal, compute, total_steps=4))

    assert [event.hit for event in cache.events] == [False, True, False, True]
    assert dense_calls == [2]
    torch.testing.assert_close(outputs[1], outputs[0])
    # Dense residuals are 10 at step 0 and 14 at step 2: slope 2/step.
    torch.testing.assert_close(outputs[3], torch.full((1, 2), 16.0))
    assert cache.receipt()["prediction"] == "first-order-stack-residual"


def test_firstblock_runs_probe_but_reuses_the_remaining_block_residual() -> None:
    calls = [0] * 4
    cache = FirstBlockFeatureCache(0.1, downsample_factor=2)
    run_block = _counting_blocks(calls)
    block_input = torch.zeros(1, 3, 4)

    with torch.no_grad():
        expected = cache.run_blocks(0, block_input, run_block, block_count=4)
        actual = cache.run_blocks(1, block_input, run_block, block_count=4)

    torch.testing.assert_close(actual, expected)
    assert calls == [2, 1, 1, 1]
    assert cache.events[-1].algorithm == "firstblock"
    assert cache.events[-1].dense_blocks == (0,)
    assert cache.events[-1].skipped_blocks == (1, 2, 3)
    assert cache.receipt()["skipped_block_calls"] == 3


def test_dualblock_runs_front_and_back_while_reusing_only_the_middle() -> None:
    calls = [0] * 11
    cache = DualBlockFeatureCache(0.1)
    run_block = _counting_blocks(calls)
    block_input = torch.zeros(1, 3, 4)

    with torch.no_grad():
        expected = cache.run_blocks(0, block_input, run_block, block_count=11)
        actual = cache.run_blocks(1, block_input, run_block, block_count=11)

    torch.testing.assert_close(actual, expected)
    assert calls == [2] * 5 + [1] + [2] * 5
    assert cache.events[-1].dense_blocks == tuple(range(5)) + tuple(range(6, 11))
    assert cache.events[-1].skipped_blocks == (5,)
    with torch.no_grad(), pytest.raises(ValueError, match="at least 11 blocks"):
        cache.run_blocks(2, block_input, run_block, block_count=10)


def test_dynamicblock_makes_and_reports_real_per_block_skip_decisions() -> None:
    calls = [0] * 4
    cache = DynamicBlockFeatureCache(0.1)
    run_block = _counting_blocks(calls)
    block_input = torch.zeros(1, 3, 4)

    with torch.no_grad():
        expected = cache.run_blocks(0, block_input, run_block, block_count=4)
        actual = cache.run_blocks(1, block_input, run_block, block_count=4)

    torch.testing.assert_close(actual, expected)
    assert calls == [1, 1, 1, 1]
    assert cache.events[-1].hit is True
    assert cache.events[-1].dense_blocks == ()
    assert cache.events[-1].skipped_blocks == (0, 1, 2, 3)
    assert cache.receipt() == {
        "algorithm": "dynamicblock",
        "residual_diff_threshold": 0.1,
        "downsample_factor": 1,
        "dense_first": 1,
        "dense_last": 0,
        "events": 2,
        "hits": 1,
        "dense_block_calls": 4,
        "skipped_block_calls": 4,
        "event_receipts": [event.receipt() for event in cache.events],
    }


@pytest.mark.parametrize(
    ("cache", "block_count"),
    (
        (FirstBlockFeatureCache(0.1), 4),
        (DualBlockFeatureCache(0.1), 11),
        (DynamicBlockFeatureCache(0.1), 4),
    ),
)
def test_block_feature_cache_missing_or_incompatible_state_is_dense(
    cache,
    block_count: int,
) -> None:
    calls = [0] * block_count
    run_block = _counting_blocks(calls)

    with torch.no_grad():
        cache.run_blocks(3, torch.zeros(1, 2, 4), run_block, block_count=block_count)
        cache.run_blocks(4, torch.zeros(1, 3, 4), run_block, block_count=block_count)

    assert calls == [2] * block_count
    assert cache.events[0].reason == "cache-missing"
    assert cache.events[1].reason == "incompatible-state"
    assert all(event.skipped_blocks == () for event in cache.events)


@pytest.mark.parametrize(
    ("cache", "block_count"),
    (
        (FirstBlockFeatureCache(0.1), 4),
        (DualBlockFeatureCache(0.1), 11),
        (DynamicBlockFeatureCache(0.1), 4),
    ),
)
def test_block_feature_cache_non_finite_metric_fails_closed_to_dense(
    cache,
    block_count: int,
) -> None:
    calls = [0] * block_count
    run_block = _counting_blocks(calls)
    with torch.no_grad():
        cache.run_blocks(0, torch.zeros(1, 2, 4), run_block, block_count=block_count)
        cache.run_blocks(
            1,
            torch.full((1, 2, 4), float("nan")),
            run_block,
            block_count=block_count,
        )

    assert calls == [2] * block_count
    assert cache.events[-1].reason == "non-finite-metric"
    assert cache.events[-1].skipped_blocks == ()


def test_block_feature_cache_dense_step_boundaries_are_explicit_and_enforced() -> None:
    calls = [0] * 4
    cache = FirstBlockFeatureCache(
        0.1,
        dense_first=2,
        dense_last=1,
    )
    run_block = _counting_blocks(calls)
    block_input = torch.zeros(1, 2, 4)

    with torch.no_grad():
        for step in range(4):
            cache.run_blocks(
                step,
                block_input,
                run_block,
                block_count=4,
                total_steps=4,
            )

    assert [event.reason for event in cache.events] == [
        "seed",
        "dense-boundary",
        "below-threshold",
        "dense-boundary",
    ]
    assert [event.hit for event in cache.events] == [False, False, True, False]
    assert calls == [4, 3, 3, 3]
    assert cache.receipt()["dense_first"] == 2
    assert cache.receipt()["dense_last"] == 1


def test_block_feature_cache_dense_last_requires_total_steps() -> None:
    cache = DynamicBlockFeatureCache(0.1, dense_last=1)
    with torch.no_grad(), pytest.raises(ValueError, match="total_steps"):
        cache.run_blocks(
            1,
            torch.zeros(1, 2, 4),
            _counting_blocks([0, 0]),
            block_count=2,
        )


@pytest.mark.parametrize(
    ("cache", "block_count"),
    (
        (FirstBlockFeatureCache(0.1), 4),
        (DualBlockFeatureCache(0.1), 11),
        (DynamicBlockFeatureCache(0.1), 4),
    ),
)
def test_block_feature_caches_force_dense_without_mutating_state_under_autograd(
    cache,
    block_count: int,
) -> None:
    calls = [0] * block_count
    run_block = _counting_blocks(calls)
    seed = torch.zeros(1, 3, 4)
    with torch.no_grad():
        cache.run_blocks(0, seed, run_block, block_count=block_count)

    before_grad = calls.copy()
    grad_input = torch.full((1, 3, 4), 100.0, requires_grad=True)
    output = cache.run_blocks(1, grad_input, run_block, block_count=block_count)
    output.sum().backward()
    assert [after - before for after, before in zip(calls, before_grad, strict=True)] == [
        1
    ] * block_count
    assert grad_input.grad is not None
    assert cache.events[-1].reason == "autograd"
    assert cache.events[-1].skipped_blocks == ()

    with torch.no_grad():
        cache.run_blocks(2, seed, run_block, block_count=block_count)
    assert cache.events[-1].hit is True


@pytest.mark.parametrize(
    "cache_type",
    (FirstBlockFeatureCache, DualBlockFeatureCache, DynamicBlockFeatureCache),
)
@pytest.mark.parametrize("threshold", (-0.1, float("nan"), float("inf")))
def test_block_feature_cache_rejects_unsafe_thresholds(cache_type, threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        cache_type(threshold)
