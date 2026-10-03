"""EasyCache controller invariants, including mutation and training isolation."""

import pytest
import torch

from worldfoundry.core.acceleration.easycache import EasyCache, EasyCacheConfig


@pytest.mark.parametrize("threshold", [0.0, 0.2])
def test_online_decision_and_dense_boundaries(threshold):
    cache = EasyCache(EasyCacheConfig(threshold=threshold, max_skip_steps=2), total_steps=8)
    calls = []

    def run(index, value):
        calls.append(index)
        return value * 2 + 1

    with torch.inference_mode():
        for step in range(8):
            value = torch.full((2, 4, 8), 1.0 + step * 0.001)
            actual = cache.run_blocks(step, value, run, block_count=2)
            if not cache.events[-1].hit:
                torch.testing.assert_close(actual, (value * 2 + 1) * 2 + 1, rtol=0, atol=0)
    assert not any(event.hit for event in cache.events[:2])
    assert not cache.events[-1].hit
    assert sum(event.hit for event in cache.events) == (0 if threshold == 0 else 4)
    assert len(calls) == cache.dense_block_calls
    assert cache.dense_block_calls + cache.skipped_block_calls == 16
    assert cache._factor is not None
    cache.reset()
    assert cache._residual is None and not cache.events


def test_snapshots_survive_caller_mutations_and_layout_changes():
    cache = EasyCache(EasyCacheConfig(0.2), total_steps=4)
    with torch.inference_mode():
        value = torch.ones(1, 8, 4)
        result = cache.run_blocks(0, value, lambda i, x: x * 2, block_count=1)
        value.add_(3)
        result.zero_()
        torch.testing.assert_close(cache._dense_input, torch.ones(1, 32))
        torch.testing.assert_close(cache._residual, torch.ones(1, 8, 4))
        cache.run_blocks(1, value, lambda i, x: x * 2, block_count=1)
        cache.run_blocks(2, torch.ones(1, 4, 4), lambda i, x: x * 2, block_count=1)
    assert cache.events[-1].reason == "layout-change"
    assert not cache.events[-1].hit


def test_training_never_reuses_detached_state():
    cache = EasyCache(EasyCacheConfig(10.0), total_steps=4)
    with torch.inference_mode():
        for step in range(2):
            cache.run_blocks(step, torch.ones(1, 4), lambda i, x: x * 2, block_count=1)
    value = torch.ones(1, 4, requires_grad=True)
    actual = cache.run_blocks(2, value, lambda i, x: x.square(), block_count=1)
    actual.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 2.0))
    assert cache.events[-1].reason == "autograd" and cache._residual is None


def test_condition_snapshot_detects_inplace_edits_to_inference_tensors():
    cache = EasyCache(EasyCacheConfig(10.0, dense_last=0), total_steps=5)
    with torch.inference_mode():
        condition = torch.ones(1, 4)
        values = {"context": condition, "info": {"frames": [0, 1]}}
        for step in range(3):
            cache.observe_conditioning(values)
            cache.run_blocks(step, torch.ones(1, 4), lambda i, x: x + condition, block_count=1)
        assert cache.events[-1].hit
        condition.mul_(2)
        cache.observe_conditioning(values)
        output = cache.run_blocks(3, torch.ones(1, 4), lambda i, x: x + condition, block_count=1)
        assert not cache.events[-1].hit and cache.events[-1].reason == "conditioning-change"
        torch.testing.assert_close(output, torch.full((1, 4), 3.0), rtol=0, atol=0)
    cache.reset()
    assert cache._conditioning is None


def test_invalid_step_and_payload_fail_before_reuse():
    cache = EasyCache(EasyCacheConfig(0.1), total_steps=3)
    with pytest.raises(ValueError, match="contiguous"):
        cache.run_blocks(1, torch.ones(1, 4), lambda i, x: x, block_count=1)
    with pytest.raises(ValueError, match="preserve"):
        cache.run_blocks(0, torch.ones(1, 4), lambda i, x: x[:, :2], block_count=1)


def test_subsampling_does_not_hide_nonfinite_input():
    cache = EasyCache(EasyCacheConfig(10.0, dense_last=0, subsample_stride=2), total_steps=4)
    with torch.inference_mode():
        for step in range(2):
            cache.run_blocks(step, torch.ones(1, 8), lambda i, x: x * 2, block_count=1)
        value = torch.ones(1, 8)
        value[0, 1] = float("nan")
        cache.run_blocks(2, value, lambda i, x: x * 2, block_count=1)
    assert cache.events[-1].reason == "nonfinite-input" and not cache.events[-1].hit
    assert cache._residual is None


@pytest.mark.parametrize(
    "options",
    [
        {"threshold": -1},
        {"threshold": float("nan")},
        {"threshold": 0.1, "subsample_stride": 0},
        {"threshold": 0.1, "max_skip_steps": True},
    ],
)
def test_config_rejects_invalid_budgets(options):
    with pytest.raises((TypeError, ValueError)):
        EasyCacheConfig(**options)
