"""Synthetic benchmarks must fail before timing, even when cosine is one."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.inference.correctness import (
    EXACT_BUDGET,
    FP8_BUDGET,
    NumericalBudget,
    compare_step,
    require_graph_execution,
    validate_steps,
)


def test_scaled_output_fails_despite_perfect_cosine():
    reference = torch.arange(1, 33, dtype=torch.float32)
    actual = reference * 2
    assert torch.nn.functional.cosine_similarity(reference, actual, dim=0).item() == pytest.approx(1)
    with pytest.raises(AssertionError, match="step 7"):
        compare_step(reference, actual, step=7, budget=FP8_BUDGET)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_output_fails(value):
    with pytest.raises(AssertionError, match="NaN/Inf"):
        compare_step(torch.ones(3), torch.tensor([1.0, value, 1.0]), step=0)


def test_shape_and_empty_outputs_fail():
    for ref, got in [(torch.ones(2, 2), torch.ones(4)), (torch.empty(0), torch.empty(0))]:
        with pytest.raises(AssertionError, match="shapes"):
            compare_step(ref, got, step=0)


def test_explicit_relative_error_budget_and_metrics():
    reference = torch.tensor([1.0, 2.0])
    metrics = compare_step(reference, reference.clone(), step=3)
    assert metrics == {"step": 3, "relative_l2": 0.0, "rmse": 0.0, "max_abs": 0.0}
    with pytest.raises(AssertionError, match="relative L2"):
        compare_step(reference, reference + 0.01, step=0, budget=NumericalBudget(1, 1, 0.001))
    with pytest.raises(AssertionError):
        compare_step(reference, reference + 0.01, step=0, budget=EXACT_BUDGET)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_empty_and_invalid_steps_fail(steps):
    with pytest.raises(ValueError, match="positive integer"):
        validate_steps(steps)


@pytest.mark.parametrize("values", [(-1, 0, 0), (0, float("nan"), 0), (0, 0, float("inf"))])
def test_invalid_budget_fails(values):
    with pytest.raises(ValueError):
        NumericalBudget(*values)


@pytest.mark.parametrize(
    "change", [{"eager": 1}, {"capture_failed": 1}, {"replay": 1}, {"graphs": 0}, {"enabled": False}]
)
def test_graph_fallback_or_lifetime_only_evidence_fails(change):
    report = {
        "enabled": True,
        "graphs": 1,
        "replay": 3,
        "eager": 0,
        "capture_failed": 0,
        "lifetime": {"capture": 99, "replay": 999},
    }
    require_graph_execution(report, steps=3)
    with pytest.raises(AssertionError, match="did not execute"):
        require_graph_execution({**report, **change}, steps=3)


def test_unsupported_cache_graph_combination_fails_before_cuda_or_timing(monkeypatch):
    from benchmarks.inference import full_stack

    monkeypatch.setattr(full_stack, "bench_paired", lambda *a, **k: pytest.fail("timing must not start"))
    with pytest.raises(ValueError, match="incompatible"):
        full_stack.run(cuda_graph=True, static_cross_kv=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_denoise_benchmark_checks_every_step_before_timing(monkeypatch):
    from benchmarks.inference import denoise_loop

    monkeypatch.setattr(denoise_loop, "_CASES", [(2, 2, 2, 2, 2, "tiny")])
    seen = []

    def timing(name, eager, graph, **kwargs):
        workload = kwargs["workload"]
        assert [item["step"] for item in workload["correctness"]] == [0, 1, 2]
        require_graph_execution(workload["graph_report"], steps=3)
        torch.testing.assert_close(eager(), graph())
        seen.append(name)
        return SimpleNamespace(speedup_median=1, speedup_ci_low=1, speedup_ci_high=1)

    monkeypatch.setattr(denoise_loop, "bench_paired", timing)
    denoise_loop.run(steps=3, dtype=torch.float32)
    assert seen == ["tiny"]
    original = denoise_loop._GraphModel.forward
    calls = []

    def corrupt_intermediate(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        calls.append(1)
        return out * 2 if len(calls) == 2 else out

    monkeypatch.setattr(denoise_loop._GraphModel, "forward", corrupt_intermediate)
    with pytest.raises(AssertionError, match="step 1"):
        denoise_loop.run(steps=3, dtype=torch.float32)
    assert seen == ["tiny"]  # No timing was permitted on the corrupted trajectory.


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph", [False, True])
def test_full_stack_actual_fp8_cache_or_graph_execution(monkeypatch, graph):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 correctness contract requires SM90+")
    from benchmarks.inference import full_stack

    # Exercise real scaled-mm even for the tiny correctness fixture; this
    # deliberately bypasses the performance threshold, not numerical gates.
    monkeypatch.setenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP", "0")
    monkeypatch.setitem(
        full_stack._PROFILES,
        "tiny",
        {
            "dim": 512,
            "heads": 8,
            "ffn": 1024,
            "n_blocks": 1,
            "seq": 128,
            "ctx": 16,
            "steps": 3,
        },
    )
    seen = []

    def timing(name, baseline, optimized, **kwargs):
        workload = kwargs["workload"]
        assert len(workload["correctness"]) == 3
        assert workload["quantization_report"]["low_precision_kernel_calls"] > 0
        if graph:
            require_graph_execution(workload["graph_report"], steps=3)
        else:
            assert workload["cross_kv_report"]["hits"] >= 2
        seen.append(name)
        return SimpleNamespace(speedup_median=1, speedup_ci_low=1, speedup_ci_high=1)

    monkeypatch.setattr(full_stack, "bench_paired", timing)
    full_stack.run("tiny", cuda_graph=graph, static_cross_kv=not graph)
    assert seen == ["full_stack_tiny"]
