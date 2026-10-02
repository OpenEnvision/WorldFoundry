"""Tests for the inference CUDA Graph runner.

The correctness/eligibility/fallback behaviour is exercised on CPU (no capture);
GPU capture/replay parity is covered by the H100 microbenchmark scripts.
"""

from __future__ import annotations

from collections import namedtuple

import torch

from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner


def test_cpu_inputs_fall_back_to_eager() -> None:
    calls = {"n": 0}

    def fn(x: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return x * 2 + 1

    runner = InferenceCUDAGraphRunner(fn)
    x = torch.randn(4, 8)
    out = runner(x)
    torch.testing.assert_close(out, x * 2 + 1)
    # CPU tensor is not graph-eligible, so it must run eagerly.
    assert runner.report()["eager"] == 1
    assert runner.report()["capture"] == 0
    assert calls["n"] == 1


def test_disabled_runner_is_transparent() -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x.sin()

    runner = InferenceCUDAGraphRunner(fn, enabled=False)
    x = torch.randn(3, 3)
    torch.testing.assert_close(runner(x), x.sin())
    assert runner.report()["enabled"] is False


def test_kwargs_do_not_break_correctness_on_cpu() -> None:
    def fn(x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return x * scale

    runner = InferenceCUDAGraphRunner(fn)
    x = torch.randn(2, 2)
    # Kwargs are supported (folded into the signature); the CPU tensor is what
    # forces the eager path here, not the presence of a keyword argument.
    torch.testing.assert_close(runner(x, scale=3.0), x * 3.0)
    assert runner.report()["eager"] == 1


def test_grad_inputs_fall_back() -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    runner = InferenceCUDAGraphRunner(fn)
    x = torch.randn(2, 2, requires_grad=True)
    out = runner(x)
    torch.testing.assert_close(out, x + 1)
    assert runner.report()["eager"] == 1


def test_report_shape() -> None:
    runner = InferenceCUDAGraphRunner(lambda x: x)
    report = runner.report()
    for key in ("enabled", "graphs", "max_graphs", "capture", "replay", "eager", "capture_failed"):
        assert key in report


def test_request_window_counters_do_not_reuse_previous_execution() -> None:
    runner = InferenceCUDAGraphRunner(lambda x: x + 1)
    x = torch.randn(2, 2)
    runner(x)
    assert runner.report()["eager"] == 1

    runner.begin_request_window()
    empty_window = runner.report()
    assert empty_window["window_id"] == 1
    assert empty_window["eager"] == 0
    assert empty_window["capture"] == 0
    assert empty_window["replay"] == 0
    assert empty_window["lifetime"]["eager"] == 1

    runner(x)
    current = runner.report()
    assert current["eager"] == 1
    assert current["lifetime"]["eager"] == 2


def test_materialize_clones_nested_outputs_and_preserves_container_types() -> None:
    result_type = namedtuple("Result", ("prediction", "label"))
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0, 4.0])
    outputs = {"nested": ([first], {"result": result_type(second, "prediction")}), "metadata": 3}
    runner = InferenceCUDAGraphRunner(lambda: outputs)

    retained = runner._materialize(outputs)
    first.add_(10)
    second.add_(20)

    assert isinstance(retained, dict)
    assert isinstance(retained["nested"], tuple)
    assert isinstance(retained["nested"][0], list)
    assert isinstance(retained["nested"][1]["result"], result_type)
    assert retained["nested"][1]["result"].label == "prediction"
    assert retained["metadata"] == 3
    torch.testing.assert_close(retained["nested"][0][0], torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(retained["nested"][1]["result"].prediction, torch.tensor([3.0, 4.0]), rtol=0, atol=0)
    assert retained["nested"][0][0].data_ptr() != first.data_ptr()
    assert retained["nested"][1]["result"].prediction.data_ptr() != second.data_ptr()


def test_materialize_exposes_static_outputs_only_when_requested() -> None:
    outputs = {"nested": [torch.ones(2)]}
    runner = InferenceCUDAGraphRunner(lambda: outputs, clone_outputs=False)

    assert runner._materialize(outputs) is outputs
