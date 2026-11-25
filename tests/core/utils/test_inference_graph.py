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


def test_nested_input_signatures_depend_on_structure_layout_not_values() -> None:
    from worldfoundry.core.execution.graphs.inference_graph import _arg_signature, _flatten_tensors

    first = {"kv": [(torch.ones(2, 3), torch.zeros(2, 3))], "scale": 2.0}
    second = {"kv": [(torch.zeros(2, 3), torch.ones(2, 3))], "scale": 2.0}
    assert _arg_signature(first) == _arg_signature(second)
    assert len(_flatten_tensors((first,))) == 2
    second["scale"] = 3.0
    assert _arg_signature(first) != _arg_signature(second)
    second["scale"] = 2.0
    second["kv"] = [(torch.ones(3, 2).t(), torch.ones(2, 3))]
    assert _arg_signature(first) != _arg_signature(second)


def test_capture_scope_restores_frozen_policy_after_error() -> None:
    import pytest

    runner = InferenceCUDAGraphRunner(lambda x: x + 1)
    runner.freeze()
    with pytest.raises(RuntimeError, match="warmup"):
        with runner.capture_scope():
            assert runner.capture_on_miss
            raise RuntimeError("warmup")
    assert not runner.capture_on_miss
    assert runner.capture(torch.ones(2)) is False
    torch.testing.assert_close(runner(torch.ones(2)), torch.full((2,), 2.0))


def test_capture_oom_propagates_without_eager_retry(monkeypatch) -> None:
    import pytest

    from worldfoundry.core.execution.graphs.inference_graph import _signature_key

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("allocator out of memory")

    runner = InferenceCUDAGraphRunner(lambda x: x + 1)
    monkeypatch.setattr(torch.cuda, "Stream", fail)
    with pytest.raises(torch.OutOfMemoryError):
        runner._capture(_signature_key((torch.ones(1),), {}, None), (torch.ones(1),), {})
    assert runner.report()["eager"] == runner.report()["capture_failed"] == 0


def test_scalar_leaf_key_prevents_bool_int_float_and_signed_zero_collisions() -> None:
    from worldfoundry.core.execution.graphs.inference_graph import _scalar_leaf_key

    # Python considers 1 == 1.0 == True and 0 == 0.0 == -0.0 == False, while
    # float("nan") != float("nan"). Keys must keep distinct types/signs apart
    # and hash/compare equal across repeated NaNs.
    distinct_values = [1, 1.0, True, 0, 0.0, -0.0, False, "0.0", "1", None, torch.float16, torch.bfloat16]
    keys = [_scalar_leaf_key(v) for v in distinct_values]
    assert len(set(keys)) == len(distinct_values)

    nan_key_a = _scalar_leaf_key(float("nan"))
    nan_key_b = _scalar_leaf_key(float("nan"))
    assert nan_key_a == nan_key_b
    assert hash(nan_key_a) == hash(nan_key_b)


def test_materialize_fast_paths_preserve_tensor_tuple_and_namedtuple_types() -> None:
    Pair = namedtuple("Pair", ("first", "second"))
    runner = InferenceCUDAGraphRunner(lambda x: x)

    # Single tensor fast path
    t = torch.tensor([1.0, 2.0])
    t_out = runner._materialize(t)
    t.add_(5.0)
    assert type(t_out) is torch.Tensor
    assert t_out.data_ptr() != t.data_ptr()
    torch.testing.assert_close(t_out, torch.tensor([1.0, 2.0]), rtol=0, atol=0)

    # Plain tuple of tensors fast path
    a = torch.tensor([3.0])
    b = torch.tensor([4.0])
    tup_out = runner._materialize((a, b))
    a.zero_()
    b.zero_()
    assert type(tup_out) is tuple
    assert tup_out[0].data_ptr() != a.data_ptr()
    assert tup_out[1].data_ptr() != b.data_ptr()
    torch.testing.assert_close(tup_out[0], torch.tensor([3.0]), rtol=0, atol=0)
    torch.testing.assert_close(tup_out[1], torch.tensor([4.0]), rtol=0, atol=0)

    # Namedtuple of tensors must NOT be demoted to plain tuple
    c = torch.tensor([7.0])
    d = torch.tensor([8.0])
    nt_out = runner._materialize(Pair(c, d))
    c.zero_()
    d.zero_()
    assert isinstance(nt_out, Pair)
    assert nt_out.first.data_ptr() != c.data_ptr()
    assert nt_out.second.data_ptr() != d.data_ptr()
    torch.testing.assert_close(nt_out.first, torch.tensor([7.0]), rtol=0, atol=0)
    torch.testing.assert_close(nt_out.second, torch.tensor([8.0]), rtol=0, atol=0)


def test_pool_property_and_external_pool_preservation_on_invalidate() -> None:
    sentinel_pool = (0, 999)
    shared = InferenceCUDAGraphRunner(lambda x: x, pool=sentinel_pool)
    owned = InferenceCUDAGraphRunner(lambda x: x)

    assert shared.pool == sentinel_pool
    assert owned.pool is None

    # Invalidating a runner sharing an external pool must not clear the handle.
    shared.invalidate()
    assert shared.pool == sentinel_pool

    # Invalidating a runner that owns its pool releases it.
    owned._pool = (0, 1000)
    owned.invalidate()
    assert owned.pool is None
