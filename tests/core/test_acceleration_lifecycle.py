from __future__ import annotations

import asyncio
import contextlib
import threading
from types import SimpleNamespace

import pytest

from worldfoundry.core.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from worldfoundry.core.acceleration.encoder_lifecycle import (
    move_tensors_to_cpu,
    release_one_shot_encoder_references,
    run_one_shot_encoder_stage,
)
from worldfoundry.core.acceleration.overlap import HostThreadOverlap
from worldfoundry.core.acceleration.prewarm import (
    PrewarmTimeoutError,
    run_async_prewarm_sequence,
    run_prewarm_sequence,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_prewarm_sequence_records_cold_and_indexed_steady_steps() -> None:
    clock = _Clock()
    calls: list[object] = []

    def cold_start() -> None:
        calls.append("cold")
        clock.advance(0.2)

    def steady(index: int) -> None:
        calls.append(index)
        clock.advance(0.1)

    timing = run_prewarm_sequence(
        cold_start=cold_start,
        steady_state=steady,
        steady_steps=2,
        timeout_s=1.0,
        time_fn=clock,
    )

    assert calls == ["cold", 0, 1]
    assert timing.cold_start is not None
    assert timing.cold_start.elapsed_ms == pytest.approx(200.0)
    assert [step.elapsed_ms for step in timing.steady_state] == pytest.approx([100.0, 100.0])
    assert timing.elapsed_ms == pytest.approx(400.0)


def test_prewarm_timeout_is_reported_after_non_interruptible_step() -> None:
    clock = _Clock()

    def slow_step(_index: int) -> None:
        clock.advance(1.25)

    with pytest.raises(PrewarmTimeoutError) as caught:
        run_prewarm_sequence(
            steady_state=slow_step,
            steady_steps=1,
            timeout_s=1.0,
            time_fn=clock,
        )

    assert caught.value.timeout_s == 1.0
    assert caught.value.elapsed_s == pytest.approx(1.25)


def test_async_prewarm_preserves_step_order() -> None:
    calls: list[object] = []

    async def cold_start() -> None:
        calls.append("cold")
        await asyncio.sleep(0)

    async def steady(index: int) -> None:
        calls.append(index)
        await asyncio.sleep(0)

    timing = asyncio.run(
        run_async_prewarm_sequence(
            cold_start=cold_start,
            steady_state=steady,
            steady_steps=3,
        )
    )

    assert calls == ["cold", 0, 1, 2]
    assert len(timing.steps) == 4


class _FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value
        self.cpu_calls = 0

    def cpu(self) -> str:
        self.cpu_calls += 1
        return f"cpu:{self.value}"


def test_encoder_lifecycle_clears_references_and_nested_tensors(monkeypatch) -> None:
    cuda_calls: list[object] = []

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize(device: object) -> None:
            cuda_calls.append(("sync", device))

        @staticmethod
        def empty_cache() -> None:
            cuda_calls.append("empty")

    fake_torch = SimpleNamespace(
        cuda=Cuda(),
        is_tensor=lambda value: isinstance(value, _FakeTensor),
    )
    owner = SimpleNamespace(text_encoder=object(), image_encoder=None)
    monkeypatch.setattr(
        "worldfoundry.core.acceleration.encoder_lifecycle.gc.collect",
        lambda: cuda_calls.append("gc"),
    )

    released = release_one_shot_encoder_references(
        owner,
        "text_encoder",
        "image_encoder",
        device="cuda:1",
        synchronize_cuda=True,
        torch_module=fake_torch,
    )
    nested = move_tensors_to_cpu(
        {"values": [_FakeTensor("a"), (_FakeTensor("b"),)]},
        torch_module=fake_torch,
    )

    assert released == ("text_encoder",)
    assert owner.text_encoder is None and owner.image_encoder is None
    assert cuda_calls == ["gc", ("sync", "cuda:1"), "empty"]
    assert nested == {"values": ["cpu:a", ("cpu:b",)]}


def test_one_shot_encoder_release_runs_when_stage_fails() -> None:
    released: list[bool] = []
    fake_torch = SimpleNamespace(no_grad=lambda: contextlib.nullcontext())

    def fail() -> None:
        raise RuntimeError("encode failed")

    with pytest.raises(RuntimeError, match="encode failed"):
        run_one_shot_encoder_stage(
            fail,
            release=lambda: released.append(True),
            torch_module=fake_torch,
        )
    assert released == [True]


def test_host_overlap_bounds_work_and_propagates_worker_errors() -> None:
    started = threading.Event()
    release = threading.Event()
    runner = HostThreadOverlap(name="test-overlap")

    def work() -> None:
        started.set()
        release.wait()

    runner.submit(work)
    assert started.wait(timeout=1.0)
    assert runner.pending is True
    assert runner.wait(timeout_s=0.001) is False
    with pytest.raises(RuntimeError, match="already has pending"):
        runner.submit(lambda: None)
    release.set()
    assert runner.wait(timeout_s=1.0, raise_error=True) is True

    runner.submit(lambda: (_ for _ in ()).throw(ValueError("worker failed")))
    with pytest.raises(ValueError, match="worker failed"):
        runner.wait(timeout_s=1.0, raise_error=True)
    runner.close()


def test_cuda_graph_dispatch_separates_branches_and_fill_from_replay() -> None:
    def eager(value: str) -> str:
        return f"eager:{value}"

    class Wrapper:
        def __init__(self, branch: int) -> None:
            self.branch = branch
            self.reset_calls = 0

        def drain(self, value: str) -> str:
            return f"drain-{self.branch}:{value}"

        def __call__(self, value: str) -> str:
            return f"graph-{self.branch}:{value}"

        def reset(self) -> None:
            self.reset_calls += 1

    wrappers: list[Wrapper] = []

    def factory(_fn, *, warmup_iters: int) -> Wrapper:
        assert warmup_iters == 3
        wrapper = Wrapper(len(wrappers))
        wrappers.append(wrapper)
        return wrapper

    dispatch = CUDAGraphDispatch(
        eager,
        enabled=True,
        capture_ar_index=2,
        warmup_iters=3,
        wrapper_factory=factory,
    )

    assert dispatch.select(0, unconditional=False)("x") == "drain-0:x"
    assert dispatch.select(2, unconditional=False)("x") == "graph-0:x"
    assert dispatch.select(2, unconditional=True)("x") == "graph-1:x"
    dispatch.reset()
    assert [wrapper.reset_calls for wrapper in wrappers] == [1, 1]
    assert cuda_graph_capture_ar_index(sink_size=4, window_size=12, chunk_size=4) == 4
    with pytest.raises(ValueError, match="divisible"):
        cuda_graph_capture_ar_index(sink_size=1, window_size=4, chunk_size=2)
