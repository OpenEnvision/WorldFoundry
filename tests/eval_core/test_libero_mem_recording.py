# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-Mem success updates preserve disabled and active recorder contracts."""

from __future__ import annotations

from typing import Any

import pytest

from worldfoundry.evals.tasks.embodied.simulators.base import DummyRecorder
from worldfoundry.evals.tasks.embodied.simulators.libero_mem.benchmark import LIBEROMemBenchmark

pytestmark = pytest.mark.unit


class _Environment:
    def __init__(self, *, done: bool, progress_done: bool) -> None:
        self.env = self
        self.done = done
        self.progress_done = progress_done
        self.obs = {"camera": "observation"}
        self.info = {"source": "simulator"}
        self.actions: list[list[float]] = []
        self.success_checks: list[bool] = []

    def step(self, action: list[float]) -> tuple[dict[str, str], float, bool, dict[str, str]]:
        self.actions.append(action)
        return self.obs, 2.5, self.done, self.info

    def _check_success(self, *, inc: bool) -> bool:
        self.success_checks.append(inc)
        return self.progress_done


class _Recorder:
    def __init__(self) -> None:
        self._next_step = 0
        self.rows: list[dict[str, Any]] = []
        self.frames: list[Any] = []

    def record_step(self, *, step: int | None = None, **fields: Any) -> None:
        if step is None:
            step = self._next_step
            self._next_step += 1
            self.rows.append({})
        self.rows[step].update(fields)

    def record_video(self, frame: Any) -> None:
        self.frames.append(frame)


def test_dummy_recorder_keeps_zero_steps() -> None:
    recorder = DummyRecorder()

    recorder.record_step(reward=2.5, done=True, success=True)
    recorder.record_video(None)

    assert recorder._next_step == 0


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("done", [False, True])
@pytest.mark.parametrize("progress_done", [False, True])
def test_step_preserves_success_and_updates_existing_record(
    monkeypatch: pytest.MonkeyPatch, recording: bool, done: bool, progress_done: bool
) -> None:
    benchmark = LIBEROMemBenchmark()
    env = _Environment(done=done, progress_done=progress_done)
    monkeypatch.setattr(benchmark, "_env", env)
    recorder = _Recorder() if recording else None
    if recorder is not None:
        monkeypatch.setattr(benchmark, "_recorder", recorder)
    actions = [0.0] * 7

    result = benchmark.step({"actions": actions})

    assert result.obs is env.obs
    assert result.info is env.info
    assert result.reward == 2.5
    assert result.done is (done or progress_done)
    assert env.success_checks == ([] if done else [True])
    assert env.actions == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    assert actions == [0.0] * 7
    if recorder is None:
        assert benchmark._recorder._next_step == 0
    else:
        assert recorder._next_step == 1
        assert recorder.rows == [{"reward": 2.5, "done": done or progress_done, "success": done or progress_done}]
        assert recorder.frames == [None]
