"""Bounded real Gloo control/data-plane tests for distributed streaming."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from worldfoundry.core.distributed.runtime.streaming_step import streaming_step_control

pytestmark = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")


@dataclass
class _ParallelOwner:
    """Minimal caller owning the actual Gloo groups created by these tests."""

    tp_size: int = 1
    tp_rank: int = 0
    cp_rank: int = 0
    tp_group: object = None
    cp_group: object = None
    control_group: object = None
    world_size: int = 1
    _closed: bool = False

    @property
    def is_main(self) -> bool:
        return self.cp_rank == self.tp_rank == 0

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("parallel owner is closed")
        if self.world_size > 1 and not dist.is_initialized():
            raise RuntimeError("parallel owner requires live groups")


def test_single_rank_reset_and_closed_controller_require_no_collectives() -> None:
    parallel = _ParallelOwner()
    control = streaming_step_control(parallel)
    with (
        patch.object(dist, "all_reduce", side_effect=AssertionError("unexpected collective")),
        patch.object(
            dist,
            "broadcast_object_list",
            side_effect=AssertionError("unexpected collective"),
        ),
    ):
        first = control.run_step(lambda admission: admission, inputs={"key": "w"})
        second = control.run_step(lambda admission: admission, reset=True, inputs={"key": "s"})
        assert (first.generation, first.step_index, first.reset) == (0, 0, False)
        assert (second.generation, second.step_index, second.reset) == (1, 0, True)
        assert second.inputs == {"key": "s"}
        control.close()
        control.close()
        assert not parallel._closed
        with pytest.raises(RuntimeError, match="closed"):
            control.run_step(lambda admission: admission)


def test_single_rank_preparation_failure_wins_over_stop() -> None:
    control = streaming_step_control(_ParallelOwner())
    executed = []
    with pytest.raises(RuntimeError, match="failed before"):
        control.run_step(executed.append, stopping=True, failed=True)
    assert executed == []
    with pytest.raises(RuntimeError, match="closed"):
        control.admit()


def _stream_worker(rank: int, scenario: str, rendezvous: str, output: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    groups = []
    record = {"steps": [], "error": "", "stop_checks": 0, "cleanup_collectives": 0}
    control = None
    try:
        group_timeout = timedelta(seconds=2)
        control_group = dist.new_group(backend="gloo", timeout=group_timeout)
        groups.append(control_group)
        data_group = dist.new_group(backend="gloo", timeout=group_timeout)
        groups.append(data_group)
        parallel = _ParallelOwner(world_size=2, cp_rank=rank, cp_group=data_group, control_group=control_group)
        control = streaming_step_control(parallel)

        def step(admission):
            if scenario == "step_failure" and rank == 1:
                raise RuntimeError("injected model step failure")
            with torch.inference_mode(False):
                value = torch.tensor(rank + 1)
                dist.all_reduce(value, group=data_group)
            assert value.item() == 3
            record["steps"].append(
                [admission.generation, admission.step_index, admission.reset, admission.inputs],
            )
            return value.item()

        if scenario == "reset":
            for index in range(4):
                with torch.inference_mode():
                    result = control.run_step(
                        step,
                        reset=(index == 2 if rank == 0 else True),
                        inputs={"command": index} if rank == 0 else {"ignored": rank},
                    )
                assert result == 3
            assert control.run_step(step, stopping=rank == 1) is None
        elif scenario == "stop_after_admission":
            stop = threading.Event()
            original_admit = control.admit

            def admit_then_stop(**kwargs):
                admission = original_admit(**kwargs)
                if rank == 0:
                    stop.set()
                return admission

            def stopping():
                record["stop_checks"] += 1
                return stop.is_set()

            control.admit = admit_then_stop
            assert control.run_step(step, stopping=stopping) == 3
            assert control.run_step(step, stopping=stopping) is None
            assert record["stop_checks"] == 2
        elif scenario == "preparation_failure":
            control.run_step(step, stopping=rank == 0, failed=rank == 1)
        elif scenario == "step_failure":
            control.run_step(step)
        elif scenario == "missing_peer":
            if rank == 0:
                control.run_step(step)
        else:
            raise AssertionError(f"unknown scenario: {scenario}")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if control is not None:

            def forbidden_collective(*args, **kwargs):
                record["cleanup_collectives"] += 1
                raise AssertionError("cleanup attempted a collective")

            with (
                patch.object(dist, "all_reduce", forbidden_collective),
                patch.object(
                    dist,
                    "broadcast_object_list",
                    forbidden_collective,
                ),
                patch.object(dist, "barrier", forbidden_collective),
            ):
                control.close()
                control.close()
        Path(output, f"rank{rank}.json").write_text(json.dumps(record), encoding="utf-8")
        for group in reversed(groups):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["reset", "stop_after_admission", "preparation_failure", "step_failure", "missing_peer"]
)
def test_real_gloo_ranks_admit_reset_stop_and_fail_with_bounded_cleanup(tmp_path, scenario: str) -> None:
    processes = [
        mp.get_context("spawn").Process(
            target=_stream_worker,
            args=(rank, scenario, str(tmp_path / "rendezvous"), str(tmp_path)),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert [process.exitcode for process in processes] == [0, 0], (
            "streaming ranks did not exit within their deadlines"
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
    leader, worker = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert leader["cleanup_collectives"] == worker["cleanup_collectives"] == 0
    if scenario in {"reset", "stop_after_admission"}:
        assert leader["error"] == worker["error"] == ""
        assert leader["steps"] == worker["steps"]
        if scenario == "reset":
            assert leader["steps"] == [
                [0, 0, False, {"command": 0}],
                [0, 1, False, {"command": 1}],
                [1, 0, True, {"command": 2}],
                [1, 1, False, {"command": 3}],
            ]
        else:
            assert leader["steps"] == [[0, 0, False, None]]
            assert leader["stop_checks"] == worker["stop_checks"] == 2
    elif scenario == "preparation_failure":
        assert "failed before" in leader["error"] and "failed before" in worker["error"]
        assert leader["steps"] == worker["steps"] == []
    elif scenario == "step_failure":
        assert leader["error"]
        assert "injected model step failure" in worker["error"]
        assert leader["steps"] == worker["steps"] == []
    else:
        assert leader["error"]
        assert worker["error"] == ""
        assert leader["steps"] == worker["steps"] == []
