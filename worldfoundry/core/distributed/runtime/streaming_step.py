"""Streaming step admission over an inference mesh's separate Gloo control group.

Every participating rank calls ``run_step`` at the same boundary. Cancellation
or preparation failure is agreed before model collectives; an admitted callback
runs even if cancellation arrives afterwards. Rank zero owns input and reset
events. The caller supervises failures inside a callback using the data group's
timeout and process launcher: cleanup never tries another collective.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch
import torch.distributed as dist

_Result = TypeVar("_Result")


class StreamingParallelOwner(Protocol):
    """Structural contract for the caller-owned inference/control mesh.

    No concrete context implementation is imported or constructed. The owner
    creates and supervises real groups, and ``_check`` verifies they are live.
    """

    tp_size: int
    tp_rank: int
    cp_rank: int
    tp_group: Any
    cp_group: Any
    control_group: Any

    @property
    def world_size(self) -> int: ...

    @property
    def is_main(self) -> bool: ...

    def _check(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StepAdmission:
    """One committed step with leader-owned inputs and reset generation."""

    generation: int
    step_index: int
    reset: bool
    inputs: Any = None


class StreamingStepControl:
    """Borrow an existing control group; create no groups or model threads.

    Construct on every rank before local preparation. The group's timeout is
    configured by its owner and bounds a peer that never reaches admission.
    ``close`` is local and idempotent; it does not release the borrowed mesh.
    Calls on an instance belong to one model thread and must remain ordered.
    """

    def __init__(self, parallel: StreamingParallelOwner) -> None:
        parallel._check()
        if parallel.world_size > 1:
            if dist.get_backend(parallel.control_group) != "gloo":
                raise ValueError("streaming admission requires a separate Gloo control group")
            if dist.get_world_size(parallel.control_group) != parallel.world_size:
                raise ValueError("streaming control group must contain the entire inference mesh")
            if dist.get_rank(parallel.control_group) != parallel.cp_rank * parallel.tp_size + parallel.tp_rank:
                raise ValueError("streaming control rank does not match the inference mesh")
            if parallel.control_group is parallel.tp_group or parallel.control_group is parallel.cp_group:
                raise ValueError("streaming control group must be separate from model data groups")
        self._parallel = parallel
        self._source_rank = dist.get_process_group_ranks(parallel.control_group)[0] if parallel.world_size > 1 else 0
        self._generation = 0
        self._step_index = 0
        self._closed = False

    @torch.inference_mode(False)
    @torch.no_grad()
    def admit(
        self,
        *,
        stopping: bool = False,
        failed: bool = False,
        reset: bool = False,
        inputs: Any = None,
    ) -> StepAdmission | None:
        """Commit all ranks or stop all ranks; broadcast only trusted CPU metadata.

        Any rank's preparation failure takes precedence over cancellation. Reset
        and inputs are authoritative on rank zero, and reset restarts the step
        index while incrementing generation. After admission, callers execute
        the step without inspecting local stop events again.
        """

        if self._closed:
            raise RuntimeError("streaming step control is closed")
        self._parallel._check()
        status = torch.tensor(2 if failed else int(stopping), dtype=torch.int64)
        try:
            if self._parallel.world_size > 1:
                dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self._parallel.control_group)
            if status.item() == 2:
                raise RuntimeError("an inference rank failed before streaming step admission")
            if status.item() == 1:
                self.close()
                return None
            admission = None
            if self._parallel.is_main:
                admission = StepAdmission(
                    generation=self._generation + int(reset),
                    step_index=0 if reset else self._step_index,
                    reset=bool(reset),
                    inputs=inputs,
                )
            payload = [admission]
            if self._parallel.world_size > 1:
                dist.broadcast_object_list(payload, group=self._parallel.control_group, src=self._source_rank)
            admission = payload[0]
            if not isinstance(admission, StepAdmission):
                raise RuntimeError("streaming leader did not provide step admission metadata")
            self._generation = admission.generation
            self._step_index = admission.step_index + 1
            return admission
        except BaseException:
            self.close()
            raise

    def run_step(
        self,
        step: Callable[[StepAdmission], _Result],
        *,
        stopping: bool | Callable[[], bool] = False,
        failed: bool = False,
        reset: bool = False,
        inputs: Any = None,
    ) -> _Result | None:
        """Sample cancellation once, then execute a committed model callback.

        A stop predicate such as ``threading.Event.is_set`` is never re-read
        after admission. Callback failure closes only this local controller;
        the data-plane timeout and caller-owned launcher handle other ranks.
        """

        try:
            admission = self.admit(
                stopping=stopping() if callable(stopping) else stopping,
                failed=failed,
                reset=reset,
                inputs=inputs,
            )
            return None if admission is None else step(admission)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Stop local use without a barrier, reduction, or borrowed-group teardown."""

        self._closed = True


def streaming_step_control(parallel: StreamingParallelOwner) -> StreamingStepControl:
    """Opt in using an existing mesh with a bounded, separate Gloo control group."""

    return StreamingStepControl(parallel)


__all__ = ["StepAdmission", "StreamingParallelOwner", "StreamingStepControl", "streaming_step_control"]
