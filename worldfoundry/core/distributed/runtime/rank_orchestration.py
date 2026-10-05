# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank orchestration: payload/signal buses and distributed op specs.

Coordinates ranks that are *not* in the same TP/CP group (for example a
preprocessor rank vs a DiT rank). :class:`PayloadBus` / :class:`SignalBus`
move CPU objects and wake-up flags without going through NCCL tensor
collectives. Use model-parallel groups for the actual forward.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Generic, TypeVar, cast

import torch
import torch.distributed as dist

_PayloadT = TypeVar("_PayloadT")
_ResultT = TypeVar("_ResultT")
_SignalT = TypeVar("_SignalT", bound=IntEnum)

_DISTRIBUTED_OP_ATTR = "__distributed_op_spec__"

# ──────────────────────────────────────────────────────────────────────────
# Signal / payload buses — CPU objects and int flags, not NCCL activations
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DistributedOpSpec(Generic[_SignalT]):
    """Metadata attached to a :func:`distributed_op` method (signal + name)."""

    signal: _SignalT
    method_name: str


@dataclass(frozen=True, slots=True)
class _InvocationPayload:
    """Picklable args/kwargs that rank 0 broadcasts to worker ranks."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class SignalBus(Generic[_SignalT]):
    """Broadcast control signals from master to workers in strict order."""

    def __init__(
        self,
        *,
        device: torch.device,
        signal_type: type[_SignalT],
        master_rank: int = 0,
    ) -> None:
        """Bind the enum type and a monotonic counter that detects missed signals."""

        self.device = device
        self.signal_type = signal_type
        self.master_rank = master_rank
        self._counter = 0

    def send(self, signal: _SignalT) -> None:
        """Broadcast ``(counter, signal)`` from ``master_rank``; increment after send.

        The counter rides the same tensor so a dropped packet cannot look like
        the next expected signal. Single-process mode still increments so a
        later ``recv`` would fail closed if someone mixed the two paths.
        """

        encoded_signal = torch.tensor([self._counter, int(signal)], dtype=torch.int64, device=self.device)
        if dist.is_initialized():
            dist.broadcast(encoded_signal, src=self.master_rank)
        self._counter += 1

    def recv(self) -> _SignalT:
        """Receive the next master signal; refuse a counter mismatch.

        Requires an initialized process group — a silent local return would
        let workers proceed without the master's payload.
        """

        if not dist.is_initialized():
            raise RuntimeError("Cannot receive distributed signals without process group")

        packet = torch.tensor([self._counter, 0], dtype=torch.int64, device=self.device)
        dist.broadcast(packet, src=self.master_rank)
        received_counter = int(packet[0].item())
        if received_counter != self._counter:
            raise RuntimeError(f"Signal counter mismatch: got {received_counter}, expected {self._counter}")
        self._counter += 1
        return self.signal_type(int(packet[1].item()))


class PayloadBus:
    """Broadcast picklable payloads from master to workers."""

    def __init__(self, *, master_rank: int = 0) -> None:
        """Record which rank owns the object-list source."""

        self.master_rank = master_rank

    def broadcast_object(self, payload: _PayloadT) -> _PayloadT:
        """Gloo/object broadcast; identity when no process group is up."""

        if not dist.is_initialized():
            return payload

        payload_list = [payload]
        dist.broadcast_object_list(payload_list, src=self.master_rank)
        return payload_list[0]


@dataclass(slots=True)
class _RegisteredHandler:
    """Worker-side callback bound to one :class:`DistributedOpSpec` signal."""

    method_name: str
    callback: Callable[[], Any]


# ──────────────────────────────────────────────────────────────────────────
# RankCoordinator — rank 0 drives invoke(); workers block in worker_loop()
# ──────────────────────────────────────────────────────────────────────────


class RankCoordinator(Generic[_SignalT]):
    """Coordinates rank0-driven distributed operation invocation."""

    def __init__(
        self,
        *,
        device: torch.device,
        signal_type: type[_SignalT],
        is_master: bool,
        master_rank: int = 0,
        signal_bus: SignalBus[_SignalT] | None = None,
        payload_bus: PayloadBus | None = None,
    ) -> None:
        """Wire buses and an empty handler table; create default buses if omitted."""

        self.signal_bus = signal_bus or SignalBus(
            device=device,
            signal_type=signal_type,
            master_rank=master_rank,
        )
        self.payload_bus = payload_bus or PayloadBus(master_rank=master_rank)
        self.is_master = is_master
        self._lock = threading.RLock()
        self._handlers: dict[_SignalT, _RegisteredHandler] = {}

    def invoke(
        self,
        *,
        signal: _SignalT,
        payload: _PayloadT | None,
        handler: Callable[[_PayloadT], _ResultT],
    ) -> _ResultT:
        """Send signal (master), broadcast payload, then run ``handler`` on every rank.

        Non-master ranks must pass ``payload=None``; a local payload would
        diverge from the broadcast and corrupt the worker call.
        """

        with self._lock:
            if self.is_master:
                self.signal_bus.send(signal)
            elif payload is not None:
                raise AssertionError(f"Non-master rank cannot provide payload for signal {signal.name}")

            synced_payload = self.payload_bus.broadcast_object(payload)
            if synced_payload is None:
                raise AssertionError(f"Synchronized payload for {signal.name} is None")
            return handler(cast(_PayloadT, synced_payload))

    def register(self, *, signal: _SignalT, method_name: str, callback: Callable[[], Any]) -> None:
        """Bind ``callback`` to ``signal``; refuse a second method on the same signal."""

        existing = self._handlers.get(signal)
        if existing is not None and existing.method_name != method_name:
            raise ValueError(f"Signal {signal.name} already registered to {existing.method_name}")
        self._handlers[signal] = _RegisteredHandler(method_name=method_name, callback=callback)

    def register_distributed_ops(self, obj: Any) -> None:
        """Scan ``type(obj)`` for :func:`distributed_op` methods and register them."""

        for method_name in dir(type(obj)):
            method_obj = getattr(type(obj), method_name, None)
            spec = getattr(method_obj, _DISTRIBUTED_OP_ATTR, None)
            if not isinstance(spec, DistributedOpSpec):
                continue
            bound_method = getattr(obj, method_name)
            self.register(
                signal=spec.signal,
                method_name=method_name,
                callback=lambda bound_method=bound_method: bound_method(),
            )

    def worker_loop(self, *, exit_signal: _SignalT) -> None:
        """Block receiving signals until ``exit_signal``; unknown signals raise."""

        while True:
            signal = self.signal_bus.recv()
            if signal == exit_signal:
                break
            handler = self._handlers.get(signal)
            if handler is None:
                raise ValueError(f"No distributed handler registered for signal {signal.name}")
            handler.callback()

    def send_exit(self, *, exit_signal: _SignalT) -> None:
        """Master-only: wake workers so :meth:`worker_loop` can return."""

        if not self.is_master:
            raise RuntimeError("Only master rank can send exit signal")
        self.signal_bus.send(exit_signal)


# ──────────────────────────────────────────────────────────────────────────
# @distributed_op — methods become rank-0 coordinated RPCs
# ──────────────────────────────────────────────────────────────────────────


def distributed_op(
    signal: _SignalT,
    *,
    coordinator_attr: str = "rank_coordinator",
) -> Callable[[Callable[..., _ResultT]], Callable[..., _ResultT]]:
    """Decorate a method as a rank0-coordinated distributed operation."""

    def decorator(method: Callable[..., _ResultT]) -> Callable[..., _ResultT]:
        """Attach :class:`DistributedOpSpec` and route calls through the coordinator."""

        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> _ResultT:
            """Master packs args; workers must call with an empty signature."""

            coordinator = getattr(self, coordinator_attr)
            if not isinstance(coordinator, RankCoordinator):
                raise TypeError(f"Expected `{coordinator_attr}` to be RankCoordinator, got {type(coordinator)}")
            payload: _InvocationPayload | None
            if coordinator.is_master:
                payload = _InvocationPayload(args=tuple(args), kwargs=dict(kwargs))
            else:
                if args or kwargs:
                    raise AssertionError(f"Non-master rank cannot provide call arguments for signal {signal.name}")
                payload = None

            def invoke_method(invocation_payload: _InvocationPayload) -> _ResultT:
                """Replay the broadcast args against the original method."""

                return method(self, *invocation_payload.args, **invocation_payload.kwargs)

            return coordinator.invoke(signal=signal, payload=payload, handler=invoke_method)

        setattr(
            wrapper,
            _DISTRIBUTED_OP_ATTR,
            DistributedOpSpec(signal=signal, method_name=getattr(method, "__name__", "distributed_op")),
        )
        return wrapper

    return decorator


__all__ = [
    "DistributedOpSpec",
    "PayloadBus",
    "RankCoordinator",
    "SignalBus",
    "distributed_op",
]
