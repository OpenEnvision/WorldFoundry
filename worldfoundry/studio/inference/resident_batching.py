# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatible batching on the existing resident process and request lock."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

from worldfoundry.core.execution.queued_batch import QueuedBatchExecutor

from .batching import native_batch_key

if TYPE_CHECKING:
    from .dispatch import _ResidentWorker
    from .execution import RunRecord


@dataclass
class ResidentRequest:
    payload: Mapping[str, Any]
    result_path: Path
    error_path: Path
    deadline: float | None
    log_callback: Callable[[str, str], None] | None
    cancel_requested: Callable[[], bool] | None
    queued_at: float = field(default_factory=time.monotonic)
    abandoned: Event = field(default_factory=Event)

    def interrupted(self) -> bool:
        return (
            self.abandoned.is_set()
            or bool(self.cancel_requested is not None and self.cancel_requested())
            or (self.deadline is not None and time.monotonic() >= self.deadline)
        )


def resident_batch_options() -> tuple[int, float]:
    """Read startup batching limits. Size one preserves ordinary resident dispatch."""
    import math

    size = int(os.getenv("WORLDFOUNDRY_STUDIO_BATCH_MAX_SIZE", "1"))
    delay = float(os.getenv("WORLDFOUNDRY_STUDIO_BATCH_DELAY_MS", "5")) / 1000.0
    if size < 1 or not math.isfinite(delay) or delay < 0:
        raise ValueError("Studio batch size must be positive and batch delay finite and non-negative")
    return size, delay


def run_queued_resident_request(
    worker: _ResidentWorker,
    request: ResidentRequest,
    *,
    max_batch_size: int,
    max_delay_seconds: float,
) -> RunRecord:
    """Wait independently while the serial owner services a compatible batch.

    Cancellation/deadlines remove queued work. Once selected, an abandoned member
    does not interrupt live or queued peers; an abandoned owner with no peers stops.
    The worker lease survives a waiting caller until its execution future completes.
    """
    from . import dispatch

    future: Future[RunRecord] | None = None
    try:
        with worker.shutdown_lock:
            if worker.shutdown_complete:
                raise RuntimeError("resident worker closed before request execution")
            if worker.batch_executor is None:
                worker.batch_executor = QueuedBatchExecutor(
                    lambda items: _execute_resident_batch(worker, items),
                    max_batch_size=max_batch_size,
                    max_delay_seconds=max_delay_seconds,
                )
            future = worker.batch_executor.submit(
                request,
                compatibility_key=native_batch_key(request.payload["run_kwargs"]),
            )
        while True:
            dispatch._check_dispatch_interrupt(request.cancel_requested, request.deadline)
            try:
                return future.result(timeout=0.05)
            except FutureTimeoutError:
                if future.done():
                    return future.result()
            except CancelledError as error:
                raise RuntimeError("resident worker closed before request execution") from error
    except BaseException:
        request.abandoned.set()
        if future is not None:
            future.cancel()
        raise
    finally:
        if future is None or future.done():
            dispatch._release_resident_worker(worker)
        else:
            future.add_done_callback(lambda _: dispatch._release_resident_worker(worker))


def _execute_resident_batch(worker: _ResidentWorker, requests: Sequence[ResidentRequest]) -> tuple[RunRecord, ...]:
    from . import dispatch
    from .execution import _persist_studio_performance_metadata

    def all_interrupted() -> bool:
        if not all(request.interrupted() for request in requests):
            return False
        # Preserve queued peers that still depend on this resident owner.
        with dispatch._RESIDENT_WORKERS_LOCK:
            return worker.in_use <= len(requests)

    def log(stream: str, text: str) -> None:
        for request in requests:
            dispatch._append_log(request.log_callback, stream, text)

    acquired = False
    try:
        dispatch._acquire_resident_worker_request_lock(
            worker,
            deadline=None,
            request_timeout=0,
            cancel_requested=all_interrupted,
        )
        acquired = True
        started = time.monotonic()
        queue_ms = [(started - request.queued_at) * 1000.0 for request in requests]
        log("system", f"resident dispatch: requests={len(requests)} model={worker.model_id}\n")
        if len(requests) == 1:
            request = requests[0]
            dispatch._send_resident_worker_request(
                worker, request_payload=request.payload, cancel_requested=all_interrupted
            )
            records = (
                dispatch._wait_for_resident_worker_result(
                    worker,
                    result_path=request.result_path,
                    error_path=request.error_path,
                    log_callback=log,
                    cancel_requested=all_interrupted,
                    request_timeout=0,
                ),
            )
        else:
            first = requests[0]
            result_path = first.result_path.with_name("batch-result.json")
            error_path = first.error_path.with_name("batch-error.json")
            for path in (result_path, error_path):
                path.unlink(missing_ok=True)
            dispatch._send_resident_worker_request(
                worker,
                cancel_requested=all_interrupted,
                request_payload={
                    "request_id": first.payload["request_id"],
                    "requests": [dict(request.payload) for request in requests],
                    "result_path": str(result_path),
                    "error_path": str(error_path),
                },
            )
            payload = dispatch._wait_for_resident_worker_payload(
                worker,
                result_path=result_path,
                error_path=error_path,
                log_callback=log,
                cancel_requested=all_interrupted,
                request_timeout=0,
            )
            raw_records = payload.get("records")
            if not isinstance(raw_records, list) or len(raw_records) != len(requests):
                raise RuntimeError("resident batch must return one record per request")
            records = tuple(dispatch.run_record_from_manifest(record) for record in raw_records)
        for request, record, wait_ms in zip(requests, records, queue_ms):
            if request.interrupted():
                record.status = "cancelled"
                record.metadata["resident_result_discarded"] = True
            _persist_studio_performance_metadata(record, {"resident_queue_wait_ms": wait_ms})
        return records
    except BaseException as error:
        if acquired:
            dispatch._drop_resident_worker(worker, force=True)
        # Queued ownership has begun. Never convert a failed batch into a
        # one-shot singleton retry in the outer dispatch facade.
        if isinstance(error, dispatch._ResidentWorkerUnavailable):
            raise RuntimeError(str(error)) from error
        raise
    finally:
        if acquired:
            worker.lock.release()
