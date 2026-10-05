"""Shared polling primitive for the Wan hosted API pipelines."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar


ResponseT = TypeVar("ResponseT")


def poll_wan_task_status(
    *,
    get_task: Callable[[str], ResponseT],
    extract_status: Callable[[ResponseT], str],
    task_id: str,
    service_label: str,
    logger: logging.Logger,
    poll_interval: float,
    max_retries: int,
    sleeper: Callable[[float], None] = time.sleep,
) -> ResponseT:
    """Poll one Wan task until the service reports a terminal status."""

    last_status: str | None = None
    for attempt in range(max_retries):
        task_info = get_task(task_id)
        task_status = extract_status(task_info).upper()
        if task_status and task_status != last_status:
            logger.info("%s task %s: %s", service_label, task_id, task_status)
            last_status = task_status
        # Keep this aligned with DashScope's BaseAsyncApi.wait terminal set.
        # UNKNOWN is returned for tasks the service can no longer resolve and
        # must not be retried until the local polling budget expires.
        if task_status in {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}:
            return task_info
        if attempt + 1 < max_retries:
            sleeper(poll_interval)
    raise TimeoutError(f"{service_label} task {task_id} polling timed out.")
