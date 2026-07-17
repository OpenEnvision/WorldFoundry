"""Stage event recorder used by profiled generation subprocesses.

The recorder is intentionally no-op unless ``WRBENCH_STAGE_EVENTS_PATH`` is set.
This keeps normal generation behavior unchanged and lets parent monitors opt in
by passing one environment variable to child processes.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


STAGE_EVENT_SCHEMA_VERSION = "wrbench_stage_event_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StageRecorder:
    """Append JSONL stage-span events for one profiled generation command."""

    def __init__(self, path: str | Path | None, *, command_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.command_id = command_id
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "StageRecorder":
        return cls(
            os.environ.get("WRBENCH_STAGE_EVENTS_PATH"),
            command_id=os.environ.get("WRBENCH_RESOURCE_COMMAND_ID"),
        )

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @contextlib.contextmanager
    def stage(
        self,
        stage: str,
        *,
        item_id: str | None = None,
        attribution_scope: str = "item",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started_at = utc_now_iso()
        start_mono = time.monotonic()
        status = "ok"
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_mono = time.monotonic()
            event: dict[str, Any] = {
                "schema_version": STAGE_EVENT_SCHEMA_VERSION,
                "command_id": self.command_id,
                "stage": stage,
                "item_id": item_id,
                "attribution_scope": attribution_scope,
                "status": status,
                "started_at": started_at,
                "ended_at": utc_now_iso(),
                "start_monotonic": start_mono,
                "end_monotonic": end_mono,
                "duration_seconds": max(0.0, end_mono - start_mono),
                "pid": os.getpid(),
                "rank": _env_int("RANK"),
                "local_rank": _env_int("LOCAL_RANK"),
                "world_size": _env_int("WORLD_SIZE"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "metadata": metadata or {},
            }
            if error:
                event["error"] = error
            self._write_event(event)

    def _write_event(self, event: dict[str, Any]) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")


def get_stage_recorder() -> StageRecorder:
    return StageRecorder.from_env()


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None
