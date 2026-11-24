"""WebSocket policy adapter for WorldFoundry embodied model servers."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from concurrent.futures import Future
from typing import Any

from worldfoundry.evals.tasks.embodied.model_server.connection import EmbodiedWebSocketConnection
from worldfoundry.evals.tasks.embodied.policy_adapter import normalize_action_payload
from worldfoundry.evals.tasks.embodied.simulators.specs import DimSpec


class WebSocketPolicyAdapter:
    """Synchronous policy adapter backed by an async WebSocket model server."""

    def __init__(
        self,
        url: str = "ws://localhost:8000",
        *,
        timeout: float = 30.0,
        benchmark: str | None = None,
    ) -> None:
        self.url = str(url)
        self.timeout = float(timeout)
        self.benchmark = benchmark
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="wf-embodied-ws-policy", daemon=True)
        self._thread.start()
        self._conn = EmbodiedWebSocketConnection(self.url, timeout=self.timeout, benchmark=self.benchmark)
        self._run(self._conn.connect())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro: Any) -> Any:
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout + 5.0)

    @property
    def server_info(self) -> Mapping[str, Any]:
        return self._conn.server_info

    def start_episode(self, config: Mapping[str, Any]) -> None:
        self._run(self._conn.start_episode(dict(config)))

    def end_episode(self, result: Mapping[str, Any]) -> None:
        self._run(self._conn.end_episode(dict(result)))

    def predict(self, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        payload = dict(obs)
        payload.setdefault("task_description", instruction)
        action = self._run(self._conn.act(payload))
        return normalize_action_payload(action)

    def get_action_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        value = self._conn.server_info.get("action_spec")
        return dict(value) if isinstance(value, Mapping) else {}

    def get_observation_spec(self) -> Mapping[str, DimSpec | Mapping[str, Any]]:
        value = self._conn.server_info.get("observation_spec")
        return dict(value) if isinstance(value, Mapping) else {}

    def cleanup(self) -> None:
        try:
            self._run(self._conn.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)


__all__ = ["WebSocketPolicyAdapter"]
