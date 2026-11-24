"""Async WebSocket client for embodied policy servers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from .protocol import Message, MessageType, hello_payload, pack_message, unpack_message

logger = logging.getLogger(__name__)


class EmbodiedWebSocketConnection:
    """Small WebSocket client for the WorldFoundry embodied episode protocol."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        benchmark: str | None = None,
    ) -> None:
        self.url = str(url)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.benchmark = benchmark
        self.server_info: dict[str, Any] = {}
        self._seq = 0
        self._ws: Any = None

    async def connect(self) -> None:
        await self._connect_with_backoff()
        await self._hello()

    async def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def reconnect(self) -> None:
        await self.close()
        await self.connect()

    async def send(self, message_type: MessageType, payload: dict[str, Any], *, seq: int | None = None) -> int:
        if self._ws is None:
            await self.connect()
        if seq is None:
            self._seq += 1
            seq = self._seq
        message = Message(type=message_type, payload=payload, seq=seq)
        await self._ws.send(pack_message(message))
        return seq

    async def recv(self, *, timeout: float | None = None) -> Message:
        if self._ws is None:
            raise RuntimeError("WebSocket connection is not open")
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout if timeout is None else timeout)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return unpack_message(raw)

    async def start_episode(self, config: dict[str, Any]) -> None:
        await self.send(MessageType.EPISODE_START, config)

    async def end_episode(self, result: dict[str, Any]) -> None:
        await self.send(MessageType.EPISODE_END, result)

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        seq = await self.send(MessageType.OBSERVATION, obs)
        response = await self.recv(timeout=self.timeout)
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"policy server error: {response.payload}")
        if response.type != MessageType.ACTION:
            raise RuntimeError(f"expected action response, got {response.type.value}")
        if response.seq != seq:
            logger.warning("policy server seq mismatch: sent %s got %s", seq, response.seq)
        return response.payload

    async def _hello(self) -> None:
        payload = hello_payload(role="client", **({"benchmark": self.benchmark} if self.benchmark else {}))
        await self.send(MessageType.HELLO, payload)
        reply = await self.recv(timeout=self.timeout)
        if reply.type != MessageType.HELLO:
            raise RuntimeError(f"expected HELLO reply, got {reply.type.value}")
        self.server_info = dict(reply.payload or {})

    async def _connect_with_backoff(self) -> None:
        import websockets

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        compression=None,
                        max_size=None,
                        ping_interval=None,
                    ),
                    timeout=self.timeout,
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(self.backoff_base**attempt)
        raise ConnectionError(f"policy server unreachable at {self.url}") from last_exc


__all__ = ["EmbodiedWebSocketConnection"]
