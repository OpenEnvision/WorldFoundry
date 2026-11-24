"""WebSocket model server for embodied policy adapters."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from worldfoundry.evals.tasks.embodied.policy_adapter import (
    EmbodiedPolicyAdapter,
    build_policy_adapter,
    normalize_action_payload,
)

from .protocol import Message, MessageType, hello_payload, pack_message, unpack_message

logger = logging.getLogger(__name__)


def _serializable_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in spec.items():
        to_dict = getattr(value, "to_dict", None)
        payload[str(key)] = to_dict() if callable(to_dict) else value
    return payload


async def _maybe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _handle_connection(ws: Any, adapter: EmbodiedPolicyAdapter) -> None:
    async for raw in ws:
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        try:
            message = unpack_message(raw)
        except Exception as exc:
            await ws.send(pack_message(Message(MessageType.ERROR, {"error": str(exc)})))
            continue

        try:
            if message.type == MessageType.HELLO:
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.HELLO,
                            hello_payload(
                                role="server",
                                action_spec=_serializable_spec(adapter.get_action_spec()),
                                observation_spec=_serializable_spec(adapter.get_observation_spec()),
                            ),
                            seq=message.seq,
                        )
                    )
                )
            elif message.type == MessageType.EPISODE_START:
                hook = getattr(adapter, "start_episode", None)
                if callable(hook):
                    await _maybe_call(hook, message.payload)
            elif message.type == MessageType.EPISODE_END:
                hook = getattr(adapter, "end_episode", None)
                if callable(hook):
                    await _maybe_call(hook, message.payload)
            elif message.type == MessageType.OBSERVATION:
                obs = (
                    cast(Mapping[str, Any], message.payload.get("obs"))
                    if isinstance(message.payload.get("obs"), Mapping)
                    else message.payload
                )
                instruction = str(
                    message.payload.get("instruction")
                    or message.payload.get("task_description")
                    or message.payload.get("language_instruction")
                    or ""
                )
                action = await asyncio.to_thread(adapter.predict, obs, instruction)
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.ACTION,
                            normalize_action_payload(action),
                            seq=message.seq,
                        )
                    )
                )
            else:
                await ws.send(
                    pack_message(
                        Message(
                            MessageType.ERROR,
                            {"error": f"unsupported message type: {message.type.value}"},
                            seq=message.seq,
                        )
                    )
                )
        except Exception as exc:
            logger.exception("embodied policy server request failed")
            await ws.send(
                pack_message(
                    Message(
                        MessageType.ERROR,
                        {"error": f"{type(exc).__name__}: {exc}"},
                        seq=message.seq,
                    )
                )
            )


async def serve_async(adapter: EmbodiedPolicyAdapter, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve a policy adapter over WebSocket until cancelled."""
    import websockets

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, adapter)

    async with websockets.serve(handler, host, int(port), compression=None, max_size=None, ping_interval=None):
        logger.info("Serving embodied policy adapter on ws://%s:%s", host, port)
        await asyncio.Future()


def serve(adapter: EmbodiedPolicyAdapter, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Blocking server entry point."""
    asyncio.run(serve_async(adapter, host=host, port=port))


def load_model_server_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"model server config must be a mapping: {path}")
    return payload


def serve_from_config(path: str | Path, *, host: str | None = None, port: int | None = None) -> None:
    """Build and serve a policy adapter from a YAML model-server config."""
    config = load_model_server_config(path)
    args = dict(config.get("args") or config.get("model_parameters") or {})
    model_id = str(config.get("model_id") or args.pop("model_id", None) or "openvla")
    if host is None:
        host = str(args.pop("host", config.get("host", "0.0.0.0")))
    if port is None:
        port = int(args.pop("port", config.get("port", 8000)))
    adapter = build_policy_adapter(model_id, args)
    try:
        serve(adapter, host=host, port=port)
    finally:
        adapter.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a WorldFoundry embodied policy adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    serve_from_config(args.config, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_model_server_config", "serve", "serve_async", "serve_from_config"]
