#!/usr/bin/env python3
"""Persistent Qwen3-VL HTTP service.

This keeps the local Qwen3-VL model loaded in one process and exposes an
OpenAI-compatible `/v1/chat/completions` endpoint for this repository's VLM
wrapper.

Example:
CUDA_VISIBLE_DEVICES=0 QWENVL_DEVICE=cuda:0 python model/qwenvl_server.py \
  --model weights/QwenVL --host 127.0.0.1 --port 8008
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.qwenvl import DEFAULT_QWEN_MODEL_NAME, chat_qwenvl_call


class QwenVLRequestHandler(BaseHTTPRequestHandler):
    server_version = "QwenVLHTTP/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model": getattr(self.server, "model_name", DEFAULT_QWEN_MODEL_NAME),
                },
            )
            return
        self._write_json(404, {"error": f"Unsupported path: {self.path}"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._write_json(404, {"error": f"Unsupported path: {self.path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(content_length)
            payload = json.loads(request_body.decode("utf-8"))
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("Request JSON must contain a 'messages' list.")

            model_name = payload.get("model") or getattr(self.server, "model_name", DEFAULT_QWEN_MODEL_NAME)
            timeout = int(payload.get("timeout", getattr(self.server, "timeout", 240)))
            max_retries = int(payload.get("max_retries", 1))
            with self.server.inference_lock:
                response = chat_qwenvl_call(
                    messages=messages,
                    model_name=model_name,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            self._write_json(200, response)
        except Exception as exc:
            self._write_json(500, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistent Qwen3-VL chat service")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8008, help="Bind port")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL_NAME, help="Model name or local model path")
    parser.add_argument("--timeout", type=int, default=240, help="Default request timeout metadata")
    parser.add_argument("--quiet", action="store_true", help="Suppress HTTP access logs")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a tiny text request at startup so model loading happens before the first real request.",
    )
    args = parser.parse_args()

    if args.warmup:
        print(f"Warming up QwenVL model: {args.model}", flush=True)
        chat_qwenvl_call(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Return JSON: {\"ok\": true}"},
            ],
            model_name=args.model,
            timeout=args.timeout,
            max_retries=1,
        )
        print("Warmup complete.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), QwenVLRequestHandler)
    server.model_name = args.model
    server.timeout = args.timeout
    server.quiet = args.quiet
    server.inference_lock = threading.Lock()
    print(f"QwenVL service listening on http://{args.host}:{args.port}", flush=True)
    print(f"Model: {args.model}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down QwenVL service.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
