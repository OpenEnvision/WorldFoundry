from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "docs" / "fumadocs" / "scripts" / "serve-static.mjs"
NODE = shutil.which("node")


pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is required for the docs static server")


def reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_response(url: str, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return urlopen(url, timeout=1)
        except (OSError, URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def test_static_server_serves_extensionless_search_export_as_json(tmp_path: Path) -> None:
    payload = {"type": "i18n", "data": {"en": {}, "zh": {}}}
    search_paths = [
        tmp_path / "out" / "api" / "search",
        tmp_path / "out" / "WorldFoundry" / "api" / "search",
    ]
    for search_path in search_paths:
        search_path.parent.mkdir(parents=True)
        search_path.write_text(json.dumps(payload), encoding="utf-8")

    port = reserve_port()
    process = subprocess.Popen(
        [NODE, str(SERVER), "--listen", f"tcp://127.0.0.1:{port}"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        for request_path, search_path in zip(
            ("/api/search", "/WorldFoundry/api/search"), search_paths, strict=True
        ):
            url = f"http://127.0.0.1:{port}{request_path}"
            with wait_for_response(url) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == "application/json"
                assert json.load(response) == payload

            request = Request(url, method="HEAD")
            with urlopen(request, timeout=2) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == "application/json"
                assert int(response.headers["Content-Length"]) == search_path.stat().st_size
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
