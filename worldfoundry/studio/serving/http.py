from __future__ import annotations

import json
import mimetypes
import os
import re
from collections.abc import Iterable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DEFAULT_CHUNK_SIZE = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), 1)
    except Exception:
        return default


class StudioThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server tuned for local artifact serving."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = _env_int("WORLDFOUNDRY_STUDIO_HTTP_BACKLOG", 64)


def parse_byte_range(value: str | None, size: int) -> tuple[int | None, int | None]:
    if size <= 0 or not value or not value.startswith("bytes="):
        return None, None
    match = re.match(r"bytes=(\d*)-(\d*)$", value.strip())
    if not match:
        return None, None
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return None, None
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            if suffix <= 0:
                return None, None
            start = max(0, size - suffix)
            end = size - 1
    except ValueError:
        return None, None
    if start < 0 or start >= size or end < start:
        return None, None
    return start, min(size - 1, end)


def path_allowed(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            root_resolved = root.resolve()
        except Exception:
            continue
        if resolved == root_resolved:
            return True
        try:
            resolved.relative_to(root_resolved)
            return True
        except ValueError:
            continue
    return False


def send_text_response(
    handler: BaseHTTPRequestHandler,
    text: str,
    content_type: str,
    *,
    status: HTTPStatus = HTTPStatus.OK,
    cache_control: str = "no-store",
) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", cache_control)
    handler.end_headers()
    handler.wfile.write(data)


def send_json_response(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, object],
    *,
    status: HTTPStatus = HTTPStatus.OK,
    cache_control: str = "no-store",
) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", cache_control)
    handler.end_headers()
    handler.wfile.write(data)


def send_file_response(
    handler: BaseHTTPRequestHandler,
    path: Path,
    content_type: str | None = None,
    *,
    range_header: str | None = None,
    cache_control: str = "no-store",
    chunk_size: int | None = None,
) -> None:
    stat = path.stat()
    file_size = stat.st_size
    if file_size <= 0:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", content_type or _guess_content_type(path))
        handler.send_header("Content-Length", "0")
        handler.send_header("Cache-Control", cache_control)
        handler.end_headers()
        return

    start, end = parse_byte_range(range_header, file_size)
    status = HTTPStatus.PARTIAL_CONTENT if start is not None else HTTPStatus.OK
    if start is None:
        start = 0
        end = file_size - 1
    assert end is not None

    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type or _guess_content_type(path))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", cache_control)
    handler.send_header("Last-Modified", handler.date_time_string(stat.st_mtime))
    if status is HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.end_headers()
    _copy_file_to_socket(handler, path, start=start, length=length, chunk_size=chunk_size)


def _copy_file_to_socket(
    handler: BaseHTTPRequestHandler,
    path: Path,
    *,
    start: int,
    length: int,
    chunk_size: int | None,
) -> None:
    chunk_size = chunk_size or _env_int("WORLDFOUNDRY_STUDIO_FILE_CHUNK_BYTES", _DEFAULT_CHUNK_SIZE)
    with path.open("rb") as handle:
        if _sendfile(handler, handle.fileno(), start=start, length=length):
            return
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def _sendfile(handler: BaseHTTPRequestHandler, file_fd: int, *, start: int, length: int) -> bool:
    sendfile = getattr(os, "sendfile", None)
    if sendfile is None:
        return False
    sent_any = False
    try:
        handler.wfile.flush()
        socket_fd = handler.connection.fileno()
        offset = start
        remaining = length
        while remaining > 0:
            sent = sendfile(socket_fd, file_fd, offset, remaining)
            if sent == 0:
                break
            sent_any = True
            offset += sent
            remaining -= sent
        return remaining == 0
    except OSError:
        return sent_any


def _guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
