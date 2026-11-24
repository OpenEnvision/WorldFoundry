"""Provider-neutral construction of multimodal judge payloads."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .evidence.package import EvidencePackage

SYSTEM_PROMPT = "You are a strict PAWEval JSON judge. Return one valid JSON object and no prose."


class MediaTransportError(OSError):
    """Evidence media could not be read for a judge request."""


def build_request_payload(
    *,
    prompt: str,
    package: EvidencePackage,
) -> dict[str, Any]:
    """Build OpenAI-compatible messages from prepared local PAWEval evidence."""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if package.source_image.attached:
        content.extend(
            [
                {"type": "text", "text": "Source image used by the generator:"},
                _image_part(package.source_image.uri),
            ]
        )
    for frame in package.frames:
        content.extend(
            [
                {"type": "text", "text": f"Sampled video frame phase: {frame.phase}"},
                _image_part(frame.uri),
            ]
        )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
    }


def _image_part(uri: str | None) -> dict[str, Any]:
    path = _local_media_path(uri)
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise MediaTransportError(f"media is not readable: {path}") from exc
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(contents).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def _local_media_path(uri: str | None) -> Path:
    if not uri:
        raise MediaTransportError("media is missing a file URI")
    parsed = urlparse(uri)
    return Path(unquote(parsed.path if parsed.scheme == "file" else uri))
