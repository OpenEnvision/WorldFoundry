# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""MCP inline media conversion follows the OpenAI Chat content schema."""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionContentPartParam
from pydantic import TypeAdapter

from worldfoundry.mcp.client import convert_result_to_openai_format

pytestmark = pytest.mark.unit


def _wav_data() -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 16)
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize("object_content", (False, True))
@pytest.mark.parametrize(
    "mime_type,audio_format",
    (("audio/wav", "wav"), ("audio/x-wav", "wav"), ("audio/mpeg", "mp3"), ("audio/mp3", "mp3")),
)
def test_audio_uses_inline_data_and_a_supported_format(object_content: bool, mime_type: str, audio_format: str) -> None:
    data = _wav_data() if audio_format == "wav" else base64.b64encode(b"ID3").decode("ascii")
    content: dict[str, str] | SimpleNamespace = {"type": "audio", "data": data, "mimeType": mime_type}
    if object_content:
        content = SimpleNamespace(**content)
    blocks = convert_result_to_openai_format(SimpleNamespace(content=[content]))
    assert blocks == [{"type": "input_audio", "input_audio": {"data": data, "format": audio_format}}]
    assert TypeAdapter(ChatCompletionContentPartParam).validate_python(blocks[0]) == blocks[0]


@pytest.mark.parametrize("object_content", (False, True))
def test_unsupported_audio_format_is_explicit(object_content: bool) -> None:
    content: dict[str, str] | SimpleNamespace = {"type": "audio", "data": "unused", "mime_type": "audio/ogg"}
    if object_content:
        content = SimpleNamespace(**content)
    with pytest.raises(ValueError, match="WAV or MP3"):
        convert_result_to_openai_format(content)


def test_text_and_image_conversion_remains_unchanged() -> None:
    blocks = convert_result_to_openai_format(
        [
            {"type": "text", "text": "ready"},
            {"type": "image", "data": "abcd", "mimeType": "image/png"},
        ]
    )
    assert blocks == [
        {"type": "text", "text": "ready"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abcd"}},
    ]
    adapter = TypeAdapter(ChatCompletionContentPartParam)
    for block in blocks:
        adapter.validate_python(block)
