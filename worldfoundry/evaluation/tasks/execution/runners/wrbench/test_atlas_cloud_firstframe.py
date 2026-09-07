from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "runtime"))

from wrbench.firstframe import AtlasCloudT2IProvider  # noqa: E402


class _Response:
    def __init__(self, payload=None, content: bytes = b"") -> None:
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self) -> None:
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _Response({"data": {"id": "prediction-1"}})

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if url.endswith("/prediction/prediction-1"):
            return _Response({"data": {"status": "completed", "outputs": ["https://cdn.example/frame.png"]}})
        return _Response(content=b"png-bytes")


def test_atlas_cloud_firstframe_submits_once_and_polls(tmp_path):
    provider = AtlasCloudT2IProvider(
        model="bytedance/seedream-v5.0-lite",
        api_key="test-key",
        endpoint="https://api.atlascloud.ai/api/v1",
        size="2048*2048",
        n=1,
        poll_interval=0,
    )
    client = _Client()
    provider._client = client
    output = tmp_path / "frame.png"

    metadata = provider.generate(prompt="A clean test frame", family_id="test", out_path=output)

    assert len(client.posts) == 1
    assert client.posts[0][0] == "https://api.atlascloud.ai/api/v1/model/generateImage"
    assert client.posts[0][1]["json"] == {
        "model": "bytedance/seedream-v5.0-lite",
        "prompt": "A clean test frame",
        "size": "2048*2048",
        "output_format": "png",
    }
    assert client.gets[0][0] == "https://api.atlascloud.ai/api/v1/model/prediction/prediction-1"
    assert output.read_bytes() == b"png-bytes"
    assert metadata["provider"] == "atlascloud"


def test_atlas_cloud_firstframe_rejects_multiple_outputs():
    with pytest.raises(RuntimeError, match="n must be 1"):
        AtlasCloudT2IProvider(
            model="bytedance/seedream-v5.0-lite",
            api_key="test-key",
            endpoint="https://api.atlascloud.ai/api/v1",
            size="2048*2048",
            n=2,
        )
