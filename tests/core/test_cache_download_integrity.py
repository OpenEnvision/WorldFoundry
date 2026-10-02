from __future__ import annotations

import io
from pathlib import Path

import pytest

from worldfoundry.core.io import cache as io_cache
from worldfoundry.core.io import download as io_download


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        super().__init__(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)


def test_cache_path_uses_worldfoundry_root_not_torch_home(tmp_path: Path, monkeypatch) -> None:
    worldfoundry_cache = tmp_path / "worldfoundry"
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(worldfoundry_cache))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))

    target = io_cache._cache_path("https://example.test/assets/model.bin")

    assert target == worldfoundry_cache / "https/example.test/assets/model.bin"


def test_download_replaces_nonempty_cache_entry_that_fails_validation(tmp_path: Path, monkeypatch) -> None:
    payload = b"complete-payload"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "asset.bin"
    cached.write_bytes(b"corrupt")

    def validate(path: Path) -> None:
        if path.read_bytes() != payload:
            raise ValueError("invalid payload")

    monkeypatch.setattr(io_download, "cache_min_free_bytes", lambda: 0)
    monkeypatch.setattr(
        io_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )

    result = io_download.download_to_cache(
        "https://example.test/asset.bin",
        cache_dir=cache_dir,
        validator=validate,
    )

    assert result == cached
    assert cached.read_bytes() == payload
    assert not list(cache_dir.glob("*.part"))


def test_download_rejects_truncated_content_length_without_publishing(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    payload = b"truncated"
    monkeypatch.setattr(io_download, "cache_min_free_bytes", lambda: 0)
    monkeypatch.setattr(
        io_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload) + 5),
    )

    with pytest.raises(RuntimeError, match="Incomplete download"):
        io_download.download_to_cache("https://example.test/asset.bin", cache_dir=cache_dir)

    assert not (cache_dir / "asset.bin").exists()
    assert not list(cache_dir.glob("*.part"))


def test_zero_byte_cache_entry_is_redownloaded(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "asset.bin"
    cached.touch()
    payload = b"payload"
    monkeypatch.setattr(io_download, "cache_min_free_bytes", lambda: 0)
    monkeypatch.setattr(
        io_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )

    result = io_download.download_to_cache("https://example.test/asset.bin", cache_dir=cache_dir)

    assert result == cached
    assert cached.read_bytes() == payload


def test_validator_false_replaces_cached_entry(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "asset.bin"
    cached.write_bytes(b"invalid")
    payload = b"valid"
    monkeypatch.setattr(io_download, "cache_min_free_bytes", lambda: 0)
    monkeypatch.setattr(
        io_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )

    result = io_download.download_to_cache(
        "https://example.test/asset.bin",
        cache_dir=cache_dir,
        validator=lambda path: path.read_bytes() == payload,
    )

    assert result == cached
    assert cached.read_bytes() == payload


def test_validator_false_does_not_publish_download(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    payload = b"invalid"
    monkeypatch.setattr(io_download, "cache_min_free_bytes", lambda: 0)
    monkeypatch.setattr(
        io_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, content_length=len(payload)),
    )

    with pytest.raises(RuntimeError, match="download validator rejected"):
        io_download.download_to_cache(
            "https://example.test/asset.bin",
            cache_dir=cache_dir,
            validator=lambda _path: False,
        )

    assert not (cache_dir / "asset.bin").exists()
    assert not list(cache_dir.glob("*.part"))
