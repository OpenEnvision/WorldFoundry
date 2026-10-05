from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from worldfoundry.studio.ui import vendor_assets as vendor_assets

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_COMMAND = "python -m worldfoundry.studio.ui.vendor_assets"


class _ChunkedResponse:
    def __init__(self, payload: bytes, *, chunk_size: int = 3) -> None:
        self._payload = payload
        self._chunk_size = chunk_size
        self._offset = 0
        self.requested_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.requested_sizes.append(size)
        if self._offset >= len(self._payload):
            return b""
        end = min(len(self._payload), self._offset + min(size, self._chunk_size))
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


def _asset(payload: bytes, *, relative_path: str = "example/module.js") -> vendor_assets.VendorAsset:
    return vendor_assets.VendorAsset(
        name="Test module",
        package="test-module",
        version="1.2.3",
        url="https://example.invalid/test-module.js",
        sha256=hashlib.sha256(payload).hexdigest(),
        relative_path=Path(relative_path),
    )


def _part_files(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.name.endswith(".part")]


@pytest.mark.parametrize("relative_path", ["/tmp/escape.js", "../escape.js", "nested/../../escape.js", "."])
def test_vendor_asset_path_must_stay_below_vendor_root(relative_path: str) -> None:
    with pytest.raises(ValueError, match="must stay below"):
        _asset(b"payload", relative_path=relative_path)


def test_pinned_vendor_asset_manifest_is_exact() -> None:
    assert vendor_assets.VENDOR_ASSET_INSTALL_COMMAND == INSTALL_COMMAND
    assert [
        (
            asset.package,
            asset.version,
            asset.url,
            asset.sha256,
            asset.relative_path.as_posix(),
        )
        for asset in vendor_assets.VENDOR_ASSETS
    ] == [
        (
            "@sparkjsdev/spark",
            "0.1.10",
            "https://cdn.jsdelivr.net/npm/@sparkjsdev/spark@0.1.10/dist/spark.module.js",
            "e2841904c3facdf2ab5177b13b4827cdc72118cb8b613673ca08d8e983c5bf9d",
            "spark/spark.module.min.js",
        ),
        (
            "three",
            "0.178.0",
            "https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.module.js",
            "bc0d236927f5163414e7c59a5567257dfe925f1929ce0a151ac4185dc45ca5a2",
            "three/three.module.js",
        ),
        (
            "three",
            "0.178.0",
            "https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.core.js",
            "562b72799ef1145f77997ece49a34f578422873757b0a13e41d76dcbfb776f06",
            "three/three.core.js",
        ),
    ]


def test_check_mode_is_offline_and_returns_nonzero_for_missing_assets(tmp_path: Path, capsys) -> None:
    def unexpected_opener(*args, **kwargs):
        raise AssertionError("--check must not open the network")

    assert vendor_assets.main(["--check"], root=tmp_path, opener=unexpected_opener) == 1
    captured = capsys.readouterr()
    assert "missing" in captured.out
    assert INSTALL_COMMAND in captured.err


def test_valid_asset_skips_network(tmp_path: Path) -> None:
    payload = b"already valid"
    asset = _asset(payload)
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    def unexpected_opener(*args, **kwargs):
        raise AssertionError("valid assets must not be downloaded again")

    statuses = vendor_assets.provision_assets(
        root=tmp_path,
        assets=(asset,),
        opener=unexpected_opener,
    )
    assert len(statuses) == 1
    assert statuses[0].valid
    assert target.read_bytes() == payload


def test_chunked_download_is_verified_and_atomically_published(tmp_path: Path) -> None:
    payload = b"a browser module delivered in several chunks"
    asset = _asset(payload)
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old invalid bytes")
    response = _ChunkedResponse(payload)
    calls = []

    def opener(url: str, *, timeout: float):
        calls.append((url, timeout))
        return response

    statuses = vendor_assets.provision_assets(
        root=tmp_path,
        assets=(asset,),
        timeout=7.5,
        opener=opener,
    )

    assert calls == [(asset.url, 7.5)]
    assert len(response.requested_sizes) > 2
    assert all(size == 1024 * 1024 for size in response.requested_sizes)
    assert statuses[0].valid
    assert target.read_bytes() == payload
    assert _part_files(target.parent) == []


def test_download_hash_mismatch_preserves_target_and_cleans_temp_file(tmp_path: Path) -> None:
    expected_payload = b"expected bytes"
    old_payload = b"existing invalid bytes"
    asset = _asset(expected_payload)
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(old_payload)

    def opener(url: str, *, timeout: float):
        return _ChunkedResponse(b"tampered download")

    with pytest.raises(vendor_assets.VendorAssetError, match="SHA-256 verification"):
        vendor_assets.provision_assets(root=tmp_path, assets=(asset,), opener=opener)

    assert target.read_bytes() == old_payload
    assert _part_files(target.parent) == []


def test_download_size_limit_preserves_target_and_cleans_temp_file(tmp_path: Path, monkeypatch) -> None:
    asset = _asset(b"expected")
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing bytes")
    monkeypatch.setattr(vendor_assets, "MAX_DOWNLOAD_BYTES", 4)

    def opener(url: str, *, timeout: float):
        return _ChunkedResponse(b"five!", chunk_size=5)

    with pytest.raises(vendor_assets.VendorAssetError, match="safety limit"):
        vendor_assets.provision_assets(root=tmp_path, assets=(asset,), opener=opener)

    assert target.read_bytes() == b"existing bytes"
    assert _part_files(target.parent) == []


def test_unreadable_asset_is_reported_with_install_command(tmp_path: Path, monkeypatch) -> None:
    asset = _asset(b"expected")
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"present")

    def unreadable(path: Path) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(vendor_assets, "_file_sha256", unreadable)
    statuses = vendor_assets.check_assets(root=tmp_path, assets=(asset,))
    assert statuses[0].state == "unreadable"
    assert statuses[0].error == "permission denied"

    with pytest.raises(vendor_assets.VendorAssetError) as exc_info:
        vendor_assets.require_vendor_assets(root=tmp_path, assets=(asset,))
    assert INSTALL_COMMAND in str(exc_info.value)


def test_unreadable_asset_can_be_reprovisioned(tmp_path: Path, monkeypatch) -> None:
    payload = b"replacement bytes"
    asset = _asset(payload)
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unreadable placeholder")
    original_file_sha256 = vendor_assets._file_sha256
    checks = 0

    def initially_unreadable(path: Path) -> str:
        nonlocal checks
        checks += 1
        if checks == 1:
            raise OSError("transient read failure")
        return original_file_sha256(path)

    def opener(url: str, *, timeout: float):
        return _ChunkedResponse(payload)

    monkeypatch.setattr(vendor_assets, "_file_sha256", initially_unreadable)
    statuses = vendor_assets.provision_assets(root=tmp_path, assets=(asset,), opener=opener)
    assert statuses[0].valid
    assert target.read_bytes() == payload


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, -math.inf])
def test_timeout_must_be_positive_and_finite(tmp_path: Path, timeout: float) -> None:
    def unexpected_opener(*args, **kwargs):
        raise AssertionError("invalid timeouts must fail before network access")

    with pytest.raises(ValueError, match="finite number greater than zero"):
        vendor_assets.provision_assets(root=tmp_path, timeout=timeout, opener=unexpected_opener)


def test_require_vendor_assets_reports_hash_mismatch_and_install_command(tmp_path: Path) -> None:
    asset = _asset(b"expected")
    target = asset.path_under(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong")

    with pytest.raises(vendor_assets.VendorAssetError) as exc_info:
        vendor_assets.require_vendor_assets(root=tmp_path, assets=(asset,))

    message = str(exc_info.value)
    assert "SHA-256 mismatch" in message
    assert INSTALL_COMMAND in message


def test_standalone_spark_frontend_checks_assets_before_binding(monkeypatch) -> None:
    from worldfoundry.studio.visualization.backends import frontends

    def missing_assets() -> None:
        raise vendor_assets.VendorAssetError(f"missing; run {INSTALL_COMMAND}")

    monkeypatch.setattr(frontends, "require_vendor_assets", missing_assets)
    monkeypatch.setattr(
        frontends,
        "StudioThreadingHTTPServer",
        lambda *args, **kwargs: pytest.fail("server must not bind before asset validation"),
    )

    with pytest.raises(SystemExit, match=INSTALL_COMMAND):
        frontends.serve_spark_frontend(object(), object())


def test_catalog_and_frontends_explain_deferred_provisioning() -> None:
    interfaces_source = (REPO_ROOT / "worldfoundry/studio/ui/interfaces.py").read_text(encoding="utf-8")
    frontends_source = (REPO_ROOT / "worldfoundry/studio/visualization/backends/frontends.py").read_text(
        encoding="utf-8"
    )
    assert "Install pinned browser modules" in interfaces_source
    assert "max-age=31536000" not in frontends_source


def test_downloaded_vendor_assets_are_excluded_from_distributions() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        packaging = tomllib.load(handle)["tool"]["setuptools"]
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "assets/vendor/**/*" in packaging["exclude-package-data"]["*"]
    assert "prune worldfoundry/studio/assets/vendor" in manifest
    assert "worldfoundry/studio/assets/vendor/" in gitignore
