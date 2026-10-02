from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worldfoundry.core.io.hf import (
    HF_URI_SCHEME,
    _allow_patterns_for_subpath,
    _download_snapshot,
    _parse_hf_uri,
    hf_download_or_fpath,
    materialize_hf_snapshot,
    resolve_hf_path,
    resolve_hf_snapshot_path,
)
from worldfoundry.core.io.easy_io import resolve_checkpoint_path


def test_resolve_hf_path_passthrough_for_non_string() -> None:
    assert resolve_hf_path(None) is None
    assert resolve_hf_path(42) == 42


def test_resolve_hf_path_returns_existing_local_path(tmp_path: Path) -> None:
    local_file = tmp_path / "checkpoint.safetensors"
    local_file.write_text("ok", encoding="utf-8")

    assert resolve_hf_path(str(local_file)) == str(local_file)


def test_resolve_hf_path_returns_non_hf_path_unchanged() -> None:
    missing = "/tmp/does-not-exist/worldfoundry-hf-test"
    assert resolve_hf_path(missing) == missing


def test_parse_hf_uri_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError, match="Invalid HF path"):
        _parse_hf_uri(f"{HF_URI_SCHEME}owner-only")


def test_parse_hf_uri_splits_repo_and_subpath() -> None:
    repo_id, subpath = _parse_hf_uri("hf://Efficient-Large-Model/SANA-WM/dit/model.safetensors")
    assert repo_id == "Efficient-Large-Model/SANA-WM"
    assert subpath == "dit/model.safetensors"


def test_allow_patterns_cover_directory_subtrees() -> None:
    assert _allow_patterns_for_subpath("refiner") == [
        "refiner",
        "refiner/*",
        "refiner/**",
    ]
    assert _allow_patterns_for_subpath("") is None


@patch("worldfoundry.core.io.hf.materialize_hf_snapshot", return_value=Path("/cache/repo-root"))
def test_resolve_hf_path_downloads_hf_uri_rank_safely(mock_materialize) -> None:
    resolved = resolve_hf_path("hf://owner/repo/checkpoints/model.pth")

    mock_materialize.assert_called_once_with(
        "owner/repo",
        allow_patterns=["checkpoints/model.pth", "checkpoints/model.pth/*", "checkpoints/model.pth/**"],
    )
    assert resolved == "/cache/repo-root/checkpoints/model.pth"


@patch("worldfoundry.core.io.hf.materialize_hf_snapshot", return_value=Path("/cache/repo-root"))
def test_hf_download_or_fpath_is_alias(mock_materialize) -> None:
    resolved = hf_download_or_fpath("hf://owner/repo")

    mock_materialize.assert_called_once_with("owner/repo", allow_patterns=None)
    assert resolved == "/cache/repo-root"


def test_materialize_hf_snapshot_downloads_on_rank_zero_then_reopens_locally() -> None:
    patterns = ["weights/model.safetensors"]
    with (
        patch("worldfoundry.core.io.hf.maybe_download_hf_repo_on_rank0") as mock_rank_safe_download,
        patch("worldfoundry.core.io.hf._snapshot_download", return_value="/cache/repo-root") as mock_snapshot,
    ):
        resolved = materialize_hf_snapshot("owner/repo", allow_patterns=patterns)

    mock_rank_safe_download.assert_called_once_with(
        "owner/repo",
        revision=None,
        cache_dir=None,
        allow_patterns=patterns,
        ignore_patterns=None,
        token=None,
    )
    mock_snapshot.assert_called_once_with(
        repo_id="owner/repo",
        revision=None,
        cache_dir=None,
        local_files_only=True,
        allow_patterns=patterns,
        ignore_patterns=None,
        token=None,
    )
    assert resolved == Path("/cache/repo-root")


def test_download_snapshot_preflights_and_holds_repo_lock(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hub"
    with (
        patch("worldfoundry.core.io.hf.cache_min_free_bytes", return_value=0),
        patch("worldfoundry.core.io.hf.ensure_free_disk") as mock_preflight,
        patch("worldfoundry.core.io.hf.FileLock") as mock_lock,
        patch("worldfoundry.core.io.hf._snapshot_download") as mock_snapshot,
    ):
        _download_snapshot(
            "owner/repo",
            revision="revision",
            cache_dir=cache_dir,
            allow_patterns=["weights/*"],
            ignore_patterns=None,
        )

    mock_preflight.assert_called_once()
    mock_lock.assert_called_once()
    assert mock_lock.call_args.args[0].endswith(".lock")
    mock_snapshot.assert_called_once_with(
        "owner/repo",
        revision="revision",
        cache_dir=str(cache_dir),
        local_files_only=False,
        allow_patterns=["weights/*"],
        ignore_patterns=None,
        token=None,
    )


def test_resolve_checkpoint_path_delegates_hf_uri() -> None:
    with patch("worldfoundry.core.io.hf.resolve_hf_path", return_value="/cache/model.pth") as mock_resolve:
        resolved = resolve_checkpoint_path("hf://owner/repo/model.pth")

    mock_resolve.assert_called_once_with("hf://owner/repo/model.pth")
    assert resolved == "/cache/model.pth"


def test_resolve_checkpoint_path_expands_local_path() -> None:
    assert resolve_checkpoint_path("~/checkpoint.pth").endswith("checkpoint.pth")


def test_resolve_hf_snapshot_path_still_handles_repo_ids(tmp_path: Path) -> None:
    local_dir = tmp_path / "local-repo"
    local_dir.mkdir()
    assert resolve_hf_snapshot_path(str(local_dir)) == local_dir
