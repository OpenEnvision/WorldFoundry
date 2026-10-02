from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_acquire_large_assets() -> ModuleType:
    """Load the large asset acquisition script for focused tests.

    Args:
        None: The repository root determines the script path.
    """
    path = REPO_ROOT / "tools" / "assets" / "acquire_large_assets.py"
    spec = importlib.util.spec_from_file_location("test_acquire_large_assets", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_zip_members_accepts_relative_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("models/raft-things.pth", "demo")

    module = _load_acquire_large_assets()
    with zipfile.ZipFile(archive_path) as archive:
        members = module.safe_zip_members(archive)

    assert [member.filename for member in members] == ["models/raft-things.pth"]


def test_safe_zip_members_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "demo")

    module = _load_acquire_large_assets()
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="parent traversal"):
            module.safe_zip_members(archive)


def test_safe_zip_members_rejects_absolute_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe_absolute.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("/tmp/escape.txt", "demo")

    module = _load_acquire_large_assets()
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="absolute"):
            module.safe_zip_members(archive)
