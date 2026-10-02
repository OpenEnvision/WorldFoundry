"""CPU-only regression tests for the ET-08 videoscore sys.path / monkeypatch fix.

`load_official_videoscore_module` used to prepend two vendored directories to
``sys.path`` permanently ("Temporarily" in name only), and
`patch_transformers_dynamic_cache_api` monkeypatched `DynamicCache` for the
whole process with no way to undo. Both leaks are now scoped.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.runners.videoscore.run_videoscore_official_runner import (
    load_official_videoscore_module,
    patch_transformers_dynamic_cache_api,
)


@pytest.fixture()
def fake_videoscore_root(tmp_path: Path) -> Path:
    root = tmp_path / "VideoScore"
    benchmark_dir = root / "benchmark"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "eval_videoscore.py").write_text("MARKER = 'fake-videoscore'\n", encoding="utf-8")
    return root


def test_loader_restores_sys_path_after_import(fake_videoscore_root: Path) -> None:
    before = list(sys.path)
    sys.modules.pop("eval_videoscore", None)
    try:
        module = load_official_videoscore_module(fake_videoscore_root)
        assert module.MARKER == "fake-videoscore"
        assert sys.path == before, "sys.path entries leaked after import"
        assert str(fake_videoscore_root) not in sys.path
        assert str(fake_videoscore_root / "benchmark") not in sys.path
    finally:
        sys.modules.pop("eval_videoscore", None)


def test_loader_restores_sys_path_when_import_fails(tmp_path: Path) -> None:
    root = tmp_path / "VideoScore"
    benchmark_dir = root / "benchmark"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "eval_videoscore.py").write_text("raise ImportError('missing heavy dep')\n", encoding="utf-8")
    before = list(sys.path)
    sys.modules.pop("eval_videoscore", None)
    try:
        with pytest.raises(ImportError, match="missing heavy dep"):
            load_official_videoscore_module(root)
        assert sys.path == before, "sys.path entries leaked after failed import"
    finally:
        sys.modules.pop("eval_videoscore", None)


def test_loader_does_not_remove_preexisting_entries(fake_videoscore_root: Path) -> None:
    preexisting = str(fake_videoscore_root)
    sys.path.insert(0, preexisting)
    sys.modules.pop("eval_videoscore", None)
    try:
        load_official_videoscore_module(fake_videoscore_root)
        assert preexisting in sys.path, "caller-owned sys.path entry must survive"
    finally:
        sys.modules.pop("eval_videoscore", None)
        while preexisting in sys.path:
            sys.path.remove(preexisting)


def test_transformers_patch_returns_working_undo(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDynamicCache:
        def get_seq_length(self, layer_idx: int = 0) -> int:
            return 11 + layer_idx

    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = FakeDynamicCache
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.cache_utils = fake_cache_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    undo = patch_transformers_dynamic_cache_api()
    assert undo is not None
    assert FakeDynamicCache().get_usable_length(128) == 11

    undo()
    assert not hasattr(FakeDynamicCache, "get_usable_length"), "undo must remove the monkeypatch"

    # Second undo call is a no-op, and re-patching still works.
    undo()
    assert patch_transformers_dynamic_cache_api() is not None
    assert FakeDynamicCache().get_usable_length(64) == 11


def test_transformers_patch_noop_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class AlreadyPatchedCache:
        def get_seq_length(self, layer_idx: int = 0) -> int:
            return 3

        def get_usable_length(self, _new_seq_length: int | None = None, layer_idx: int = 0) -> int:
            return 3

    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = AlreadyPatchedCache
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.cache_utils = fake_cache_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    assert patch_transformers_dynamic_cache_api() is None
