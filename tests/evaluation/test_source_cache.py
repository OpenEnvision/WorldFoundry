# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free, network-free contracts for pinned upstream source fetching."""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from worldfoundry.evals.tasks.execution.framework.source_cache import (
    SourceSpec,
    SourceUnavailableError,
    ensure_source,
    source_cache_root,
)

pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[str, str]:
    """A local repository served over ``file://``, pinned below its branch tip."""
    repo = tmp_path / "upstream"
    (repo / "evaluator").mkdir(parents=True)
    (repo / "evaluator" / "score.py").write_text("print('score')\n", encoding="utf-8")
    (repo / "assets.bin").write_bytes(b"large asset")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "WorldFoundry")
    _git(repo, "config", "user.email", "tests@worldfoundry.invalid")
    _git(repo, "config", "uploadpack.allowReachableSHA1InWant", "true")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    revision = _git(repo, "rev-parse", "HEAD")
    (repo / "evaluator" / "score.py").write_text("print('newer score')\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "newer")
    return repo.resolve().as_uri(), revision


@pytest.fixture
def cache_env(tmp_path: Path) -> dict[str, str]:
    return {"WORLDFOUNDRY_CACHE_DIR": str(tmp_path / "cache")}


def test_ensure_source_returns_the_requested_directory_of_the_pinned_tree(upstream, cache_env) -> None:
    url, revision = upstream

    root = ensure_source(SourceSpec("demo", url, revision, subdir="evaluator"), env=cache_env)

    assert root == source_cache_root(cache_env) / "demo" / revision / "evaluator"
    assert (root / "score.py").read_text(encoding="utf-8") == "print('score')\n"
    assert not (root.parent / ".git").exists()
    assert stat.S_IMODE(root.parent.stat().st_mode) == 0o755


def test_ensure_source_reuses_a_completed_cache_entry(upstream, cache_env, tmp_path: Path) -> None:
    url, revision = upstream
    first = ensure_source(SourceSpec("demo", url, revision), env=cache_env)
    shutil.rmtree(tmp_path / "upstream")

    assert ensure_source(SourceSpec("demo", url, revision), env=cache_env) == first
    assert (first / "assets.bin").read_bytes() == b"large asset"


def test_ensure_source_reports_unreachable_revisions(upstream, cache_env) -> None:
    url, _ = upstream
    missing = "0" * 40

    with pytest.raises(SourceUnavailableError, match=missing):
        ensure_source(SourceSpec("demo", url, missing), env=cache_env)
    assert list((source_cache_root(cache_env) / "demo").iterdir()) == []


def test_ensure_source_rejects_a_missing_subdirectory(upstream, cache_env) -> None:
    url, revision = upstream

    with pytest.raises(SourceUnavailableError, match="no directory 'absent'"):
        ensure_source(SourceSpec("demo", url, revision, subdir="absent"), env=cache_env)


def test_ensure_source_returns_an_absolute_path_for_a_relative_cache_dir(
    upstream, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, revision = upstream
    monkeypatch.chdir(tmp_path)

    root = ensure_source(SourceSpec("demo", url, revision), env={"WORLDFOUNDRY_CACHE_DIR": "relative-cache"})

    assert root == Path.cwd() / "relative-cache" / "sources" / "demo" / revision


def test_ensure_source_refuses_an_entry_it_did_not_complete(upstream, cache_env) -> None:
    url, revision = upstream
    entry = source_cache_root(cache_env) / "demo" / revision
    entry.mkdir(parents=True)
    (entry / "partial.py").write_text("", encoding="utf-8")

    with pytest.raises(SourceUnavailableError, match="not a completed fetch"):
        ensure_source(SourceSpec("demo", url, revision), env=cache_env)
    assert [path.name for path in entry.iterdir()] == ["partial.py"]


@pytest.mark.parametrize(
    ("source_id", "revision", "subdir", "message"),
    [
        ("demo", "main", "", "full commit SHA"),
        ("demo", "a" * 39, "", "full commit SHA"),
        ("nested/id", "a" * 40, "", "single path segment"),
        ("demo", "a" * 40, "../escape", "inside the repository"),
        ("demo", "a" * 40, "/abs", "inside the repository"),
    ],
)
def test_source_spec_requires_a_pinned_relative_tree(source_id, revision, subdir, message) -> None:
    with pytest.raises(ValueError, match=message):
        SourceSpec(source_id, "https://example.invalid/repo.git", revision, subdir=subdir)
