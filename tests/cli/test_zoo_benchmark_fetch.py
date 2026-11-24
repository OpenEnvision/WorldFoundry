# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""``worldfoundry zoo benchmark-fetch`` contracts against a local pinned repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from worldfoundry.cli.main import main

pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _write_manifest(manifest_dir: Path, runtime: dict[str, str]) -> None:
    entry = {
        "id": "pinned-bench",
        "name": "Pinned Bench",
        "status": "confirmed_official_code",
        "runner": {
            "runner_target": "worldfoundry.evals.tasks.contracts.external:vbench",
            "runtime": {"kind": "in_tree_official_source", **runtime},
        },
        "metrics": [{"id": "overall_quality", "higher_is_better": True, "primary": True}],
    }
    manifest_dir.mkdir()
    (manifest_dir / "external_benchmarks.yaml").write_text(json.dumps({"entries": [entry]}), encoding="utf-8")


def test_benchmark_fetch_materializes_the_pinned_official_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "upstream"
    (repo / "eval").mkdir(parents=True)
    (repo / "eval" / "score.py").write_text("print('score')\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "WorldFoundry")
    _git(repo, "config", "user.email", "tests@worldfoundry.invalid")
    _git(repo, "config", "uploadpack.allowReachableSHA1InWant", "true")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    revision = _git(repo, "rev-parse", "HEAD")
    (repo / "eval" / "score.py").write_text("print('newer score')\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "newer")
    manifest_dir = tmp_path / "benchmark_zoo"
    _write_manifest(
        manifest_dir,
        {
            "repo_url": repo.resolve().as_uri(),
            "repo_revision": revision,
            "repo_subdir": "eval",
            "runtime_root_env": "PINNED_BENCH_RUNTIME_ROOT",
        },
    )
    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(tmp_path / "cache"))
    # The runner's local-checkout override does not stop an explicit prefetch.
    monkeypatch.setenv("PINNED_BENCH_RUNTIME_ROOT", str(tmp_path / "local-checkout"))

    argv = ["zoo", "benchmark-fetch", "--manifest-dir", str(manifest_dir), "--benchmark-id", "pinned-bench", "--json"]
    assert main(argv) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["revision"] == revision
    assert payload["subdir"] == "eval"
    assert Path(payload["path"]) == tmp_path / "cache" / "sources" / "pinned-bench" / revision / "eval"
    assert (Path(payload["path"]) / "score.py").read_text(encoding="utf-8") == "print('score')\n"
