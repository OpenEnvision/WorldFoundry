# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise pull-request checks against real, isolated Git histories."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.dev import check_python_changes
from scripts.dev.check_dco import unsigned_commits
from scripts.dev.check_python_changes import changed_python_files

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ruff_hooks_do_not_repeat_the_upstream_entry_exclusion_flag() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    ruff = next(repo for repo in config["repos"] if repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit")
    assert {hook["id"] for hook in ruff["hooks"]} == {"ruff-check", "ruff-format"}
    for hook in ruff["hooks"]:
        assert "--force-exclude" not in hook.get("args", [])
    imports = next(hook for hook in ruff["hooks"] if hook.get("alias") == "ruff-import-sort")
    assert imports["args"] == ["--fix", "--select", "I"]


def test_ci_installs_declared_quality_and_protocol_extras() -> None:
    workflow = (REPO_ROOT / ".github/workflows/pr-checks.yml").read_text()
    assert 'python -m pip install -e ".[lint,test,mcp]" imageio' in workflow
    assert '"torch==2.9.0+cpu" "torchvision==0.24.0+cpu" --index-url https://download.pytorch.org/whl/cpu' in workflow
    assert "opencv-python-headless" in workflow
    assert '"einops>=0.7,<0.9"' in workflow


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    git(tmp_path, "symbolic-ref", "HEAD", "refs/heads/main")
    git(tmp_path, "config", "user.name", "Test Contributor")
    git(tmp_path, "config", "user.email", "contributor@example.com")
    git(tmp_path, "commit", "--allow-empty", "-m", "Base")
    return tmp_path


@pytest.mark.parametrize(
    ("message", "signed"),
    [
        ("Add feature\n\nSigned-off-by: Test Contributor <contributor@example.com>", True),
        ("Add feature", False),
        ("Add feature\n\nSigned-off-by: Test Contributor", False),
        ("Document Signed-off-by\n\nSigned-off-by: Name <name@example.com>\n\nExample only", False),
    ],
)
def test_dco_checks_each_contribution_commit(repo: Path, message: str, signed: bool) -> None:
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "commit", "--allow-empty", "-m", message)
    head = git(repo, "rev-parse", "HEAD")
    assert unsigned_commits(base, head, repo=repo) == ([] if signed else [head])
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/dev/check_dco.py"), "--base", base],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if signed else 1)


def test_dco_does_not_hide_an_unsigned_commit_behind_a_signed_one(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "commit", "--allow-empty", "-m", "Unsigned change")
    unsigned = git(repo, "rev-parse", "HEAD")
    git(repo, "commit", "--allow-empty", "--signoff", "-m", "Signed change")
    assert unsigned_commits(base, "HEAD", repo=repo) == [unsigned]


def test_python_selection_uses_pr_changes_and_ignores_deleted_files(repo: Path) -> None:
    (repo / "old.py").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Existing module")
    git(repo, "checkout", "-b", "feature")
    (repo / "new.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "README.md").write_text("Feature\n", encoding="utf-8")
    (repo / "old.py").unlink()
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Add feature")
    git(repo, "checkout", "main")
    (repo / "unrelated.py").write_text("value = 3\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Advance main")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "feature")
    assert changed_python_files(base, "HEAD", repo=repo) == ["new.py"]


def test_docs_only_python_check_does_not_require_quality_tools(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    (repo / "notes.md").write_text("Documentation\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Document feature")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/dev/check_python_changes.py"), "--base", base],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "No Python changes" in result.stdout


@pytest.mark.parametrize("failure_stage", [0, 1, 2, None])
def test_python_checks_propagate_tool_failures(monkeypatch: pytest.MonkeyPatch, failure_stage: int | None) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 1 if len(commands) - 1 == failure_stage else 0)

    monkeypatch.setattr(sys, "argv", ["check_python_changes.py", "--base", "main"])
    monkeypatch.setattr(check_python_changes, "changed_python_files", lambda *args, **kwargs: ["changed.py"])
    monkeypatch.setattr(check_python_changes.subprocess, "run", run)
    assert check_python_changes.main() == (0 if failure_stage is None else 1)
    assert [command[2:4] for command in commands] == [["ruff", "check"], ["ruff", "format"], ["ty", "check"]][
        : len(commands)
    ]
    if len(commands) == 3:
        assert commands[2][4:6] == ["--python", sys.executable]
