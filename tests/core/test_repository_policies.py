# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Repository policy discovery, reporting channels and source publication."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("CODE_OF_CONDUCT.md", "SECURITY.md")


@pytest.mark.parametrize("name", POLICIES)
def test_policies_are_discoverable(name: str) -> None:
    assert (REPO_ROOT / name).read_text(encoding="utf-8").strip()
    for entrypoint in ("README.md", "CONTRIBUTING.md"):
        assert f"]({name})" in (REPO_ROOT / entrypoint).read_text(encoding="utf-8")
    for language in ("", ".zh"):
        page = REPO_ROOT / f"docs/fumadocs/content/docs/maintainers/contributing{language}.mdx"
        assert f"/blob/main/{name}" in page.read_text(encoding="utf-8")


def test_reporting_uses_existing_maintainers_and_community_channels() -> None:
    assert (REPO_ROOT / ".github/CODEOWNERS").is_file()
    assert "discord.gg/" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in POLICIES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "](.github/CODEOWNERS)" in text
        assert "](README.md#-community)" in text
    security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "when that option is available" in security
    assert "current `main` branch" in security


def test_source_manifest_preserves_policy_documents() -> None:
    from setuptools._distutils.filelist import FileList

    files = FileList()
    files.allfiles = list(POLICIES)
    files.files = list(POLICIES)
    for line in (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            files.process_template_line(line)
    assert set(POLICIES) <= set(files.files)
