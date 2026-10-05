"""Video catalog YAML must stay aligned with dispatch registries.

There is no historical pytest tree on this public branch (``tests/`` and
``test_*.py`` are gitignored). This module is the shipped exception: CI runs
the same assertions via ``make workspace-registry-check`` / ``make lint``.
"""

from __future__ import annotations

from worldfoundry.evaluation.tasks.catalog.integrity import (
    catalog_runner_table_issues,
    video_catalog_dispatch_issues,
)
from worldfoundry.evaluation.tasks.catalog.workspace_registry import (
    validate_workspace_registry,
)


def test_validate_workspace_registry_has_no_issues() -> None:
    issues = validate_workspace_registry()
    assert issues == [], "\n".join(issues)


def test_video_catalog_and_runner_registry_are_explained() -> None:
    issues = video_catalog_dispatch_issues()
    assert issues == [], "\n".join(issues)


def test_catalog_is_the_only_runner_table() -> None:
    issues = catalog_runner_table_issues()
    assert issues == [], "\n".join(issues)


if __name__ == "__main__":
    test_validate_workspace_registry_has_no_issues()
    test_video_catalog_and_runner_registry_are_explained()
    test_catalog_is_the_only_runner_table()
    print("ok")
