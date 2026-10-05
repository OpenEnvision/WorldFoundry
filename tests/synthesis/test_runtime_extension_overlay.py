from __future__ import annotations

import sys
from pathlib import Path

from worldfoundry.base_models.three_dimensions.three_d_four_d.runtime_extension_overlay import (
    add_runtime_extension_overlay,
)


def test_add_runtime_extension_overlay_prepends_existing_directory(
    monkeypatch, tmp_path: Path
) -> None:
    overlay = tmp_path / "simple_knn"
    overlay.mkdir()
    monkeypatch.setenv("TEST_EXTENSION_DIR", str(overlay))

    configured = add_runtime_extension_overlay(
        "ignored", environment_variable="TEST_EXTENSION_DIR"
    )

    assert configured == overlay
    assert sys.path[0] == str(overlay)


def test_add_runtime_extension_overlay_ignores_missing_directory(
    monkeypatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("TEST_EXTENSION_DIR", str(missing))
    assert (
        add_runtime_extension_overlay(
            "ignored", environment_variable="TEST_EXTENSION_DIR"
        )
        is None
    )
