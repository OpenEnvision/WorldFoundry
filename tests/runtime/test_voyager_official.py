from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.hunyuan_world.voyager_official import (
    CHECKPOINT_FILES,
    VoyagerOfficialRuntime,
)


@pytest.fixture
def runtime(tmp_path):
    source = tmp_path / "source"
    (source / "voyager").mkdir(parents=True)
    (source / "sample_image2video.py").write_text("print('local source')\n", encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    for relative in CHECKPOINT_FILES:
        path = checkpoints / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(1_000_000)
    return VoyagerOfficialRuntime(
        source_root=source,
        checkpoint_root=checkpoints,
        python_executable=sys.executable,
    )


def test_preflight_accepts_local_source_edits_without_scanning_bytes(runtime, monkeypatch) -> None:
    (runtime.source_root / "voyager" / "local.py").write_text("# local adjustment\n", encoding="utf-8")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("preflight scanned source bytes")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)

    report = runtime.preflight()

    assert report["status"] == "ready"
    assert report["missing"] == []
    assert "source_revision" not in report
    assert "source_digest" not in report


def test_preflight_requires_the_official_entrypoint(runtime) -> None:
    (runtime.source_root / "sample_image2video.py").unlink()

    report = runtime.preflight()

    assert report["status"] == "blocked"
    assert any("official entrypoint missing" in reason for reason in report["missing"])
