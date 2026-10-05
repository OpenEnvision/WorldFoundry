from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workspace" / "validate_generated_videos.py"
SPEC = importlib.util.spec_from_file_location("worldfoundry_validate_generated_videos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_generated_videos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_generated_videos
SPEC.loader.exec_module(validate_generated_videos)


def test_duration_uses_ffprobe_value() -> None:
    assert validate_generated_videos._duration_seconds(
        {"duration": "2.75"}, decoded_frames=12, fps=4.0
    ) == pytest.approx(2.75)


@pytest.mark.parametrize("raw_duration", [None, "N/A", "nan", "-1"])
def test_duration_falls_back_to_decoded_frames_and_fps(raw_duration: str | None) -> None:
    assert validate_generated_videos._duration_seconds(
        {"duration": raw_duration}, decoded_frames=12, fps=4.0
    ) == pytest.approx(3.0)


def test_duration_reports_zero_without_usable_timing() -> None:
    assert validate_generated_videos._duration_seconds(
        {"duration": "N/A"}, decoded_frames=12, fps=0.0
    ) == 0.0
