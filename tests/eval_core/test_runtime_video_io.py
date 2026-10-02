from __future__ import annotations

from pathlib import Path

import numpy as np
import torchvision.io

from worldfoundry.core.io import video as video_io
from worldfoundry.core.io import (
    VIDEO_EXTENSIONS,
    coerce_video_frames,
    extract_frames_from_video_url,
    load_video_frames,
    materialize_video_input,
    save_video_frames,
    video_tensor_to_uint8_frames,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hosted_video_runtimes_share_url_frame_extractor() -> None:
    assert callable(extract_frames_from_video_url)

    runtime_paths = [
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/minimax/minimax_runtime.py",
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/runway/gen3_runtime.py",
    ]
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        assert "from worldfoundry.core.io import extract_frames_from_video_url" in text
        assert "def extract_frames_from_url" not in text
        assert "cv2.VideoCapture(tmp_file.name)" not in text


def test_local_video_helpers_are_shared_runtime_api() -> None:
    assert ".mp4" in VIDEO_EXTENSIONS
    for helper in (
        coerce_video_frames,
        load_video_frames,
        materialize_video_input,
        save_video_frames,
        video_tensor_to_uint8_frames,
    ):
        assert callable(helper)

    lyra_text = (REPO_ROOT / "worldfoundry/pipelines/lyra/lyra_utils.py").read_text(encoding="utf-8")
    inspatio_synthesis_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_synthesis.py"
    ).read_text(encoding="utf-8")
    inspatio_runtime_text = (
        REPO_ROOT / "worldfoundry/synthesis/visual_generation/inspatio_world/worldfoundry_runtime.py"
    ).read_text(encoding="utf-8")

    assert "from worldfoundry.core.io import" in lyra_text
    assert "from worldfoundry.core.io import" in inspatio_runtime_text
    assert "worldfoundry.pipelines.lyra.lyra_utils" not in inspatio_runtime_text
    assert "worldfoundry.pipelines.lyra.lyra_utils" not in inspatio_synthesis_text
    assert "def materialize_video_input(" not in lyra_text
    assert "def load_video_frames(" not in lyra_text


def test_torchvision_writer_falls_back_for_newer_pyav_pict_type(monkeypatch, tmp_path) -> None:
    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    encoded = {}

    def incompatible_torchvision_writer(*args, **kwargs):
        raise TypeError("an integer is required")

    def fallback_writer(video_frames, output_path, *, fps):
        encoded["shape"] = video_frames.shape
        encoded["path"] = output_path
        encoded["fps"] = fps

    monkeypatch.setattr(torchvision.io, "write_video", incompatible_torchvision_writer)
    monkeypatch.setattr(video_io, "save_video_h264", fallback_writer)
    output_path = tmp_path / "output.mp4"

    video_io.write_video_torchvision(output_path, frames, 16)

    assert encoded == {"shape": (2, 4, 6, 3), "path": output_path, "fps": 16}
