from __future__ import annotations

import inspect

from worldfoundry.synthesis.visual_generation.evoke import worldfoundry_runtime


def test_evoke_probe_accepts_the_unified_torch_api_floor() -> None:
    probe = worldfoundry_runtime._COMPATIBILITY_PROBE

    assert 'Version("2.5") <= Version(versions["torch"])' in probe
    assert 'Version("2.7") <= Version(versions["torch"])' not in probe
    assert "import video_reader" not in probe


def test_evoke_video_conditioning_reuses_worldfoundry_video_io() -> None:
    from worldfoundry.synthesis.visual_generation.evoke.evoke_runtime.evoke.utils import ev_validation

    source = inspect.getsource(ev_validation._load_test_clip)
    assert "from worldfoundry.core.media.codecs.video import" in source
    assert "load_frames_from_video" in source
    assert "resize_video_tensor_to_resolution" in source
    assert "video_reader" not in source
