"""Regression coverage for WorldCam checkpoint-class bindings."""

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
    WanTextEncoder,
)
from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime import (
    model_registry,
)


def test_worldcam_registry_text_encoder_has_native_checkpoint_converter() -> None:
    classes = model_registry._MODEL_CLASSES

    assert classes["WanTextEncoder"] is WanTextEncoder
    converter = WanTextEncoder.state_dict_converter()
    state_dict = {"token_embedding.weight": object()}
    assert converter.from_civitai(state_dict) is state_dict
    assert converter.from_diffusers(state_dict) is state_dict


def test_worldcam_condition_video_disables_center_crop(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    from worldfoundry.core.io import VideoData
    from worldfoundry.pipelines.worldcam.pipeline_worldcam import WorldCamPipeline

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    wide = Image.new("RGB", (160, 80), (0, 255, 0))
    wide.paste((255, 0, 0), (0, 0, 20, 80))
    wide.paste((0, 0, 255), (140, 0, 160, 80))
    wide.save(frame_dir / "0.png")

    cropped = np.asarray(VideoData(image_folder=str(frame_dir), height=80, width=80)[0])
    uncropped = np.asarray(VideoData(image_folder=str(frame_dir), height=80, width=80, center_crop=False)[0])
    assert cropped[40, 0].tolist() == [0, 255, 0]
    assert uncropped[40, 0, 0] > 200
    assert uncropped[40, 79, 2] > 200

    mp4_path = tmp_path / "clip.mp4"
    mp4_path.write_bytes(b"")
    pipe = WorldCamPipeline(device="cpu")
    pipe.synthesis_model = type("_Model", (), {"height": 480, "width": 832})()

    # Probe _condition_video passes center_crop=False without decoding video bytes.
    import worldfoundry.core.io as core_io

    calls: list[dict[str, object]] = []

    class _FakeVideoData:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, **kwargs})
            self.length = 0

    orig = core_io.VideoData
    core_io.VideoData = _FakeVideoData
    try:
        video = pipe._condition_video_from_source(mp4_path, conditioning_frames=17)
    finally:
        core_io.VideoData = orig

    assert video.length == 17
    assert calls[0]["center_crop"] is False
