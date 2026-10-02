from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import worldfoundry.core.video.rtx as rtx_module
from worldfoundry.core.video.postprocess import VideoPostprocessChain, VideoPostprocessStream, VideoSpec
from worldfoundry.core.video.rtx import (
    RTXVFXCapability,
    RTXVideoSuperResolutionConfig,
    RTXVideoSuperResolutionPostProcessor,
    inspect_rtx_vfx_capability,
    probe_rtx_vfx_runtime,
    rtx_postprocessor_from_preset,
)


class _FakeVideoSuperRes:
    class QualityLevel:
        HIGH = "HIGH"
        ULTRA = "ULTRA"
        DEBLUR_ULTRA = "DEBLUR_ULTRA"

    instances: list["_FakeVideoSuperRes"] = []

    def __init__(self, *, quality: str, device: int) -> None:
        self.quality = quality
        self.device = device
        self.output_width = 0
        self.output_height = 0
        self.loaded = False
        self.closed = False
        self.calls: list[tuple[tuple[int, ...], bool, int]] = []
        type(self).instances.append(self)

    def load(self) -> None:
        self.loaded = True

    def run(self, frame: torch.Tensor, *, non_blocking: bool, stream_ptr: int) -> SimpleNamespace:
        assert self.loaded
        assert frame.dtype == torch.float32
        assert float(frame.min()) >= 0.0
        assert float(frame.max()) <= 1.0
        self.calls.append((tuple(frame.shape), non_blocking, stream_ptr))
        image = F.interpolate(
            frame.unsqueeze(0),
            size=(self.output_height, self.output_width),
            mode="nearest",
        )[0]
        return SimpleNamespace(image=image)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_vfx(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeVideoSuperRes.instances.clear()
    monkeypatch.setattr(rtx_module, "_load_video_super_res_class", lambda: _FakeVideoSuperRes)
    monkeypatch.setattr(rtx_module, "_torch_device_for_nvvfx_device", lambda _device: torch.device("cpu"))


def test_rtx_config_declares_scaled_preserved_stream_contract() -> None:
    config = RTXVideoSuperResolutionConfig(scale=2.0)

    output = config.output_spec(VideoSpec(height=360, width=640, fps=24, dtype="uint8"))

    assert output == VideoSpec(height=720, width=1280, fps=24, channels=3, dtype="uint8")
    with pytest.raises(ValueError, match="both output_width and output_height"):
        RTXVideoSuperResolutionConfig(output_width=1280)
    with pytest.raises(ValueError, match="same-resolution"):
        RTXVideoSuperResolutionConfig(scale=2.0, quality="DEBLUR_ULTRA").output_spec(VideoSpec(height=360, width=640))
    with pytest.raises(ValueError, match="at most 4x"):
        RTXVideoSuperResolutionConfig(scale=5.0).output_spec(VideoSpec(height=360, width=640))


def test_rtx_frame_list_preserves_numpy_uint8_contract_and_closes_effect() -> None:
    first = np.array(
        [
            [[0, 16, 32], [64, 80, 96]],
            [[128, 144, 160], [224, 240, 255]],
        ],
        dtype=np.uint8,
    )
    frames = [first, np.flip(first, axis=1).copy()]
    stream = VideoPostprocessStream(
        chain=VideoPostprocessChain((RTXVideoSuperResolutionPostProcessor(scale=2.0),)),
        fps=12,
    )

    [chunk] = stream.process(frames, layout="frame-list", metadata={"seed": 42})

    assert chunk.layout == "frame-list"
    assert len(chunk.frames) == 2
    assert all(isinstance(frame, np.ndarray) for frame in chunk.frames)
    assert all(frame.shape == (4, 4, 3) and frame.dtype == np.uint8 for frame in chunk.frames)
    expected = np.repeat(np.repeat(first, 2, axis=0), 2, axis=1)
    assert np.array_equal(chunk.frames[0], expected)
    assert chunk.metadata["seed"] == 42
    assert chunk.metadata["postprocess"] == "rtx-video-super-resolution"
    assert chunk.metadata["rtx_output_size"] == [4, 4]
    assert stream.output_spec == VideoSpec(height=4, width=4, fps=12, channels=3, dtype="uint8")
    assert len(_FakeVideoSuperRes.instances) == 1
    effect = _FakeVideoSuperRes.instances[0]
    assert effect.calls == [((3, 2, 2), False, 0), ((3, 2, 2), False, 0)]

    stream.reset()
    assert effect.closed


@pytest.mark.parametrize(
    ("layout", "shape", "expected_shape"),
    [
        ("thwc", (2, 2, 3, 3), (2, 4, 6, 3)),
        ("tchw", (2, 3, 2, 3), (2, 3, 4, 6)),
        ("bthwc", (1, 2, 2, 3, 3), (1, 2, 4, 6, 3)),
        ("btchw", (1, 2, 3, 2, 3), (1, 2, 3, 4, 6)),
        ("bcthw", (1, 3, 2, 2, 3), (1, 3, 2, 4, 6)),
        ("bvthwc", (1, 2, 2, 2, 3, 3), (1, 2, 2, 4, 6, 3)),
        ("bvtchw", (1, 2, 2, 3, 2, 3), (1, 2, 2, 3, 4, 6)),
    ],
)
def test_rtx_preserves_tensor_layout_dtype_and_minus_one_one_range(
    layout: str,
    shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    video = torch.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=torch.float16).reshape(shape)
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((RTXVideoSuperResolutionPostProcessor(scale=2.0),)))

    [chunk] = stream.process(video, layout=layout)  # type: ignore[arg-type]

    assert torch.is_tensor(chunk.frames)
    assert tuple(chunk.frames.shape) == expected_shape
    assert chunk.frames.dtype == torch.float16
    assert float(chunk.frames.min()) >= -1.0
    assert float(chunk.frames.max()) <= 1.0


def test_rtx_preset_overrides_are_validated() -> None:
    processor = rtx_postprocessor_from_preset(
        "rtx-super-resolution-ultra",
        output_width=1280,
        output_height=720,
        device=2,
    )

    assert processor.config.quality == "ULTRA"
    assert processor.config.output_width == 1280
    assert processor.config.output_height == 720
    assert processor.config.device == 2
    with pytest.raises(ValueError, match="Unknown RTX postprocess preset"):
        rtx_postprocessor_from_preset("unknown")


def test_rtx_capability_reports_missing_optional_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtx_module.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(rtx_module, "_installed_nvidia_vfx_version", lambda: None)

    capability = inspect_rtx_vfx_capability(device=0)

    assert capability.available is False
    assert "not installed" in capability.reason
    assert capability.to_payload()["compute_capability"] is None
    assert capability.to_payload()["probe_status"] == "not-run"


def test_rtx_quality_resolver_supports_published_wheel_enum_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _VideoSuperResWithoutNestedEnum:
        pass

    fallback = SimpleNamespace(HIGH="wheel-high")
    monkeypatch.setattr(rtx_module, "_load_quality_level_enum", lambda: fallback)

    assert rtx_module._resolve_quality(_VideoSuperResWithoutNestedEnum, "HIGH") == "wheel-high"


def test_rtx_runtime_probe_loads_runs_and_releases_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rtx_module,
        "inspect_rtx_vfx_capability",
        lambda *, device: RTXVFXCapability(
            available=True,
            device=device,
            reason="static checks passed",
            package_version="0.1.0.1",
            gpu_name="Fake RTX",
        ),
    )

    capability = probe_rtx_vfx_runtime(
        device=0,
        config=RTXVideoSuperResolutionConfig(scale=2.0),
        input_spec=VideoSpec(height=2, width=3, dtype="uint8"),
    )

    assert capability.available is True
    assert capability.probe_status == "passed"
    assert capability.backend_error is None
    effect = _FakeVideoSuperRes.instances[-1]
    assert effect.calls == [((3, 2, 3), False, 0)]
    assert effect.closed is True


def test_rtx_runtime_probe_preserves_backend_error_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingVideoSuperRes(_FakeVideoSuperRes):
        def load(self) -> None:
            raise OSError("NGX feature unavailable")

    monkeypatch.setattr(rtx_module, "_load_video_super_res_class", lambda: _FailingVideoSuperRes)
    monkeypatch.setattr(
        rtx_module,
        "inspect_rtx_vfx_capability",
        lambda *, device: RTXVFXCapability(
            available=True,
            device=device,
            reason="static checks passed",
        ),
    )

    capability = probe_rtx_vfx_runtime(
        device=0,
        config=RTXVideoSuperResolutionConfig(scale=2.0),
        input_spec=VideoSpec(height=2, width=3, dtype="uint8"),
    )

    assert capability.available is False
    assert capability.probe_status == "failed"
    assert capability.backend_error is not None
    assert "NGX feature unavailable" in capability.backend_error
    assert _FailingVideoSuperRes.instances[-1].closed is True


def test_rtx_effect_initialization_failure_is_actionable_and_releases_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingVideoSuperRes(_FakeVideoSuperRes):
        def load(self) -> None:
            raise OSError("driver is too old")

    monkeypatch.setattr(rtx_module, "_load_video_super_res_class", lambda: _FailingVideoSuperRes)
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((RTXVideoSuperResolutionPostProcessor(scale=2.0),)))

    with pytest.raises(RuntimeError, match="platform and driver requirements"):
        stream.process([np.zeros((2, 3, 3), dtype=np.uint8)], layout="frame-list")

    assert _FailingVideoSuperRes.instances[-1].closed
