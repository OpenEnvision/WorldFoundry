from __future__ import annotations

import numpy as np
import pytest

import worldfoundry.core.media.processing.presets as presets_module
from worldfoundry.core.media.processing.postprocess import (
    IdentityVideoPostProcessor,
    VideoChunk,
    VideoPostprocessChain,
    VideoPostProcessor,
    VideoPostProcessorSession,
    VideoPostprocessStream,
    VideoSpec,
    frame_list_from_chunks,
)
from worldfoundry.core.media.processing.presets import (
    available_postprocess_presets,
    build_postprocessor,
    normalize_postprocess_preset_name,
)
from worldfoundry.core.media.processing.rtx import (
    RTXVideoSuperResolutionConfig,
    RTXVideoSuperResolutionPostProcessor,
    rtx_postprocessor_from_preset,
)

pytestmark = pytest.mark.unit

_STUB_PRESET = "test-one-chunk-delay"


class _OneChunkDelayProcessor(VideoPostProcessor):
    def __init__(self, *, preset: str, device: int | None, options: dict[str, object]) -> None:
        self.preset = preset
        self.device = device
        self.options = options
        self.events: list[str] = []

    @property
    def name(self) -> str:
        return _STUB_PRESET

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        self.events.append("start")
        return _OneChunkDelaySession(self.events)


class _OneChunkDelaySession(VideoPostProcessorSession):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.pending: VideoChunk | None = None
        self.closed = False

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        if self.closed:
            raise RuntimeError("cannot process after flush()")
        self.events.append(f"process:{chunk.metadata['autoregressive_index']}")
        previous, self.pending = self.pending, chunk
        return [] if previous is None else [previous]

    def flush(self) -> list[VideoChunk]:
        if self.closed:
            return []
        self.closed = True
        self.events.append("flush")
        tail, self.pending = self.pending, None
        return [] if tail is None else [tail]

    def close(self) -> None:
        self.events.append("close")
        self.pending = None
        self.closed = True


def _one_chunk_delay_preset(preset: str, *, device: int | None = None, **options: object) -> _OneChunkDelayProcessor:
    return _OneChunkDelayProcessor(preset=preset, device=device, options=options)


@pytest.fixture
def stub_preset(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(
        presets_module._POSTPROCESS_PRESET_FACTORIES,
        _STUB_PRESET,
        f"{__name__}:_one_chunk_delay_preset",
    )
    return _STUB_PRESET


def _frames(value: int) -> list[np.ndarray]:
    return [np.full((4, 6, 3), value, dtype=np.uint8)]


def test_available_postprocess_presets_lists_identity_and_rtx() -> None:
    assert available_postprocess_presets() == (
        "identity",
        "rtx-super-resolution",
        "rtx-super-resolution-ultra",
        "rtx-deblur-ultra",
    )


@pytest.mark.parametrize("name", ["", "  ", "none", "OFF", " Identity ", "identity"])
def test_identity_aliases_build_identity_processor(name: str) -> None:
    assert type(build_postprocessor(name)) is IdentityVideoPostProcessor


def test_identity_preset_rejects_options() -> None:
    with pytest.raises(ValueError, match="accepts no options"):
        build_postprocessor("identity", options={"scale": 2.0})


def test_preset_names_normalize_case_whitespace_and_underscores() -> None:
    assert normalize_postprocess_preset_name("  RTX_Super_Resolution_ULTRA ") == "rtx-super-resolution-ultra"
    processor = build_postprocessor(" RTX_Deblur_Ultra ")

    assert isinstance(processor, RTXVideoSuperResolutionPostProcessor)
    assert processor.config.quality == "DEBLUR_ULTRA"


def test_unknown_preset_lists_available_names() -> None:
    with pytest.raises(ValueError, match="Unknown postprocess preset 'swift-vr'") as error:
        build_postprocessor("swift-vr")

    for name in available_postprocess_presets():
        assert name in str(error.value)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rtx-super-resolution", RTXVideoSuperResolutionConfig()),
        ("rtx-super-resolution-ultra", RTXVideoSuperResolutionConfig(quality="ULTRA")),
        ("rtx-deblur-ultra", RTXVideoSuperResolutionConfig(scale=1.0, quality="DEBLUR_ULTRA")),
    ],
)
def test_rtx_presets_resolve_to_rtx_preset_configs(name: str, expected: RTXVideoSuperResolutionConfig) -> None:
    processor = build_postprocessor(name)

    assert isinstance(processor, RTXVideoSuperResolutionPostProcessor)
    assert processor.config == expected
    assert processor.config == rtx_postprocessor_from_preset(name).config


def test_rtx_preset_forwards_device_and_options() -> None:
    processor = build_postprocessor(
        "rtx-super-resolution",
        device=2,
        options={"scale": 1.5, "quality": "ULTRA", "non_blocking": True, "output_width": None},
    )

    assert isinstance(processor, RTXVideoSuperResolutionPostProcessor)
    assert processor.config == RTXVideoSuperResolutionConfig(scale=1.5, quality="ULTRA", device=2, non_blocking=True)
    assert processor.processing_device == "cuda:2"


def test_registered_preset_receives_normalized_name_device_and_options(stub_preset: str) -> None:
    assert stub_preset in available_postprocess_presets()

    processor = build_postprocessor("Test_One_Chunk_Delay", device=1, options={"window": 2})

    assert isinstance(processor, _OneChunkDelayProcessor)
    assert processor.preset == stub_preset
    assert processor.device == 1
    assert processor.options == {"window": 2}


def test_default_probe_is_a_no_op(stub_preset: str) -> None:
    processor = build_postprocessor(stub_preset)

    assert processor.probe(VideoSpec(height=4, width=6)) is None
    assert processor.probe(None) is None
    assert isinstance(processor, _OneChunkDelayProcessor)
    assert processor.events == []


def test_registered_stateful_preset_streams_in_order_and_flushes_tail(stub_preset: str) -> None:
    processor = build_postprocessor(stub_preset)
    assert isinstance(processor, _OneChunkDelayProcessor)
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((processor,)), fps=16)

    assert stream.process(_frames(0), layout="frame-list") == []
    assert stream.last_stats is not None
    assert stream.last_stats.buffering is True
    [first] = stream.process(_frames(1), layout="frame-list")
    assert stream.last_stats.buffering is False
    [tail] = stream.finish()

    assert first.metadata["autoregressive_index"] == 0
    assert tail.metadata["autoregressive_index"] == 1
    assert [int(frame[0, 0, 0]) for frame in frame_list_from_chunks([first, tail])] == [0, 1]
    assert stream.finish() == []
    with pytest.raises(RuntimeError, match="after finish"):
        stream.process(_frames(2), layout="frame-list")
    assert processor.events == ["start", "process:0", "process:1", "flush"]


def test_registered_stateful_preset_reset_closes_and_discards_buffer(stub_preset: str) -> None:
    processor = build_postprocessor(stub_preset)
    assert isinstance(processor, _OneChunkDelayProcessor)
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((processor,)), fps=16)

    assert stream.process(_frames(0), layout="frame-list") == []
    stream.reset()
    assert stream.process(_frames(5), layout="frame-list") == []
    [tail] = stream.finish()

    assert int(frame_list_from_chunks([tail])[0][0, 0, 0]) == 5
    assert tail.metadata["autoregressive_index"] == 0
    assert processor.events == ["start", "process:0", "close", "start", "process:0", "flush"]
