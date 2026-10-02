from __future__ import annotations

import numpy as np
import pytest

from worldfoundry.core.media.processing.postprocess import (
    IdentityVideoPostProcessor,
    VideoChunk,
    VideoPostprocessChain,
    VideoPostProcessor,
    VideoPostProcessorSession,
    VideoPostprocessStream,
    VideoSpec,
    frame_list_from_chunks,
    infer_video_spec,
    video_frame_count,
)


def _frames(count: int = 2, *, height: int = 4, width: int = 5) -> list[np.ndarray]:
    return [np.full((height, width, 3), index, dtype=np.uint8) for index in range(count)]


def test_identity_postprocess_stream_is_zero_copy_and_reports_stats() -> None:
    frames = _frames()
    stream = VideoPostprocessStream(
        chain=VideoPostprocessChain((IdentityVideoPostProcessor(),)),
        fps=16,
    )

    [chunk] = stream.process(frames, layout="frame-list", metadata={"source": "test"})

    assert chunk.frames is frames
    assert chunk.metadata == {"source": "test", "autoregressive_index": 0}
    assert stream.input_spec == VideoSpec(
        height=4,
        width=5,
        channels=3,
        fps=16,
        dtype="uint8",
    )
    assert stream.output_spec == stream.input_spec
    assert stream.last_stats is not None
    assert stream.last_stats.input_frames == stream.last_stats.output_frames == 2
    assert stream.last_stats.buffering is False
    assert stream.finish() == []
    with pytest.raises(RuntimeError, match="after finish"):
        stream.process(frames, layout="frame-list")

    stream.reset()
    [fresh] = stream.process(frames, layout="frame-list")
    assert fresh.metadata["autoregressive_index"] == 0


class _RecordingProcessor(VideoPostProcessor):
    def __init__(self, name: str, events: list[str], *, buffered_tail: int) -> None:
        self._name = name
        self.events = events
        self.buffered_tail = buffered_tail

    @property
    def name(self) -> str:
        return self._name

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        self.events.append(f"start:{self._name}:{spec.width}x{spec.height}")
        return _RecordingSession(self._name, self.events, self.buffered_tail)


class _RecordingSession(VideoPostProcessorSession):
    def __init__(self, name: str, events: list[str], buffered_tail: int) -> None:
        self.name = name
        self.events = events
        self.buffered_tail = buffered_tail
        self.closed = False

    def prepare(self) -> None:
        self.events.append(f"prepare:{self.name}")

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        self.events.append(f"process:{self.name}:{chunk.frame_count}")
        chunk.metadata.setdefault("processors", []).append(self.name)
        return [chunk]

    def flush(self) -> list[VideoChunk]:
        if self.closed:
            return []
        self.closed = True
        self.events.append(f"flush:{self.name}")
        if not self.buffered_tail:
            return []
        return [
            VideoChunk(
                frames=_frames(self.buffered_tail),
                layout="frame-list",
                metadata={"tail": self.name},
            )
        ]


def test_postprocess_chain_flushes_each_tail_only_through_downstream_sessions() -> None:
    events: list[str] = []
    stream = VideoPostprocessStream(
        chain=VideoPostprocessChain(
            (
                _RecordingProcessor("first", events, buffered_tail=1),
                _RecordingProcessor("second", events, buffered_tail=2),
            )
        ),
        fps=12,
    )

    [chunk] = stream.process(_frames(), layout="frame-list")
    tails = stream.finish()

    assert chunk.metadata["processors"] == ["first", "second"]
    assert [tail.frame_count for tail in tails] == [1, 2]
    assert tails[0].metadata == {"tail": "first", "processors": ["second"]}
    assert tails[1].metadata == {"tail": "second"}
    assert events == [
        "start:first:5x4",
        "start:second:5x4",
        "prepare:first",
        "prepare:second",
        "process:first:2",
        "process:second:2",
        "flush:first",
        "process:second:1",
        "flush:second",
    ]
    assert stream.finish() == []


def test_postprocess_stream_rejects_midstream_shape_changes() -> None:
    stream = VideoPostprocessStream()
    stream.process(_frames(height=4), layout="frame-list")

    with pytest.raises(ValueError, match="specification changed"):
        stream.process(_frames(height=6), layout="frame-list")


def test_video_layout_helpers_validate_shapes_and_preserve_frame_order() -> None:
    tensor = np.zeros((1, 2, 3, 4, 5, 3), dtype=np.float32)
    spec = infer_video_spec(tensor, layout="bvthwc", fps=24)

    assert spec == VideoSpec(height=4, width=5, channels=3, fps=24, dtype="float32")
    assert video_frame_count(tensor, layout="bvthwc") == 3
    chunks = [
        VideoChunk(frames=_frames(1), layout="frame-list"),
        VideoChunk(frames=_frames(2), layout="frame-list"),
    ]
    assert len(frame_list_from_chunks(chunks)) == 3
    with pytest.raises(ValueError, match="requires 5 dimensions"):
        infer_video_spec(np.zeros((2, 4, 5, 3)), layout="bthwc")
