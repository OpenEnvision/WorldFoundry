from __future__ import annotations

import numpy as np
import pytest

from worldfoundry.core.media.processing.postprocess import (
    VideoPostprocessChain,
    VideoPostProcessor,
    VideoPostProcessorSession,
    VideoPostprocessStream,
)


class Processor(VideoPostProcessor):
    def __init__(self, name, events, *, fail=None):
        self.label, self.events, self.fail = name, events, fail

    def start(self, spec):
        self.events.append((self.label, "start"))
        if self.fail == "start":
            raise ValueError("start failed")
        return Session(self)


class Session(VideoPostProcessorSession):
    def __init__(self, processor):
        self.processor = processor
        self.buffer = []

    def prepare(self):
        self.processor.events.append((self.processor.label, "prepare"))
        if self.processor.fail == "prepare":
            raise ValueError("prepare failed")

    def process(self, chunk):
        self.buffer.append(chunk)
        if self.processor.fail == "process":
            raise ValueError("process failed")
        return []

    def flush(self):
        self.processor.events.append((self.processor.label, "flush"))
        if self.processor.fail == "flush":
            raise ValueError("flush failed")
        result, self.buffer = self.buffer, []
        return result

    def close(self):
        self.processor.events.append((self.processor.label, "close"))
        self.buffer.clear()
        if self.processor.fail == "close":
            raise ValueError("close failed")


def frames(value=0):
    return [np.full((4, 5, 3), value, dtype=np.uint8)]


def test_partial_start_closes_opened_sessions_and_preserves_start_error():
    events = []
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((
        Processor("first", events, fail="close"), Processor("second", events, fail="start"),
    )))
    with pytest.raises(ValueError, match="start failed"):
        stream.process(frames(), layout="frame-list")
    assert ("first", "close") in events
    assert stream.input_spec is None and stream.output_spec is None


def test_prepare_failure_closes_all_stages_and_allows_a_fresh_stream():
    events = []
    failing = Processor("first", events, fail="prepare")
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((failing, Processor("second", events))))
    with pytest.raises(ValueError, match="prepare failed"):
        stream.process(frames(17), layout="frame-list")
    assert all((name, "close") in events for name in ("first", "second"))
    failing.fail = None
    stream.process(frames(31), layout="frame-list")
    [chunk] = stream.finish()
    assert np.array_equal(chunk.frames[0], frames(31)[0])


def test_reset_closes_every_stage_even_when_one_close_fails():
    events = []
    first = Processor("first", events, fail="close")
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((first, Processor("second", events))))
    stream.process(frames(7), layout="frame-list")
    with pytest.raises(ValueError, match="close failed"):
        stream.reset()
    assert ("second", "close") in events
    assert stream.input_spec is None and stream.output_spec is None and stream.chunk_index == 0
    first.fail = None
    stream.process(frames(43), layout="frame-list")
    [chunk] = stream.finish()
    assert np.array_equal(chunk.frames[0], frames(43)[0])


def test_flush_failure_releases_remaining_stages_and_does_not_retry_tails():
    events = []
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((
        Processor("first", events, fail="flush"), Processor("second", events),
    )))
    stream.process(frames(), layout="frame-list")
    with pytest.raises(ValueError, match="flush failed"):
        stream.finish()
    assert ("second", "close") in events
    assert stream.finish() == []
    stream.reset()
    assert events.count(("first", "flush")) == 1


def test_reset_discards_prior_video_tail_and_preserves_per_stream_metadata():
    events = []
    processor = Processor("buffer", events)
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((processor,)))
    for value in (11, 73, 11):
        stream.process(frames(value), layout="frame-list", metadata={"session": value})
        [chunk] = stream.finish()
        assert chunk.metadata == {"session": value, "autoregressive_index": 0}
        assert np.array_equal(chunk.frames[0], frames(value)[0])
        stream.reset()


def test_reset_releases_resources_after_a_successful_flush_only_once():
    events = []
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((Processor("buffer", events),)))
    stream.process(frames(11), layout="frame-list")
    assert len(stream.finish()) == 1
    assert stream.finish() == []
    stream.reset()
    stream.reset()
    assert events.count(("buffer", "flush")) == 1
    assert events.count(("buffer", "close")) == 1


def test_failed_process_closes_all_stages_and_requires_reset_before_retry():
    events = []
    first = Processor("first", events, fail="process")
    stream = VideoPostprocessStream(chain=VideoPostprocessChain((first, Processor("second", events))))
    with pytest.raises(ValueError, match="process failed"):
        stream.process(frames(7), layout="frame-list")
    assert all((name, "close") in events for name in ("first", "second"))
    first.fail = None
    with pytest.raises(RuntimeError, match="after flush"):
        stream.process(frames(31), layout="frame-list")
    assert stream.finish() == []
    stream.reset()
    stream.process(frames(43), layout="frame-list")
    [chunk] = stream.finish()
    assert np.array_equal(chunk.frames[0], frames(43)[0])


def test_cleanup_preserves_primary_failure_without_exception_notes():
    from worldfoundry.core.media.processing.postprocess import _close_sessions

    class LegacyError(ValueError):
        add_note = None

    events = []
    primary = LegacyError("primary failed")
    _close_sessions([Session(Processor("first", events, fail="close")),
                     Session(Processor("second", events))], primary_error=primary)
    assert events == [("first", "close"), ("second", "close")]
