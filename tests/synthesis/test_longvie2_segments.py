from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from worldfoundry.pipelines.longvie.pipeline_longvie import LongVie2Pipeline
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends.frontends import resolve_frontend_mode
from worldfoundry.studio.visualization.backends.world import world_frontend_html
from worldfoundry.studio.visualization.backends.world_realtime_client import (
    WORLD_REALTIME_CLIENT_JS,
)
from worldfoundry.synthesis.visual_generation.longvie.longvie_synthesis import (
    LONGVIE_HISTORY_FRAMES,
    LONGVIE_SEGMENT_FPS,
    LONGVIE_SEGMENT_FRAMES,
    LongVieSynthesis,
)


DATA_ROOT = Path(__file__).resolve().parents[2] / "worldfoundry" / "data"


def _control_frames(count: int = LONGVIE_SEGMENT_FRAMES) -> np.ndarray:
    return np.zeros((count, 2, 2, 3), dtype=np.uint8)


@dataclass
class _FakeRuntime:
    use_usp: bool = True
    ring_degree: int = 1
    ulysses_degree: int = 4
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        return True

    def generate_segment(self, **kwargs: Any) -> tuple[list[Image.Image], Any]:
        self.calls.append(kwargs)
        segment = len(self.calls)
        frames = [Image.new("RGB", (2, 2), (segment, index % 255, 0)) for index in range(kwargs["num_frames"])]
        return frames, {"segment": segment}

    @staticmethod
    def is_output_rank() -> bool:
        return True

    def save_video(self, *_: Any, **__: Any) -> None:  # pragma: no cover - output is deliberately omitted
        raise AssertionError("segment tests must not encode MP4 output")


def _synthesis() -> tuple[LongVieSynthesis, _FakeRuntime]:
    runtime = _FakeRuntime()
    return LongVieSynthesis(model_id="longvie-2", runtime=runtime, execute_by_default=True), runtime


def test_longvie2_continuation_reuses_final_frame_history_and_noise() -> None:
    synthesis, runtime = _synthesis()
    controls = _control_frames()
    seed = Image.new("RGB", (2, 2), "white")

    first = synthesis.predict(
        prompt="move through the scene",
        images=seed,
        dense_video=controls,
        sparse_video=controls,
        target_size=(2, 2),
        return_dict=True,
    )
    first_final = first["video"][-1]
    second = synthesis.predict(
        prompt="continue along the path",
        dense_video=controls,
        sparse_video=controls,
        target_size=(2, 2),
        continue_from_memory=True,
        return_dict=True,
    )

    assert np.array_equal(np.asarray(runtime.calls[1]["input_image"]), np.asarray(first_final))
    assert runtime.calls[1]["history"] == first["video"][-LONGVIE_HISTORY_FRAMES:]
    assert runtime.calls[1]["noise"] == {"segment": 1}
    assert synthesis.history == second["video"][-LONGVIE_HISTORY_FRAMES:]
    assert synthesis.last_frame is second["video"][-1]
    assert second["metadata"]["continued_from_previous_final_frame"] is True
    assert second["segment_spec"]["mode"] == "queued_control_video_segments"
    assert second["segment_spec"]["realtime"] is False
    assert second["realtime_spec"] == {
        "fps": LONGVIE_SEGMENT_FPS,
        "first_chunk_frames": LONGVIE_SEGMENT_FRAMES,
        "steady_chunk_frames": LONGVIE_SEGMENT_FRAMES,
        "controls": ["dense_depth_video", "sparse_pointmap_or_track_video"],
        "transport": "queued-segment-rgb",
        "stateful": True,
    }


def test_longvie2_requires_all_user_conditioning_with_clear_error() -> None:
    synthesis, _ = _synthesis()

    with pytest.raises(ValueError) as exc_info:
        synthesis.predict(prompt="", execute=True)

    message = str(exc_info.value)
    assert "non-empty prompt" in message
    assert "initial image" in message
    assert "dense depth control video" in message
    assert "sparse pointmap/track control video" in message


def test_longvie2_rejects_short_controls_in_quality_mode() -> None:
    synthesis, _ = _synthesis()
    short_controls = _control_frames(9)

    with pytest.raises(ValueError, match="needs 81"):
        synthesis.predict(
            prompt="move through the scene",
            images=Image.new("RGB", (2, 2), "white"),
            dense_video=short_controls,
            sparse_video=short_controls,
            target_size=(2, 2),
            execute=True,
        )


def test_longvie2_catalog_has_runnable_full_quality_demo_defaults() -> None:
    entry = find_entry("longvie-2")

    assert entry.default_prompt
    assert entry.default_call_kwargs == {
        "execute": True,
        "dense_video": str(DATA_ROOT / "test_cases" / "longvie" / "dense_control.mp4"),
        "sparse_video": str(DATA_ROOT / "test_cases" / "longvie" / "sparse_control.mp4"),
        "num_frames": LONGVIE_SEGMENT_FRAMES,
        "height": 352,
        "width": 640,
        "fps": LONGVIE_SEGMENT_FPS,
        "seed": 0,
        "tiled": False,
        "num_inference_steps": 50,
    }
    assert entry.default_load_kwargs["torchrun_nproc_per_node"] == 4
    assert entry.default_load_kwargs["use_usp"] is True
    assert entry.default_load_kwargs["ring_degree"] == 1
    assert entry.default_load_kwargs["ulysses_degree"] == 4
    assert entry.default_input_path == str(DATA_ROOT / "test_cases" / "studio_demo" / "00" / "image.jpg")


def test_longvie2_auto_routes_to_native_queued_segment_surface() -> None:
    entry = find_entry("longvie-2")
    html = world_frontend_html(
        entry,
        StudioLaunchConfig(model_id=entry.model_id, frontend="auto"),
    )

    assert resolve_frontend_mode(entry, "auto") == "world"
    assert "QUEUED SEGMENTS" in html
    assert 'id="denseVideoInput"' in html
    assert 'id="sparseVideoInput"' in html
    assert ">RUN<" in html
    assert "HOLD WASD / DRAG STICKS" not in html
    assert 'type: "segment_update"' in WORLD_REALTIME_CLIENT_JS
    assert "dense_video_path" in WORLD_REALTIME_CLIENT_JS
    assert "sparse_video_path" in WORLD_REALTIME_CLIENT_JS
    assert "denseWasConsumed" in WORLD_REALTIME_CLIENT_JS
    assert "DISCONNECTED · START A NEW SESSION" in WORLD_REALTIME_CLIENT_JS


def test_longvie2_pipeline_stream_marks_continuation_without_encoding() -> None:
    synthesis, runtime = _synthesis()
    pipeline = LongVie2Pipeline(synthesis_model=synthesis)
    controls = _control_frames()

    pipeline(
        prompt="first segment",
        images=Image.new("RGB", (2, 2), "white"),
        dense_video=controls,
        sparse_video=controls,
        target_size=(2, 2),
        execute=True,
        return_dict=True,
    )
    result = pipeline.stream(
        prompt="second segment",
        dense_video=controls,
        sparse_video=controls,
        target_size=(2, 2),
        execute=True,
        return_dict=True,
    )

    assert len(runtime.calls) == 2
    assert result["metadata"]["continued_from_previous_final_frame"] is True
    assert result["artifact_path"] is None


def test_longvie2_extend_refuses_to_fake_state_after_process_restart() -> None:
    synthesis, _ = _synthesis()
    pipeline = LongVie2Pipeline(synthesis_model=synthesis)

    with pytest.raises(RuntimeError, match="completed segment in this resident pipeline"):
        pipeline.stream(
            prompt="cannot honestly continue yet",
            dense_video=_control_frames(),
            sparse_video=_control_frames(),
            target_size=(2, 2),
            execute=True,
            return_dict=True,
        )


@pytest.mark.parametrize(
    ("world_size", "use_usp", "ulysses_degree"),
    [("1", False, 1), ("4", True, 4)],
)
def test_longvie2_loader_normalizes_supported_gpu_topology(
    monkeypatch, world_size, use_usp, ulysses_degree
) -> None:
    monkeypatch.setenv("WORLD_SIZE", world_size)

    synthesis = LongVieSynthesis.from_pretrained(
        {"model_id": "longvie-2", "use_usp": not use_usp, "ring_degree": 9, "ulysses_degree": 9},
        device="cuda",
    )

    assert synthesis.runtime.loaded is False
    assert synthesis.runtime.use_usp is use_usp
    assert synthesis.runtime.ring_degree == 1
    assert synthesis.runtime.ulysses_degree == ulysses_degree
