
import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from worldfoundry.pipelines.sana_wm import SanaWMPipeline
from worldfoundry.studio.inference.catalog import CURATED_OVERRIDES
from worldfoundry.synthesis.visual_generation.sana_wm.realtime import (
    DEFAULT_CFG_SCALE,
    DEFAULT_REALTIME_WINDOW_FRAMES,
    DEFAULT_SAMPLING_STEPS,
    SanaWMRealtimeSession,
    _compress_action_frames,
    _device_topology,
    _frames_for_segments,
    _snap_num_frames,
    _validate_output_video,
)


def test_sana_wm_temporal_window_preserves_8k_plus_one_layout() -> None:
    assert _snap_num_frames(9) == 9
    assert _snap_num_frames(159) == 161
    assert _snap_num_frames(161) == 161


def test_sana_wm_maps_local_component_checkpoints_to_their_real_layout(tmp_path) -> None:
    sana_root = tmp_path / "Efficient-Large-Model--SANA-WM_bidirectional"
    gemma_root = tmp_path / "Efficient-Large-Model--gemma-2-2b-it"

    overrides = SanaWMPipeline._checkpoint_overrides(str(sana_root), {})

    assert overrides == {
        "dit": str(sana_root),
        "text-encoder": str(gemma_root),
        "tokenizer": str(gemma_root),
        "codec": str(sana_root),
    }


def test_sana_wm_segments_cover_exact_output_window() -> None:
    frames = _frames_for_segments(
        [
            {"duration": 0.25, "keys": ["w"]},
            {"duration": 0.25, "keys": ["l"]},
        ],
        [],
        frame_count=8,
        fps=16,
    )
    assert frames == [frozenset("w")] * 4 + [frozenset("l")] * 4
    assert _compress_action_frames(frames) == "w-4,l-4"


def test_sana_wm_pipeline_stays_lazy_until_realtime_prepare() -> None:
    pipeline = SanaWMPipeline.from_pretrained("/tmp/checkpoint", model_id="sana-wm")
    assert pipeline.checkpoint_source == "/tmp/checkpoint"
    assert pipeline._realtime_session is None


def test_sana_wm_quality_defaults_use_short_native_window_and_full_sampler() -> None:
    assert DEFAULT_REALTIME_WINDOW_FRAMES == 81
    assert (DEFAULT_REALTIME_WINDOW_FRAMES - 1) % 8 == 0
    assert DEFAULT_SAMPLING_STEPS == 60
    assert DEFAULT_CFG_SCALE == 5.0


def test_sana_wm_catalog_has_no_baked_image_or_prompt_fixture() -> None:
    override = CURATED_OVERRIDES["sana-wm"]
    assert "default_input_path" not in override
    assert override["default_prompt"] == ""
    assert override["default_call_kwargs"]["window_frames"] == 81


def test_sana_wm_requires_a_user_image_before_loading_weights() -> None:
    pipeline = SanaWMPipeline.from_pretrained("/tmp/checkpoint", model_id="sana-wm")
    with pytest.raises(ValueError, match="requires a PIL image"):
        pipeline.configure_realtime(images=None, prompt="user prompt")
    assert pipeline._realtime_session is None


def test_sana_wm_requires_a_user_prompt_before_loading_weights() -> None:
    pipeline = SanaWMPipeline.from_pretrained("/tmp/checkpoint", model_id="sana-wm")
    with pytest.raises(ValueError, match="user-provided text prompt"):
        pipeline.configure_realtime(images=Image.new("RGB", (32, 18)), prompt="  ")
    assert pipeline._realtime_session is None


def test_sana_wm_forwards_user_window_and_quality_settings() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.config: dict[str, Any] = {}

        def configure(self, **kwargs: Any) -> dict[str, Any]:
            self.config = kwargs
            return {"realtime_spec": {"fps": 16}}

    session = FakeSession()
    pipeline = SanaWMPipeline.from_pretrained("/tmp/checkpoint", model_id="sana-wm")
    pipeline._realtime_session = session
    result = pipeline.configure_realtime(
        images=Image.new("RGB", (32, 18)),
        prompt="A user-authored world prompt",
        window_frames=161,
        step=60,
        cfg_scale=5.0,
        seed=7,
    )
    assert result["realtime_spec"] == {"fps": 16}
    assert session.config["num_frames"] == 161
    assert session.config["step"] == 60
    assert session.config["cfg_scale"] == 5.0
    assert session.config["seed"] == 7


def test_sana_wm_four_gpu_topology_partitions_resident_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    for name in (
        "WORLDFOUNDRY_SANA_WM_STAGE1_DEVICE",
        "WORLDFOUNDRY_SANA_WM_REFINER_DEVICE",
        "WORLDFOUNDRY_SANA_WM_REFINER_TEXT_DEVICE",
        "WORLDFOUNDRY_SANA_WM_VAE_DEVICE",
        "WORLDFOUNDRY_SANA_WM_STAGE1_TEXT_DEVICE",
        "WORLDFOUNDRY_SANA_WM_INTRINSICS_DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)
    topology = _device_topology()
    assert str(topology["stage1"]) == "cuda:0"
    assert str(topology["refiner"]) == "cuda:1"
    assert str(topology["refiner_text"]) == "cuda:2"
    assert str(topology["vae"]) == "cuda:3"
    assert topology["stage1_text"] == topology["vae"]
    assert topology["intrinsics"] == topology["vae"]


def test_sana_wm_rejects_mismatched_native_frame_count() -> None:
    with pytest.raises(RuntimeError, match="returned 79 frames; expected 80"):
        _validate_output_video(
            np.empty((79, 4, 4, 3), dtype=np.uint8),
            expected_frames=80,
        )


def _prompt_update_session() -> tuple[SanaWMRealtimeSession, Any, Any]:
    session = object.__new__(SanaWMRealtimeSession)
    session._state_lock = threading.RLock()
    session._configured = True
    session._prompt = "old prompt"
    session._image = object()
    session._intrinsics = np.ones((9, 4), dtype=np.float32)
    session._world_pose = np.eye(4, dtype=np.float32)
    session._chunk_index = 7
    session.last_metrics = {}

    old_stage1_cache = object()
    old_refiner_cache = object()
    refiner = SimpleNamespace(
        _cached_prompt="old prompt",
        _cached_prompt_tensors=old_refiner_cache,
    )
    pipeline = SimpleNamespace(
        _stage1_prompt_cache_key=("old prompt", ""),
        _stage1_prompt_cache=old_stage1_cache,
        refiner=refiner,
    )

    def encode_stage1(prompt: str, negative: str) -> None:
        pipeline._stage1_prompt_cache_key = (prompt, negative)
        pipeline._stage1_prompt_cache = f"stage1:{prompt}"

    def encode_refiner(prompt: str) -> None:
        refiner._cached_prompt = prompt
        refiner._cached_prompt_tensors = f"refiner:{prompt}"

    pipeline._encode_prompts = encode_stage1
    refiner._encode_prompt = encode_refiner
    session.pipeline = pipeline
    return session, old_stage1_cache, old_refiner_cache


def test_sana_wm_prompt_update_reconditions_both_stages_without_resetting_world() -> None:
    session, _, _ = _prompt_update_session()
    anchor = session._image
    intrinsics = session._intrinsics
    pose = session._world_pose.copy()

    assert session.update_prompt("  a rainy night  ") is True
    assert session._prompt == "a rainy night"
    assert session.pipeline._stage1_prompt_cache_key == ("a rainy night", "")
    assert session.pipeline.refiner._cached_prompt == "a rainy night"
    assert session._image is anchor
    assert session._intrinsics is intrinsics
    np.testing.assert_array_equal(session._world_pose, pose)
    assert session._chunk_index == 7
    assert session.update_prompt("a rainy night") is False


def test_sana_wm_generate_commits_prompt_before_the_same_chunk() -> None:
    session, _, _ = _prompt_update_session()
    observed: dict[str, Any] = {}

    def generate_current_prompt(**kwargs: Any) -> dict[str, Any]:
        observed["prompt"] = session._prompt
        observed["kwargs"] = kwargs
        return {"frames": [], "realtime_metrics": {}}

    session._generate_current_prompt = generate_current_prompt  # type: ignore[method-assign]
    result = session.generate(
        prompt="a rainy night",
        interactions=["forward"],
        seed=9,
    )

    assert observed == {
        "prompt": "a rainy night",
        "kwargs": {
            "interactions": ["forward"],
            "control_segments": None,
            "seed": 9,
        },
    }
    assert result["realtime_metrics"]["condition_ms"] >= 0.0


def test_sana_wm_prompt_update_rolls_back_both_text_caches_on_failure() -> None:
    session, old_stage1_cache, old_refiner_cache = _prompt_update_session()

    def fail_refiner(_prompt: str) -> None:
        raise RuntimeError("refiner encode failed")

    session.pipeline.refiner._encode_prompt = fail_refiner
    with pytest.raises(RuntimeError, match="refiner encode failed"):
        session.update_prompt("new prompt")

    assert session._prompt == "old prompt"
    assert session.pipeline._stage1_prompt_cache_key == ("old prompt", "")
    assert session.pipeline._stage1_prompt_cache is old_stage1_cache
    assert session.pipeline.refiner._cached_prompt == "old prompt"
    assert session.pipeline.refiner._cached_prompt_tensors is old_refiner_cache
    assert session._chunk_index == 7


def test_sana_wm_pipeline_forwards_prompt_update_to_same_chunk_boundary() -> None:
    generated: dict[str, Any] = {}

    class FakeSession:
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            generated.update(kwargs)
            return {"frames": np.zeros((8, 2, 2, 3), dtype=np.uint8)}

    pipeline = SanaWMPipeline.from_pretrained("/tmp/checkpoint", model_id="sana-wm")
    pipeline._realtime_session = FakeSession()
    pipeline.stream_realtime(
        prompt="a rainy night",
        interactions=["forward"],
        realtime_segments=[{"duration": 0.25, "keys": ["w"]}],
        seed=9,
    )

    assert generated == {
        "interactions": ["forward"],
        "control_segments": [{"duration": 0.25, "keys": ["w"]}],
        "seed": 9,
        "prompt": "a rainy night",
    }
