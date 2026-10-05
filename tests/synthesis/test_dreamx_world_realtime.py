from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "diffusers" dependency at import time; skip when it is unavailable.
pytest.importorskip("diffusers")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from worldfoundry.pipelines.dreamx_world import DreamXWorld5BCamPipeline
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.dispatch import _requested_torchrun_nproc
from worldfoundry.studio.inference.execution import TORCHRUN_LINGBOT_FAST_ENV, StudioManager
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout
from worldfoundry.synthesis.visual_generation.dreamx_world import realtime
from worldfoundry.synthesis.visual_generation.dreamx_world.runtime.utils.prompt_embeddings import (
    validate_prompt_embedding_pair,
)
from worldfoundry.synthesis.visual_generation.shared.wan_diffusers import WanDiffusersInferenceMixin
from worldfoundry.core.nn import AutoencoderKLOutput, DiagonalGaussianDistribution


def _checkpoint(root: Path, required: tuple[str, ...]) -> Path:
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    return root


def test_checkpoint_resolver_requires_every_released_component(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "DreamX", realtime._DREAMX_REQUIRED)
    assert realtime._resolve_checkpoint(
        checkpoint,
        default_name="unused",
        required=realtime._DREAMX_REQUIRED,
        label="DreamX",
    ) == checkpoint.resolve()

    (checkpoint / realtime._DREAMX_REQUIRED[-1]).unlink()
    with pytest.raises(FileNotFoundError, match=realtime._DREAMX_REQUIRED[-1]):
        realtime._resolve_checkpoint(
            checkpoint,
            default_name="unused",
            required=realtime._DREAMX_REQUIRED,
            label="DreamX",
        )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(5, 5), (17, 17), (33, 33), (34, 33), (81, 81)],
)
def test_native_frame_count_is_snapped_to_one_plus_four_k(requested: int, expected: int) -> None:
    assert realtime._snap_model_frames(requested) == expected


def test_compile_warmup_minimum_matches_native_vae_window() -> None:
    assert realtime.MIN_MODEL_FRAMES == 5
    assert realtime._snap_model_frames(realtime.MIN_MODEL_FRAMES) == 5


def test_controls_preserve_composed_keyboard_motion_and_segment_duration() -> None:
    result = realtime._frame_keys(
        [
            {"duration": 1.0, "keys": ["w", "l"]},
            {"duration": 1.0, "keys": ["a"]},
        ],
        [],
        frame_count=4,
        fps=2,
    )

    assert result == [frozenset({"w", "l"}), frozenset({"w", "l"}), frozenset({"a"}), frozenset({"a"})]


def test_decoded_video_is_contiguous_rgb_uint8() -> None:
    decoded = np.linspace(0.0, 1.0, 1 * 3 * 5 * 4 * 6, dtype=np.float32).reshape(1, 3, 5, 4, 6)
    video = realtime._normalize_decoded_video(decoded, expected_frames=5)

    assert video.shape == (5, 4, 6, 3)
    assert video.dtype == np.uint8
    assert video.flags.c_contiguous


def test_shared_wan_vae_encoder_accepts_native_output_object() -> None:
    class _VAE:
        def encode(self, value):
            moments = torch.cat((value, torch.zeros_like(value)), dim=1)
            return AutoencoderKLOutput(
                latent_dist=DiagonalGaussianDistribution(moments, deterministic=True)
            )

    pipeline = object.__new__(WanDiffusersInferenceMixin)
    pipeline.vae = _VAE()
    value = torch.ones((2, 1, 1, 2, 2))

    result = pipeline._encode_vae_mode(value, device="cpu", dtype=torch.float32)

    torch.testing.assert_close(result, value)


def test_cached_prompt_embeddings_allow_different_token_lengths() -> None:
    positive = [np.zeros((12, 4096), dtype=np.float32)]
    negative = [np.zeros((3, 4096), dtype=np.float32)]

    validate_prompt_embedding_pair(positive, negative)

    with pytest.raises(ValueError, match="hidden width"):
        validate_prompt_embedding_pair(
            positive,
            [np.zeros((3, 2048), dtype=np.float32)],
        )


def test_pipeline_keeps_one_session_and_forwards_user_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class _Session:
        def __init__(self, checkpoint_source, *, wan_model_path):
            calls.append(("init", (checkpoint_source, wan_model_path)))

        def realtime_spec(self):
            return SimpleNamespace(to_payload=lambda: {"fps": 16})

        def prepare(self):
            calls.append(("prepare", None))
            return {"realtime_spec": {"fps": 16}}

        def runtime_info(self):
            return {"resident": True}

        def configure(self, image, **kwargs):
            calls.append(("configure", (image, kwargs)))
            return {"configured": True}

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return {"frames": np.zeros((4, 8, 8, 3), dtype=np.uint8)}

        def reset(self):
            calls.append(("reset", None))

    monkeypatch.setattr(realtime, "DreamXWorldRealtimeSession", _Session)
    pipeline = DreamXWorld5BCamPipeline.from_pretrained(
        "/checkpoints/dreamx",
        wan_model_path="/checkpoints/wan",
    )
    image = Image.new("RGB", (32, 32), "navy")
    assert pipeline.prepare_realtime() == {"realtime_spec": {"fps": 16}}
    pipeline.configure_realtime(image, prompt="user prompt")
    result = pipeline.stream_realtime(
        interactions=["forward", "camera_r"],
        realtime_segments=[{"duration": 0.25, "keys": ["w", "l"]}],
    )

    assert result["frames"].shape == (4, 8, 8, 3)
    assert [name for name, _ in calls].count("init") == 1
    assert [name for name, _ in calls].count("prepare") == 1
    assert calls[-1][1]["interactions"] == ["forward", "camera_r"]


def test_catalog_is_user_input_only_and_uses_local_assets() -> None:
    entry = find_entry("dreamx-world-5b-cam")

    assert entry.default_prompt == ""
    assert entry.default_input_path == ""
    assert entry.default_interactions == ()
    assert entry.default_load_kwargs["nproc_per_node"] == 8
    assert "user-input-only" in entry.tags
    assert "in-tree-runtime" in entry.tags


def test_user_can_select_dreamx_sequence_parallel_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", "6")
    assert _requested_torchrun_nproc("dreamx-world-5b-cam", {}) == 6

    monkeypatch.setenv("WORLD_SIZE", "6")
    assert launch_uses_lingbot_torchrun_rollout(
        StudioLaunchConfig(model_id="dreamx-world-5b-cam", frontend="world")
    )
    monkeypatch.setenv(TORCHRUN_LINGBOT_FAST_ENV, "1")
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    assert manager._should_use_torchrun_lingbot_fast(
        SimpleNamespace(model_id="dreamx-world-5b-cam"),
        SimpleNamespace(backend="from_pretrained"),
    )
