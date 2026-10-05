from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldfoundry.pipelines.dreamx_world import DreamXWorld5BARPipeline
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.dreamx_world.ar_realtime import (
    DreamXWorldARRealtimeSession,
    _pixel_key_frames,
    _rotation_matrix,
)


def test_ar_catalog_uses_the_causal_user_input_runtime() -> None:
    entry = find_entry("dreamx-world")

    assert entry.model_id == "dreamx-world-5b"
    assert "causal" in entry.tags
    assert "autoregressive" in entry.tags
    assert "user-input-only" in entry.tags
    assert "num_inference_steps" not in entry.default_call_kwargs
    assert DreamXWorld5BARPipeline.MODEL_ID == "dreamx-world-5b"


def test_ar_first_and_steady_control_blocks_follow_native_cadence() -> None:
    first = _pixel_key_frames(("forward", "camera_l"), None, first_block=True)
    steady = _pixel_key_frames(("forward", "camera_l"), None, first_block=False)

    assert first == [frozenset({"w", "j"})] * 8
    assert steady == [frozenset({"w", "j"})] * 12


def test_ar_control_resampling_preserves_short_taps() -> None:
    frames = _pixel_key_frames(
        (),
        (
            {"duration": 0.0625, "keys": ["w"]},
            {"duration": 0.4375, "keys": ["l"]},
        ),
        first_block=True,
    )

    assert frames[0] == frozenset({"w"})
    assert frames[1:] == [frozenset({"l"})] * 7


def test_ar_identity_rotation_is_stable() -> None:
    np.testing.assert_allclose(_rotation_matrix(0.0, 0.0), np.eye(3), atol=1e-6)


def test_ar_camera_pose_uses_causal_vae_timestamps_across_chunks() -> None:
    session = DreamXWorldARRealtimeSession.__new__(DreamXWorldARRealtimeSession)
    session.device = torch.device("cpu")
    session.dtype = torch.float32
    session._first_block = True
    session._position = np.zeros(3, dtype=np.float32)
    session._pitch = 0.0
    session._yaw = 0.0
    session._previous_latent_c2w = np.eye(4, dtype=np.float32)

    first = session._camera_condition(("forward",), None)
    session._first_block = False
    second = session._camera_condition(("forward",), None)

    first_samples = first["viewmats"][0, ::880, 2, 3].float().numpy()
    np.testing.assert_allclose(first_samples, [0.0, -0.05, -0.25], atol=1e-6)
    np.testing.assert_allclose(session._position[2], 1.0, atol=1e-6)

    # The first steady latent is pixel frame 9 and is expressed relative to
    # the previous sampled latent at pixel frame 5: four frames * 0.05 units.
    second_first = second["viewmats"][0, 0, 2, 3].float().item()
    np.testing.assert_allclose(second_first, -0.2, atol=1e-6)


def test_ar_commits_only_the_clean_latent_to_persistent_kv_cache() -> None:
    calls: list[str] = []

    class Scheduler:
        @staticmethod
        def add_noise(denoised, _noise, _timesteps):
            return denoised

    class Generator:
        scheduler = Scheduler()

        def __call__(self, **kwargs):
            calls.append(kwargs["cache_update_policy"])
            latent = kwargs["noisy_image_or_video"]
            return torch.zeros_like(latent), latent * 0.5

    class VAE:
        @staticmethod
        def decode_to_pixel(_latent, *, use_cache):
            assert use_cache is True
            return torch.zeros(1, 9, 3, 2, 2)

    session = DreamXWorldARRealtimeSession.__new__(DreamXWorldARRealtimeSession)
    session.device = torch.device("cpu")
    session.dtype = torch.float32
    session.checkpoint = Path("checkpoint")
    session.generator = Generator()
    session.vae = VAE()
    session.timesteps = torch.tensor([1000.0, 750.0, 500.0, 250.0])
    session._configured = True
    session._conditional_dict = {"prompt_embeds": torch.zeros(1, 1, 1)}
    session._initial_latent = torch.zeros(1, 1, 48, 44, 80)
    session._kv_cache = []
    session._crossattn_cache = []
    session._first_block = True
    session._current_start_frame = 0
    session._seed = 42
    session._chunk_index = 0
    session._camera_condition = lambda *_args: {}

    session.generate(interactions=())

    assert calls == ["none", "none", "none", "none", "commit_detached"]
