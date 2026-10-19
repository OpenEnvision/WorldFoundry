from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from worldfoundry.operators.matrix_game_3_operator import MatrixGame3Operator
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3_runtime.realtime import (
    MatrixGame3RealtimeSession,
)


class _FakeRuntime:
    defaults: dict = {}

    def __init__(self, sp_size: int = 1) -> None:
        self.pipeline = SimpleNamespace(sp_size=sp_size, device=torch.device("cpu"))

    def ensure_resident_pipeline(self):
        return self.pipeline


def _session(monkeypatch) -> MatrixGame3RealtimeSession:
    monkeypatch.setattr(MatrixGame3RealtimeSession, "_load_helpers", lambda _self: {})
    return MatrixGame3RealtimeSession(_FakeRuntime(), MatrixGame3Operator())


def test_native_realtime_cadence_and_latent_windows(monkeypatch) -> None:
    session = _session(monkeypatch)

    assert session.realtime_spec().to_payload() == {
        "fps": 17,
        "first_chunk_frames": 57,
        "steady_chunk_frames": 40,
        "controls": [
            "forward",
            "backward",
            "left",
            "right",
            "camera_up",
            "camera_down",
            "camera_l",
            "camera_r",
        ],
        "transport": "in-memory-rgb",
        "stateful": True,
    }
    assert session.next_output_frames() == 57
    assert session._latent_index(0) == 0
    assert session._latent_index(57) == 15
    assert session._latent_index(41) == 11
    assert session._latent_index(97) == 25
    assert session._latent_index(97) - session._latent_index(41) == 14
    assert session._vae_cache == [None] * 32

    session._append_latents(torch.arange(15).view(1, 1, 15, 1, 1))
    session._append_latents(torch.arange(15, 25).view(1, 1, 10, 1, 1))
    gathered = session._gather_latent_memory([0, 14, 15, 24])
    assert gathered.flatten().tolist() == [0, 14, 15, 24]

    session._clip_index = 1
    assert session.next_output_frames() == 40
    session.reset()
    assert session.next_output_frames() == 57
    assert session._vae_cache == [None] * 32


def test_controls_are_sampled_from_user_input(monkeypatch) -> None:
    session = _session(monkeypatch)

    frame_actions = session._actions_for_frames(
        ["backward", "camera_left"],
        None,
        num_frames=3,
    )
    assert frame_actions == [["back", "camera_l"]] * 3
    keyboard, mouse = session._encode_frame_actions(frame_actions)
    assert keyboard.shape == (3, 6)
    assert mouse.shape == (3, 2)
    assert torch.all(keyboard[:, 1] == 1)
    assert torch.all(mouse[:, 1] == -0.1)

    segmented = session._actions_for_frames(
        [],
        [
            {"duration": 1.0, "keys": ["w", "j"]},
            {"duration": 1.0, "keys": ["d"]},
        ],
        num_frames=4,
    )
    assert segmented == [
        ["forward", "camera_l"],
        ["forward", "camera_l"],
        ["right"],
        ["right"],
    ]


def test_size_is_session_input_not_a_baked_fixture() -> None:
    assert MatrixGame3RealtimeSession._normalize_size("704*1280") == (704, 1280)
    assert MatrixGame3RealtimeSession._normalize_size([512, 896]) == (512, 896)


def test_session_accepts_matching_initialized_sequence_parallel_world(monkeypatch) -> None:
    monkeypatch.setattr(MatrixGame3RealtimeSession, "_load_helpers", lambda _self: {})
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    session = MatrixGame3RealtimeSession(_FakeRuntime(sp_size=4), MatrixGame3Operator())

    assert session.distributed is True
    assert session.rank == 2
    assert session.world_size == 4


def test_pipeline_materializes_requested_artifact_path(monkeypatch, tmp_path) -> None:
    class Synthesis:
        def predict(self, **kwargs):
            generated = Path(kwargs["output_dir"]) / f"{kwargs['save_name']}.mp4"
            generated.write_bytes(b"matrix-game-3-video")
            return {
                "video": torch.zeros(1, 3, 1, 2, 2),
                "generated_video_path": str(generated),
                "fps": kwargs["fps"],
            }

    pipeline = MatrixGame3Pipeline(synthesis_model=Synthesis())
    monkeypatch.setattr(
        pipeline,
        "process",
        lambda **_: {
            "image": Image.new("RGB", (2, 2)),
            "keyboard_condition": torch.zeros(1, 6),
            "mouse_condition": torch.zeros(1, 2),
            "num_iterations": 1,
        },
    )
    output_path = tmp_path / "generated" / "matrix-game-3.mp4"
    result = pipeline(
        images=Image.new("RGB", (2, 2)),
        output_dir=str(tmp_path / "scratch"),
        output_path=output_path,
        return_dict=True,
    )

    assert result["artifact_path"] == str(output_path.resolve())
    assert output_path.read_bytes() == b"matrix-game-3-video"


@pytest.mark.parametrize("use_base_model", [False, True])
@pytest.mark.parametrize(("world_size", "rank"), [(1, 0), (2, 0), (2, 1)])
def test_resident_camera_preparation_is_per_chunk_and_cfg_branch(monkeypatch, use_base_model, world_size, rank):
    session = _session(monkeypatch)
    session.world_size = world_size
    session.rank = rank
    prepared = []
    received = []

    class Model(torch.nn.Module):
        def prepare_camera_condition(self, source, *, token_length):
            condition = {"source": source, "token_length": token_length}
            prepared.append(condition)
            return condition

        def forward(self, *, x, plucker_emb, x_memory, **kwargs):
            received.append((plucker_emb, x_memory))
            return torch.zeros_like(x)

    class VAE:
        dtype = torch.bfloat16

        def stream_decode(self, latent, cache, *, first_chunk, **kwargs):
            assert rank == 0
            frames = 57 if first_chunk else 40
            return torch.zeros(1, 3, frames, 2, 2), cache

    session.core.model = Model().eval()
    session.core.vae = VAE()
    session.core.fa_version = 0
    monkeypatch.setattr(session, "_append_actions", lambda *args, **kwargs: None)

    def camera_conditions(**kwargs):
        camera = torch.full((1, 6, 2, 2, 2), float(session._clip_index + 1))
        memory = None if kwargs["first_clip"] else torch.zeros(1, 48, 5, 2, 2)
        return camera, camera + 10, memory, None, None, None, None

    monkeypatch.setattr(session, "_build_camera_conditions", camera_conditions)

    def configure():
        session._configured = True
        session._generator = torch.Generator().manual_seed(42)
        session._image_condition = torch.zeros(1, 48, 1, 2, 2, dtype=session.dtype)
        session._keyboard_history = torch.zeros(1, 200, 6)
        session._mouse_history = torch.zeros(1, 200, 2)
        session._num_inference_steps = 3
        session._max_seq_len = 60
        session._use_base_model = use_base_model

    configure()
    for frames in (57, 40):
        before = len(received)
        result = session.generate(interactions=["forward"])
        if rank == 0:
            assert result["video"].shape[0] == frames
            chunk = result["video_chunk"]
            assert chunk.frames is result["video"]
            assert chunk.layout == "thwc"
            assert chunk.value_range == "0,255"
            assert chunk.frame_count == frames
        else:
            assert result["video"] is None
            assert result["video_chunk"] is None
        chunk_calls = received[before:]
        branches = 2 if use_base_model else 1
        assert len(chunk_calls) == 3 * branches
        for branch in range(branches):
            assert all(call[0] is chunk_calls[branch][0] for call in chunk_calls[branch::branches])
        if use_base_model:
            assert chunk_calls[0][0] is not chunk_calls[1][0]
            assert chunk_calls[1][1] is None
        if frames == 40:
            assert chunk_calls[0][1] is not None

    expected_preparations = 2 * (2 if use_base_model else 1) if world_size == 1 else 0
    assert len(prepared) == expected_preparations
    if prepared:
        assert all(condition["token_length"] == 60 for condition in prepared)
        assert not torch.equal(prepared[0]["source"], prepared[-1]["source"])
    session.reset()
    configure()
    session.generate(interactions=["right"])
    if prepared:
        assert len(prepared) == expected_preparations + (2 if use_base_model else 1)
