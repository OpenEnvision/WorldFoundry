from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

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
            generated = Path(kwargs["output_dir"]) / f'{kwargs["save_name"]}.mp4'
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
