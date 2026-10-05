import numpy as np
import random
import torch
from PIL import Image

from worldfoundry.operators.matrix_game_2_operator import MatrixGame2Operator
from worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
import worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_synthesis as mg2_synthesis
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import (
    MatrixGame2Synthesis,
)


class _DummyPipeline:
    def inference(self, noise, conditional_dict, return_latents, mode, profile):
        del conditional_dict, return_latents, mode, profile
        # Simulate the extra random draws that happen during MG2 denoising.
        return [torch.randn((1, 2, 3, 2, 2), device=noise.device, dtype=noise.dtype)]


class _DummyMatrixGame2Pipe:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(kwargs["num_frames"])]


def test_matrix_game_2_from_pretrained_unpacks_typed_cli_mapping(monkeypatch):
    captured = {}

    def fake_from_pretrained(cls, **kwargs):
        del cls
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(MatrixGame2Synthesis, "from_pretrained", classmethod(fake_from_pretrained))

    pipeline = MatrixGame2Pipeline.from_pretrained(
        model_path={
            "model_path": "/checkpoints/Skywork--Matrix-Game-2.0",
            "mode": "gta_drive",
            "checkpoint_path": "custom.safetensors",
            "pipeline_binding": "matrix-game-2",
        },
        device="cuda:1",
    )

    assert captured["pretrained_model_path"] == "/checkpoints/Skywork--Matrix-Game-2.0"
    assert captured["mode"] == "gta_drive"
    assert captured["device"] == "cuda:1"
    assert captured["checkpoint_path"] == "custom.safetensors"
    assert captured["pipeline_binding"] == "matrix-game-2"
    assert pipeline.synthesis_model is not None


def test_matrix_game_2_predict_seed_controls_full_rng_chain():
    original_process_video = mg2_synthesis.process_video
    mg2_synthesis.process_video = lambda video, *args, **kwargs: video
    try:
        model = MatrixGame2Synthesis(
            pipeline=_DummyPipeline(),
            vae=None,
            weight_dtype=torch.float32,
            mode="templerun",
            device="cpu",
        )
        cond_concat = torch.zeros((1, 20, 2, 2, 2), dtype=torch.float32)
        visual_context = torch.zeros((1, 4, 8), dtype=torch.float32)
        operator_condition = {
            "keyboard_condition": torch.zeros((5, 7), dtype=torch.float32),
        }

        video_a = model.predict(
            cond_concat=cond_concat,
            visual_context=visual_context,
            operator_condition=operator_condition,
            num_output_frames=2,
            seed=123,
            operation_visualization=False,
        )
        video_b = model.predict(
            cond_concat=cond_concat,
            visual_context=visual_context,
            operator_condition=operator_condition,
            num_output_frames=2,
            seed=123,
            operation_visualization=False,
        )
        video_c = model.predict(
            cond_concat=cond_concat,
            visual_context=visual_context,
            operator_condition=operator_condition,
            num_output_frames=2,
            seed=456,
            operation_visualization=False,
        )

        assert np.array_equal(video_a, video_b)
        assert not np.array_equal(video_a, video_c)
    finally:
        mg2_synthesis.process_video = original_process_video


def test_matrix_game_2_infer_contract_passes_smoke_parameters(monkeypatch, tmp_path):
    pipeline = MatrixGame2Pipeline(
        operators=None,
        synthesis_model=None,
        memory_module=None,
        device="cpu",
        weight_dtype=torch.float32,
    )
    process_kwargs = {}
    pipeline.process = lambda **kwargs: process_kwargs.update(kwargs) or {
        "cond_concat": "cond",
        "visual_context": "ctx",
        "operator_condition": "ops",
    }

    class _Synthesis:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(kwargs["num_output_frames"])]

    synthesis = _Synthesis()
    pipeline.synthesis_model = synthesis
    exported = {}

    def fake_write_video(video, output_path, fps):
        exported["frames"] = len(video)
        exported["output_path"] = str(output_path)
        exported["fps"] = fps

    monkeypatch.setattr("worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2.write_video", fake_write_video)
    output_path = tmp_path / "matrix_game2_worldfoundry_universal_seed42_15f.mp4"

    result = pipeline(
        images=Image.new("RGB", (8, 8), "green"),
        interactions=["forward", "camera_r", "forward_right", "camera_l"],
        output_path=output_path,
        num_frames=15,
        fps=12,
        seed=42,
        visualize_ops=False,
    )

    assert len(result) == 15
    assert process_kwargs["num_output_frames"] == 15
    assert process_kwargs["interaction_signal"] == ["forward", "camera_r", "forward_right", "camera_l"]
    assert synthesis.calls[0]["seed"] == 42
    assert synthesis.calls[0]["operation_visualization"] is False
    assert exported == {"frames": 15, "output_path": str(output_path), "fps": 12}


def test_matrix_game_2_official_bench_actions_match_expected_condition_shape():
    operator = MatrixGame2Operator(mode="universal")
    random.seed(42)
    official = operator.process_official_bench_actions(num_frames=57)

    operator.get_interaction(["forward", "camera_r", "forward_right", "camera_l"])
    manual = operator.process_interaction(num_frames=57)
    operator.delete_last_interaction()

    assert official["keyboard_condition"].shape == (57, 4)
    assert official["mouse_condition"].shape == (57, 2)
    assert not torch.equal(official["keyboard_condition"], manual["keyboard_condition"])
    assert not torch.equal(official["mouse_condition"], manual["mouse_condition"])


def test_matrix_game_2_return_dict_writes_generation_artifact(monkeypatch, tmp_path):
    pipeline = MatrixGame2Pipeline(
        operators=None,
        synthesis_model=None,
        memory_module=None,
        device="cpu",
        weight_dtype=torch.float32,
    )
    pipeline.process = lambda **kwargs: {
        "cond_concat": "cond",
        "visual_context": "ctx",
        "operator_condition": "ops",
    }

    class _Synthesis:
        def predict(self, **kwargs):
            return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(kwargs["num_output_frames"])]

    exported = {}

    def fake_write_video(video, output_path, fps):
        exported["frames"] = len(video)
        exported["output_path"] = str(output_path)
        exported["fps"] = fps

    monkeypatch.setattr("worldfoundry.pipelines.matrix_game.pipeline_matrix_game_2.write_video", fake_write_video)
    pipeline.synthesis_model = _Synthesis()
    output_path = tmp_path / "generated" / "sample.mp4"

    result = pipeline(
        images=Image.new("RGB", (8, 8), "blue"),
        interactions=["forward"],
        num_frames=15,
        output_path=output_path,
        fps=12,
        visualize_ops=False,
        seed=42,
        return_dict=True,
    )

    assert result["status"] == "ok"
    assert result["artifact_kind"] == "generated_video"
    assert result["artifact_path"] == str(output_path)
    assert result["num_output_frames"] == 15
    assert result["fps"] == 12
    assert exported == {"frames": 15, "output_path": str(output_path), "fps": 12}
