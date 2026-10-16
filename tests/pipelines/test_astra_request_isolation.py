# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Astra requests retain independent rollout state while reusing loaded weights."""

from __future__ import annotations

from asyncio import CancelledError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from worldfoundry.core.nn import FlowMatchScheduler
from worldfoundry.operators.astra_operator import AstraOperator
from worldfoundry.pipelines.kling.pipeline_astra import AstraPipeline
from worldfoundry.synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from worldfoundry.synthesis.visual_generation.memory.kling.astra import AstraMemory

pytestmark = pytest.mark.unit


class _Denoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0625), requires_grad=False)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        clean_latents: torch.Tensor,
        context: torch.Tensor,
        cam_emb: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        return sample * 0.125 + self.weight + context + clean_latents.mean() * 0.125 + cam_emb[
            :, 19:27, 11
        ].mean(), None


class _Pipe:
    def __init__(self) -> None:
        self.device = "cpu"
        self.dit = _Denoiser()
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        self.scratch: list[torch.Tensor] = []
        self.vae = SimpleNamespace(model=SimpleNamespace(clear_cache=self.scratch.clear))

    def encode_video(self, frames: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert self.scratch == []
        self.scratch.append(frames)
        return frames

    def decode_video(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert self.scratch == []
        self.scratch.append(latents)
        return latents.repeat_interleave(4, dim=2)[:, :, : 4 * (latents.shape[2] - 1) + 1].clamp(-1, 1)

    def encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        return {"context": torch.tensor(0.125 if prompt == "A" else -0.125)}

    def prepare_extra_input(self, latents: torch.Tensor) -> dict[str, Any]:
        return {}


def _load(monkeypatch: pytest.MonkeyPatch) -> tuple[AstraPipeline, AstraSynthesis, list[str]]:
    synthesis = AstraSynthesis(_Pipe())
    loads: list[str] = []

    def load(config: Any, device: str) -> AstraSynthesis:
        loads.append(config.astra_path)
        return synthesis

    monkeypatch.setattr(AstraSynthesis, "from_pretrained", load)
    pipeline = AstraPipeline.from_pretrained(
        "local-astra", {"wan_model_path": "local-wan"}, "cpu", num_inference_steps=3
    )

    def perception(**kwargs: Any) -> torch.Tensor:
        return torch.full((3, 1, 2, 2), 0.25 if kwargs["condition_image"] == "A.png" else -0.25)

    monkeypatch.setattr(AstraOperator, "process_perception", lambda self, **kwargs: perception(**kwargs))
    return pipeline, synthesis, loads


def _call(pipeline: AstraPipeline, name: str = "A", **kwargs: Any) -> np.ndarray:
    options: dict[str, Any] = {"num_frames": 65, "num_inference_steps": 3, "seed": 7, "return_dict": True, **kwargs}
    result = pipeline(
        image_path=f"{name}.png", prompt=name, interactions=["forward" if name == "A" else "backward"], **options
    )
    assert isinstance(result, dict)
    assert result["video_chunk"].frames is result["frames"]
    return np.stack([np.asarray(frame) for frame in result["frames"]])


def test_two_sessions_reuse_one_loaded_synthesis_with_independent_a_b_a_state(monkeypatch: pytest.MonkeyPatch) -> None:
    first, synthesis, loads = _load(monkeypatch)
    second = AstraPipeline(AstraOperator("cpu"), synthesis, AstraMemory(capacity=100), replace(first.config))
    first.operator.current_interaction = ["camera_l"]
    second.operator.current_interaction = ["camera_r"]
    weights = tuple(synthesis.pipe.dit.parameters())
    originals = [weight.clone() for weight in weights]
    rng = torch.random.get_rng_state().clone()
    a = _call(first)
    a_memory = first.memory
    assert a_memory.total_latent_frames == 17
    b = _call(second, "B", seed=11, num_inference_steps=7, num_frames=97)
    assert not np.array_equal(a[:65], b[:65])
    assert second.memory is not first.memory and first.memory is a_memory
    assert second.memory.total_latent_frames == 25
    np.testing.assert_array_equal(_call(first), a)
    assert first.synthesis is second.synthesis is synthesis and loads == ["local-astra"]
    assert first.memory is not a_memory and first.memory.total_latent_frames == 17
    assert first.operator.current_interaction == ["camera_l"]
    assert second.operator.current_interaction == ["camera_r"]
    assert first.config.num_inference_steps == second.config.num_inference_steps == 3
    assert synthesis.pipe.scratch == []
    for current, original, owned in zip(synthesis.pipe.dit.parameters(), originals, weights):
        assert current is owned
        torch.testing.assert_close(current, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    second.memory.reset()
    assert second.memory.history_latents is None and second.memory.total_latent_frames == 0
    assert first.memory.total_latent_frames == 17
    np.testing.assert_array_equal(_call(second, "B", seed=11, num_inference_steps=7, num_frames=97), b)


def test_interleaved_calls_on_one_pipeline_keep_active_memory_camera_rng_and_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, synthesis, _ = _load(monkeypatch)
    a = _call(pipeline)
    b = _call(pipeline, "B", seed=11, num_inference_steps=7, num_frames=97)
    forward = synthesis.pipe.dit.forward
    results: list[np.ndarray] = []
    histories: list[AstraMemory] = []

    def interleave(sample: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, None]:
        if not results:
            results.append(np.empty(0))
            results[0] = _call(pipeline, "B", seed=11, num_inference_steps=7, num_frames=97)
            histories.append(pipeline.memory)
        return forward(sample, **kwargs)

    monkeypatch.setattr(synthesis.pipe.dit, "forward", interleave)
    np.testing.assert_array_equal(_call(pipeline), a)
    np.testing.assert_array_equal(results[0], b)
    assert histories[0] is not pipeline.memory and histories[0].total_latent_frames == 25
    assert pipeline.memory.total_latent_frames == 17
    assert pipeline.operator.current_interaction == [] and synthesis.pipe.scratch == []


@pytest.mark.parametrize("stage", ["encode", "predict", "decode", "export"])
@pytest.mark.parametrize("error", [RuntimeError, CancelledError])
def test_failed_or_unwound_request_preserves_completed_state_and_reuses_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, error: type[BaseException]
) -> None:
    pipeline, synthesis, loads = _load(monkeypatch)
    expected = _call(pipeline)
    memory = pipeline.memory
    history = memory.history_latents
    assert history is not None
    original = history.clone()
    timesteps, sigmas = synthesis.pipe.scheduler.timesteps, synthesis.pipe.scheduler.sigmas
    weight = synthesis.pipe.dit.weight
    rng = torch.random.get_rng_state().clone()

    def fail(*args: Any, **kwargs: Any) -> Any:
        if stage in {"encode", "decode"}:
            synthesis.pipe.scratch.append(torch.empty(0))
        raise error("request stopped")

    with monkeypatch.context() as patch:
        if stage == "encode":
            patch.setattr(synthesis.pipe, "encode_video", fail)
        elif stage == "predict":
            patch.setattr(synthesis.pipe.dit, "forward", fail)
        elif stage == "decode":
            patch.setattr(synthesis.pipe, "decode_video", fail)
        else:
            patch.setattr("worldfoundry.core.io.write_video", fail)
        with pytest.raises(error, match="request stopped"):
            _call(pipeline, "B", seed=11, num_inference_steps=7, output_path=tmp_path / "result.mp4")
    assert pipeline.memory is memory and memory.history_latents is history
    assert memory.total_latent_frames == 17 and synthesis.pipe.scratch == []
    assert not synthesis._vae_lock.locked()
    assert pipeline.operator.current_interaction == []
    assert synthesis.pipe.scheduler.timesteps is timesteps and synthesis.pipe.scheduler.sigmas is sigmas
    assert synthesis.pipe.dit.weight is weight and loads == ["local-astra"]
    torch.testing.assert_close(history, original, rtol=0, atol=0)
    np.testing.assert_array_equal(_call(pipeline), expected)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [49, 100])
def test_long_direction_schedule_advances_camera_after_window_eviction(
    monkeypatch: pytest.MonkeyPatch, capacity: int
) -> None:
    pipeline, synthesis, _ = _load(monkeypatch)
    pipeline.memory = AstraMemory(capacity=capacity)
    packs: list[dict[str, Any]] = []
    predict = synthesis.predict

    def record(pack: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        packs.append(pack)
        return predict(pack, *args, **kwargs)

    monkeypatch.setattr(synthesis, "predict", record)
    directions = ["forward", "backward", "camera_l", "camera_r", "s_curve"]
    expected_camera = pipeline.process("A.png", {"prompt": "A", "direction": directions})["camera_embedding_full"]
    pipeline(image_path="A.png", prompt="A", interactions=directions, seed=7)
    assert len(packs) == 15 and pipeline.memory.total_latent_frames == 121
    assert pipeline.memory.history_latents is not None and pipeline.memory.history_latents.shape[1] == capacity
    assert packs[13]["current_length"] == 105 and packs[13]["next_length"] == 113
    torch.testing.assert_close(packs[13]["camera_embedding"][19:27, :-1], expected_camera[105:113, :-1], rtol=0, atol=0)
    torch.testing.assert_close(packs[14]["camera_embedding"][19:27, :-1], expected_camera[113:121, :-1], rtol=0, atol=0)
    assert not torch.equal(expected_camera[100:108, :-1], expected_camera[105:113, :-1])


def test_explicit_operator_direction_retains_legacy_interaction_history() -> None:
    operator = AstraOperator("cpu")
    operator.current_interaction = ["backward"]
    explicit = operator.process_interaction(direction="forward")
    legacy = operator.process_interaction()
    assert explicit[1, 11].item() < 0 < legacy[1, 11].item()
    assert operator.current_interaction == ["backward"]
    torch.testing.assert_close(
        operator.process_interaction("sekai", 0, 1, 8, False, None, None), legacy, rtol=0, atol=0
    )


@pytest.mark.parametrize("use_torch", [False, True])
def test_relative_pose_is_callable_from_class_and_operator_without_mutation(use_torch: bool) -> None:
    previous = np.eye(4, dtype=np.float32)
    previous[0, 3] = 2
    current = np.eye(4, dtype=np.float32)
    current[2, 3] = 3
    expected = current @ np.linalg.inv(previous)
    before_previous, before_current = previous.copy(), current.copy()
    for compute in (AstraOperator.compute_relative_pose, AstraOperator("cpu").compute_relative_pose):
        actual = compute(previous, current, use_torch=use_torch)
        np.testing.assert_array_equal(np.asarray(actual), expected)
    np.testing.assert_array_equal(previous, before_previous)
    np.testing.assert_array_equal(current, before_current)


@pytest.mark.parametrize("modality", ["sekai", "openx"])
def test_real_extrinsics_use_the_operator_relative_pose_contract(modality: str) -> None:
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 125, axis=0)
    extrinsics[:, 0, 3] = np.arange(125, dtype=np.float32) / 4
    original = extrinsics.copy()
    camera = AstraOperator("cpu").process_interaction(
        modality_type=modality, use_real_poses=True, encoded_data={"cam_emb": {"extrinsic": extrinsics}}
    )
    assert camera.shape == (30, 13) and camera.dtype == torch.bfloat16
    torch.testing.assert_close(camera[:, 3], torch.ones(30, dtype=torch.bfloat16), rtol=0, atol=0)
    np.testing.assert_array_equal(extrinsics, original)
