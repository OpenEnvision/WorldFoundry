# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""DreamDojo native component, public-loader, and generated-pixel contracts."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput, DiffusionRequest
from worldfoundry.base_models.diffusion_model.models.denoisers.dreamdojo import (
    DreamDojoDenoiser,
    convert_dreamdojo_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.encoders.dreamdojo import DreamDojoConditioner, robot_actions
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.action import ActionChunkCosmosTransformer
from worldfoundry.pipelines.component_pipelines import DreamDojoPipeline
from worldfoundry.synthesis.action_generation.dreamdojo.worldfoundry_runtime import DreamDojoRuntime

pytestmark = pytest.mark.unit


def tiny_model() -> ActionChunkCosmosTransformer:
    torch.manual_seed(7)
    return ActionChunkCosmosTransformer(
        in_channels=3,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        text_in_channels=8,
        text_embed_dim=8,
        adaln_lora_dim=8,
    ).eval()


def test_action_chunks_change_future_frames_and_leave_first_embedding_unconditioned() -> None:
    model = tiny_model()
    latent = torch.randn(1, 2, 4, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, 4)
    context = torch.randn(1, 3, 8)
    actions = torch.zeros(1, 12, 384)
    with torch.inference_mode():
        first = model(latent, torch.tensor([[0.5]]), context, action=actions, condition_mask=mask)
        changed = model(latent, torch.tensor([[0.5]]), context, action=actions + 1, condition_mask=mask)
        repeated = model(latent, torch.tensor([[0.5]]), context, action=actions, condition_mask=mask)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    assert not torch.equal(first[:, :, 1:], changed[:, :, 1:])
    with pytest.raises(ValueError, match="actions must have shape"):
        model(latent, torch.tensor([[0.5]]), context, action=actions[:, :-1], condition_mask=mask)


def test_denoiser_preserves_observed_frame_velocity() -> None:
    denoiser = DreamDojoDenoiser(tiny_model())
    latent = torch.randn(1, 2, 4, 4, 4)
    condition = torch.randn_like(latent)
    noise = torch.randn_like(latent)
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[:, :, 0] = 1
    result = denoiser(
        DenoiserInput(
            latents=latent,
            timestep=torch.tensor(500.0),
            next_timestep=torch.tensor(0.0),
            step_index=0,
            total_steps=1,
            conditioning={
                "context": torch.randn(1, 3, 8),
                "actions": torch.zeros(1, 12, 384),
                "condition_latents": condition,
                "condition_mask": mask,
                "initial_noise": noise,
            },
        )
    )
    torch.testing.assert_close(result.sample[:, :, 0], (noise - condition)[:, :, 0], rtol=0, atol=0)


def test_ema_mapping_keeps_action_weights() -> None:
    state = convert_dreamdojo_state_dict(
        {
            "net_ema.action_embedder_B_D.fc1.weight": torch.ones(2, 3),
            "net.action_embedder_B_D.fc1.weight": torch.zeros(2, 3),
        }
    )
    weight = state["action_embedder_B_D.fc1.weight"]
    assert isinstance(weight, torch.Tensor)
    assert torch.equal(weight, torch.ones(2, 3))


def test_conditioner_preserves_actions_and_official_embedding_contract() -> None:
    embeddings = torch.ones(1, 2, 100352)
    request = DiffusionRequest(prompt="robot", num_frames=13, inputs={"actions": torch.zeros(12, 384)})
    conditioning = DreamDojoConditioner(embeddings=embeddings).encode(
        request, device=torch.device("cpu"), dtype=torch.float32
    )
    context = conditioning.positive["context"]
    negative = conditioning.negative["context"]
    actions = conditioning.shared["actions"]
    assert isinstance(context, torch.Tensor) and isinstance(negative, torch.Tensor)
    assert isinstance(actions, torch.Tensor)
    assert context.shape == (1, 2, 100352)
    assert torch.count_nonzero(negative) == 0
    assert actions.shape == (1, 12, 384)


@pytest.mark.parametrize("shape", [(12, 58), (1, 12, 383), (2, 12, 384)])
def test_reject_wrong_action_contract(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="normalized actions"):
        robot_actions(torch.zeros(shape))


def test_native_recipe_resolves_with_shared_cosmos_roles() -> None:
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry

    recipe = default_native_diffusion_registry().resolve("nvidia/DreamDojo")
    assert recipe.model_id == "dreamdojo"
    assert recipe.execution.strategy == "standard"
    assert {str(component.key) for component in recipe.components} == {
        "denoiser:main",
        "conditioner:main",
        "latent_initializer:main",
        "scheduler:main",
    }


def test_public_loader_executes_generated_pixels_and_chunk_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []
    saves = []

    def native(request: DiffusionRequest) -> SimpleNamespace:
        requests.append(request)
        return SimpleNamespace(sample=torch.full((1, 3, 13, 16, 16), 0.25))

    def save(pixels: torch.Tensor, path: str, *, fps: int) -> None:
        saves.append((pixels.clone(), fps))
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("worldfoundry.core.media.codecs.video.save_image_or_video_tensor", save)
    pipe = DreamDojoPipeline.from_pretrained({"native_pipeline": native}, device="cpu")
    result = pipe(
        images=Image.new("RGB", (16, 16)),
        interactions=torch.zeros(24, 384).tolist(),
        output_path=tmp_path / "robot.mp4",
        height=16,
        width=16,
        seed=42,
        return_dict=True,
    )
    assert len(requests) == 2
    assert [request.sampling.seed for request in requests] == [42, 43]
    assert saves[0][0].shape == (1, 3, 25, 16, 16)
    assert torch.equal(saves[0][0], torch.full_like(saves[0][0], 0.25))
    assert requests[1].inputs["image"].getpixel((0, 0)) == (159, 159, 159)
    assert Path(result["artifact_path"]).is_file()


def test_public_action_file_reaches_native_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    action_file = tmp_path / "actions.pt"
    torch.save(torch.zeros(12, 384), action_file)
    requests = []

    def native(request: DiffusionRequest) -> SimpleNamespace:
        requests.append(request)
        return SimpleNamespace(sample=torch.zeros(1, 3, 13, 16, 16))

    monkeypatch.setattr(
        "worldfoundry.core.media.codecs.video.save_image_or_video_tensor",
        lambda pixels, path, fps: Path(path).write_bytes(b"video"),
    )
    pipe = DreamDojoPipeline.from_pretrained({"native_pipeline": native}, device="cpu")
    pipe(
        images=Image.new("RGB", (16, 16)),
        action_file=action_file,
        height=16,
        width=16,
        output_path=tmp_path / "video.mp4",
    )
    assert requests[0].inputs["actions"].shape == (1, 12, 384)


@pytest.mark.parametrize(
    "options",
    [
        {"height": 16.5},
        {"width": 0},
        {"num_inference_steps": True},
        {"num_inference_steps": 1.5},
        {"num_inference_steps": 0},
        {"fps": float("nan")},
        {"guidance_scale": float("inf")},
    ],
)
def test_bad_sampling_options_fail_before_loading(options: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DreamDojoRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_load_pipeline", lambda: pytest.fail("loaded weights before validation"))
    with pytest.raises(ValueError):
        runtime.predict(images=Image.new("RGB", (16, 16)), actions=torch.zeros(12, 384), **options)


def test_bad_inputs_fail_before_checkpoint_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DreamDojoRuntime(device="cpu")
    monkeypatch.setattr(runtime, "_load_pipeline", lambda: pytest.fail("loaded weights before validation"))
    with pytest.raises(ValueError, match="initial image"):
        runtime.predict(actions=torch.zeros(12, 384))
    with pytest.raises(ValueError, match="complete chunks"):
        runtime.predict(images=Image.new("RGB", (16, 16)), actions=torch.zeros(11, 384))
