# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free real-DiT student math, action alignment, and public lifecycle contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from einops import repeat
from PIL import Image

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos2p5.student import DreamDojoStudentCodec
from worldfoundry.base_models.diffusion_model.models.denoisers.dreamdojo_student import (
    DreamDojoStudentDenoiser,
    trigflow_scaling,
)
from worldfoundry.base_models.diffusion_model.models.encoders.dreamdojo import DreamDojoConditioner
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.action_causal import (
    CausalActionCosmosTransformer,
)
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.runners.causal_trigflow import CausalTrigFlowRunner
from worldfoundry.base_models.diffusion_model.schedulers.trigflow import SAMPLING_TIMES, TrigFlowScheduler
from worldfoundry.core.attention.cache.kvcache import CompactingKVCache
from worldfoundry.core.execution.realtime.contracts import InteractiveWorldPipeline
from worldfoundry.pipelines.dreamdojo.pipeline_dreamdojo_student import DreamDojoStudentPipeline
from worldfoundry.synthesis.action_generation.dreamdojo.realtime import DreamDojoStudentSession
from worldfoundry.synthesis.action_generation.dreamdojo.worldfoundry_runtime import DreamDojoRuntime

pytestmark = pytest.mark.unit


def tiny_student(*, dtype: torch.dtype = torch.float32, text_dim: int = 8) -> CausalActionCosmosTransformer:
    """A real two-block action DiT with checkpoint-identical parameter names."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return (
            CausalActionCosmosTransformer(
                in_channels=3,
                out_channels=2,
                num_attention_heads=2,
                attention_head_dim=16,
                num_layers=2,
                text_in_channels=text_dim,
                text_embed_dim=8,
                adaln_lora_dim=8,
            )
            .to(dtype=dtype)
            .eval()
        )


def dense_causal_forward(
    model: Any, latent: torch.Tensor, context: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Independent dense temporal mask reference, using the same model weights."""
    batch, _, frames, height, width = latent.shape
    tokens, rotary, grid = model._prepare_tokens(
        latent,
        embedder=model.patch_embed,
        fps=None,
        condition_mask=latent.new_zeros((batch, 1, frames, height, width)),
        padding_mask=None,
    )
    projected, temb = model.time_embed(tokens, latent.new_zeros((batch * frames,)))
    grouped = actions.reshape(batch, frames - 1, -1)
    projected = projected.reshape(batch, frames, -1)
    temb = temb.reshape(batch, frames, -1)
    projected[:, 1:] += model.action_embedder_B_D(grouped)
    temb[:, 1:] += model.action_embedder_B_3D(grouped)
    projected = model.time_norm(projected)
    projected, temb = (repeat(value, "b t c -> b (t h w) c", h=grid[1], w=grid[2]) for value in (projected, temb))
    positions = torch.arange(tokens.shape[1], device=latent.device) // (grid[1] * grid[2])
    allowed = positions[:, None] >= positions[None, :]
    context = model.text_embed(context)
    for block in model.transformer_blocks:
        normalized, gate = block.norm1(tokens, projected, temb)
        tokens = tokens + gate * block.attn1(normalized, rotary_emb=rotary, attention_mask=allowed)
        normalized, gate = block.norm2(tokens, projected, temb)
        tokens = tokens + gate * block.attn2(normalized, encoder_hidden_states=context)
        normalized, gate = block.norm3(tokens, projected, temb)
        tokens = tokens + gate * block.ff(normalized)
    return model._unpatchify(model.proj_out(model.norm_out(tokens, projected, temb)), grid)


def test_cached_causal_forward_matches_dense_temporal_reference() -> None:
    model = tiny_student()
    latent, context, actions = torch.randn(1, 2, 4, 4, 4), torch.randn(1, 3, 8), torch.randn(1, 12, 384)
    caches: dict[int, CompactingKVCache] = {}
    with torch.inference_mode():
        expected = dense_causal_forward(model, latent, context, actions)
        frames = [
            model.forward_frame(
                latent[:, :, index : index + 1],
                torch.zeros(1),
                context,
                action=None if index == 0 else actions[:, (index - 1) * 4 : index * 4],
                frame_index=index,
                kv_cache=caches,
                store_kv=True,
                condition_mask=torch.zeros(1, 1, 1, 4, 4),
            )
            for index in range(4)
        ]
    torch.testing.assert_close(torch.cat(frames, dim=2), expected, rtol=2e-5, atol=2e-6)
    assert all(len(cache.blocks) == 3 for cache in caches.values())


def test_sampling_reads_do_not_commit_noise_or_exchange_session_history() -> None:
    model = tiny_student()
    latent, context = torch.randn(1, 2, 1, 4, 4), torch.randn(1, 3, 8)
    first: dict[int, CompactingKVCache] = {}
    second: dict[int, CompactingKVCache] = {}
    with torch.inference_mode():
        for caches, observation in ((first, latent), (second, latent + 1)):
            model.forward_frame(
                observation,
                torch.zeros(1),
                context,
                action=None,
                frame_index=0,
                kv_cache=caches,
                store_kv=True,
                condition_mask=torch.zeros(1, 1, 1, 4, 4),
            )

        def sample(caches: dict[int, CompactingKVCache], noise: torch.Tensor) -> torch.Tensor:
            return model.forward_frame(
                noise,
                torch.ones(1),
                context,
                action=torch.zeros(1, 4, 384),
                frame_index=1,
                kv_cache=caches,
                condition_mask=torch.zeros(1, 1, 1, 4, 4),
            )

        a = sample(first, latent)
        b = sample(second, latent)
        sample(first, latent + 2)
        repeat_a = sample(first, latent)
    torch.testing.assert_close(a, repeat_a, atol=0, rtol=0)
    assert not torch.equal(a, b)
    assert all(len(cache.blocks) == 1 for cache in (*first.values(), *second.values()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_trigflow_scaling_matches_released_float64_formula(dtype: torch.dtype) -> None:
    time = torch.tensor(SAMPLING_TIMES, dtype=dtype)
    skip, out, scale, noise_time = trigflow_scaling(time)
    denominator = time.double().cos() + time.double().sin()
    torch.testing.assert_close(skip, denominator.reciprocal().to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(out, (-time.double().sin() / denominator).to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(scale, skip, rtol=0, atol=0)
    torch.testing.assert_close(noise_time, (time.double().sin() / denominator).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_selected_schedule_renoises_with_original_noise(steps: int) -> None:
    scheduler = TrigFlowScheduler()
    sampling = SamplingConfig(num_inference_steps=steps, guidance_scale=1)
    schedule = scheduler.schedule(sampling, device=torch.device("cpu"))
    assert len(schedule) == steps
    noise, prediction = torch.randn(1, 2, 1, 4, 4), torch.randn(1, 2, 1, 4, 4)
    for index, step in enumerate(schedule):
        next_time = torch.tensor(SAMPLING_TIMES[index + 1] if index + 1 < steps else 0, dtype=torch.bfloat16)
        result = scheduler.step(prediction, step, noise, generator=torch.Generator())
        torch.testing.assert_close(result, next_time.cos() * prediction + next_time.sin() * noise, rtol=0, atol=0)
    torch.testing.assert_close(result, prediction, rtol=0, atol=0)


class TinyVideoVAE:
    """Deterministic toy codec; this fixture is not Wan checkpoint evidence."""

    def encode(self, pixels: list[torch.Tensor], device: torch.device, **_: Any) -> torch.Tensor:
        video = torch.stack(pixels).to(device)
        return (
            functional.avg_pool3d(video.mean(1, keepdim=True), (1, 4, 4))[:, :, ::4]
            .expand(-1, 2, -1, -1, -1)
            .contiguous()
        )

    def decode(self, latent: torch.Tensor, device: torch.device, **_: Any) -> torch.Tensor:
        return functional.interpolate(
            latent.mean(1, keepdim=True).expand(-1, 3, -1, -1, -1), (13, 16, 16), mode="trilinear", align_corners=True
        ).to(device)


@pytest.fixture
def native_student() -> NativeDiffusionPipeline:
    """Real tiny DiT/schedule/initializer composition with an explicit toy VAE."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        embeddings = torch.randn(1, 2, 100352)
    codec = DreamDojoStudentCodec(TinyVideoVAE(), device=torch.device("cpu"), dtype=torch.float32)  # type: ignore[arg-type]
    return NativeDiffusionPipeline(
        CausalTrigFlowRunner(
            model_id="dreamdojo-student",
            device=torch.device("cpu"),
            dtype=torch.float32,
            components=RunnerComponents(
                denoiser=DreamDojoStudentDenoiser(tiny_student(text_dim=100352)),
                conditioner=DreamDojoConditioner(embeddings=embeddings),
                latent_initializer=codec,
                decoder=codec,
                scheduler=TrigFlowScheduler(),
            ),
        )
    )


def test_codec_zero_pads_history_and_preserves_numpy_noise_seed(native_student: NativeDiffusionPipeline) -> None:
    import numpy as np

    codec = native_student.components.latent_initializer
    frames = torch.full((1, 3, 9, 16, 16), 255, dtype=torch.uint8)
    result = codec.initialize(
        DiffusionRequest(prompt="", height=16, width=16, num_frames=13, inputs={"conditioning_frames": frames}),
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result.conditioning["start_index"] == 3
    observation = result.conditioning["condition_latents"]
    assert torch.equal(observation[:, :, :3], torch.ones_like(observation[:, :, :3]))
    assert torch.equal(observation[:, :, 3:], -torch.ones_like(observation[:, :, 3:]))
    expected = torch.from_numpy(np.random.RandomState(42).standard_normal(result.latents.shape).astype(np.float32))
    torch.testing.assert_close(result.latents, expected, atol=0, rtol=0)


def test_public_student_streaming_cadence_reset_and_interleaved_sessions(
    native_student: NativeDiffusionPipeline,
) -> None:
    pipe = DreamDojoStudentPipeline.from_pretrained({"native_pipeline": native_student}, device="cpu")
    other = DreamDojoStudentPipeline.from_pretrained({"native_pipeline": native_student}, device="cpu")
    assert isinstance(pipe, InteractiveWorldPipeline)
    assert pipe.synthesis_model.runtime.variant == "student"
    spec = pipe.prepare_realtime()["realtime_spec"]
    assert (spec["fps"], spec["first_chunk_frames"], spec["steady_chunk_frames"]) == (10, 13, 4)
    assert (spec["first_input_frames"], spec["steady_input_frames"]) == (12, 4)
    image = Image.new("RGB", (16, 16), (90, 20, 10))
    options = {"height": 16, "width": 16, "seed": 42, "num_inference_steps": 2}
    rng = torch.random.get_rng_state()
    pipe.configure_realtime(image, **options)
    a = pipe.stream_realtime(actions=torch.zeros(12, 384))["video_chunk"].frames
    other.configure_realtime(Image.new("RGB", (16, 16), (10, 220, 70)), **options)
    other.stream_realtime(actions=torch.ones(12, 384))
    assert pipe.realtime_next_output_frames() == 4
    continued = pipe.stream_realtime(actions=torch.zeros(4, 384))["video_chunk"].frames
    assert (a.shape[2], continued.shape[2]) == (13, 4)
    assert torch.equal(rng, torch.random.get_rng_state())
    pipe.reset_realtime()
    assert pipe.synthesis_model.runtime.native_pipeline is native_student
    with pytest.raises(RuntimeError, match="configured"):
        pipe.stream_realtime(actions=torch.zeros(12, 384))
    pipe.configure_realtime(image, **options)
    repeated = pipe.stream_realtime(actions=torch.zeros(12, 384))["video_chunk"].frames
    torch.testing.assert_close(a, repeated, rtol=0, atol=0)
    torch.testing.assert_close(
        continued, pipe.stream_realtime(actions=torch.zeros(4, 384))["video_chunk"].frames, rtol=0, atol=0
    )


def test_new_actions_condition_generated_frame_not_history(native_student: NativeDiffusionPipeline) -> None:
    session = DreamDojoStudentSession(native_student)
    observation = Image.new("RGB", (16, 16), (50, 70, 80))
    results = []
    for value in (0.0, 0.5, 0.0):
        session.configure(observation, height=16, width=16, num_inference_steps=1)
        session.generate(torch.zeros(12, 384))
        results.append(session.generate(torch.full((4, 384), value))["video_chunk"].frames)
    torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)
    assert not torch.equal(results[0], results[1])
    assert session.condition_frames.shape == (1, 3, 9, 16, 16)
    assert session.action_history.shape == (1, 8, 384)


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_failed_decode_clears_rollout_and_kv_then_allows_reuse(
    native_student: NativeDiffusionPipeline,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    session = DreamDojoStudentSession(native_student)
    session.configure(Image.new("RGB", (16, 16)), height=16, width=16)
    saved_caches = []
    model = native_student.components.denoiser.model
    original = model.forward_frame

    def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
        saved_caches.append(kwargs["kv_cache"])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_frame", capture)
    decoder = native_student.components.decoder
    decode = decoder.decode

    def fail(*_: Any) -> torch.Tensor:
        raise failure("decode failed")

    monkeypatch.setattr(decoder, "decode", fail)
    with pytest.raises(failure, match="decode failed"):
        session.generate(torch.zeros(12, 384))
    assert session.condition_frames is None and session.text_conditioning is None
    assert saved_caches and all(not caches for caches in saved_caches)
    monkeypatch.setattr(decoder, "decode", decode)
    session.configure(Image.new("RGB", (16, 16)), height=16, width=16)
    assert session.generate(torch.zeros(12, 384))["status"] == "success"


def test_public_generator_emits_native_chunks_and_close_preserves_resident_rollout(
    native_student: NativeDiffusionPipeline,
) -> None:
    pipe = DreamDojoStudentPipeline.from_pretrained({"native_pipeline": native_student}, device="cpu")
    image = Image.new("RGB", (16, 16))
    pipe.configure_realtime(image, height=16, width=16)
    pipe.stream_realtime(actions=torch.zeros(12, 384))
    resident = pipe.synthesis_model.runtime._realtime_session
    frames = resident.condition_frames.clone()
    stream = pipe.stream(images=image, interactions=torch.zeros(20, 384), height=16, width=16, num_inference_steps=1)
    first = next(stream)
    assert first["video_chunk"].frames.shape[2] == 13
    assert next(stream)["video_chunk"].frames.shape[2] == 4
    stream.close()
    assert resident.chunk_index == 1
    torch.testing.assert_close(resident.condition_frames, frames, rtol=0, atol=0)
    packets = list(
        pipe.stream(images=image, interactions=torch.zeros(20, 384), height=16, width=16, num_inference_steps=1)
    )
    assert [packet["num_frames"] for packet in packets] == [13, 4, 4]
    torch.testing.assert_close(first["video_chunk"].frames, packets[0]["video_chunk"].frames, rtol=0, atol=0)


def test_bad_action_packet_does_not_consume_rollout(native_student: NativeDiffusionPipeline) -> None:
    session = DreamDojoStudentSession(native_student)
    session.configure(Image.new("RGB", (16, 16)), height=16, width=16)
    with pytest.raises(ValueError, match="12 new actions"):
        session.generate(torch.zeros(4, 384))
    assert session.chunk_index == 0 and session.condition_frames is not None


@pytest.mark.parametrize("representation", ["list", "tensor", "numpy"])
def test_public_student_offline_export_uses_isolated_rollout(
    native_student: NativeDiffusionPipeline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
) -> None:
    saves = []

    def save(pixels: torch.Tensor, path: str, *, fps: float) -> None:
        saves.append((pixels.clone(), fps))
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("worldfoundry.core.media.codecs.video.save_image_or_video_tensor", save)
    pipe = DreamDojoStudentPipeline.from_pretrained({"native_pipeline": native_student}, device="cpu")
    pipe.configure_realtime(Image.new("RGB", (16, 16)), height=16, width=16)
    pipe.stream_realtime(actions=torch.zeros(12, 384))
    state = pipe.synthesis_model.runtime._realtime_session
    frames = state.condition_frames.clone()
    actions: Any = torch.zeros(24, 384)
    if representation == "numpy":
        actions = actions.numpy()
    elif representation == "list":
        actions = actions.tolist()
    for index in range(2):
        result = pipe(
            images=Image.new("RGB", (16, 16)),
            interactions=actions,
            height=16,
            width=16,
            num_inference_steps=1,
            output_path=tmp_path / f"student-{index}.mp4",
            return_dict=True,
        )
        assert result["num_frames"] == 25 and result["variant"] == "student"
    assert saves[0][0].shape == (1, 3, 25, 16, 16) and saves[0][1] == 10
    torch.testing.assert_close(saves[0][0], saves[1][0], rtol=0, atol=0)
    torch.testing.assert_close(state.condition_frames, frames, rtol=0, atol=0)
    assert state.chunk_index == 1


@pytest.mark.parametrize(
    "options",
    [{"num_inference_steps": 35}, {"num_inference_steps": True}, {"fps": 5}, {"guidance_scale": 7}, {"height": 16.5}],
)
def test_invalid_student_options_rejected_before_loading(
    options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = DreamDojoRuntime(variant="student", device="cpu")
    monkeypatch.setattr(runtime, "_load_pipeline", lambda: pytest.fail("weights loaded before validation"))
    with pytest.raises(ValueError):
        runtime.predict(images=Image.new("RGB", (16, 16)), actions=torch.zeros(12, 384), **options)


def test_student_requires_explicit_weights_and_never_uses_teacher_default(tmp_path: Path) -> None:
    runtime = DreamDojoRuntime(variant="student", checkpoints_dir=tmp_path)
    torch.save({"teacher": True}, tmp_path / "model_ema_bf16.pt")
    with pytest.raises(ValueError, match="explicit student_checkpoint_path"):
        runtime._checkpoint_path()
    runtime.student_checkpoint_path = tmp_path / "missing-student.pt"
    with pytest.raises(FileNotFoundError, match="distilled student"):
        runtime._checkpoint_path()


def test_student_native_recipe_and_studio_contract_resolve() -> None:
    from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
    from worldfoundry.studio.inference.catalog import find_entry

    recipe = default_native_diffusion_registry().resolve("dreamdojo-student")
    assert recipe.execution.strategy == "causal-trigflow"
    assert recipe.checkpoints["transformer"].repo_id is None
    assert recipe.metadata["student_checkpoint_required"] is True
    assert load_model_zoo_registry().get("dreamdojo-student").integration_status == "integrated"
    entry = find_entry("dreamdojo-student")
    assert "student_checkpoint_path" in entry.load_params
    assert entry.default_call_kwargs["num_inference_steps"] == 4
    assert entry.default_call_kwargs["fps"] == 10
    assert {"images", "interactions", "actions", "action_file"} <= set(entry.stream_params)
    from worldfoundry.runtime.inference_catalog import get_model_inference_spec

    spec = get_model_inference_spec("dreamdojo-student")
    assert spec.default_task_id == "robot-world-generation"
    fields = {field.field_id: field for field in spec.tasks[0].inputs}
    assert fields["student_checkpoint_path"].required
    assert fields["num_inference_steps"].default == 4
    assert spec.tasks[0].default_call_kwargs["guidance_scale"] == 1.0
    from worldfoundry.cli.model_run import load_model_run_schema

    schema = load_model_run_schema("dreamdojo", variant_id="dreamdojo-2b-gr1-distilled")
    assert schema.model_id == "dreamdojo"
    assert schema.catalog_variant_id == "dreamdojo-2b-gr1-distilled"
    assert schema.runtime_status == "native_student_checkpoint_required"
    cli_fields = {field.option: field for field in schema.fields}
    assert cli_fields["--pipeline.num-inference-steps"].default == 4
    assert cli_fields["--pipeline.fps"].default == 10
    assert cli_fields["--pipeline.load.student-checkpoint-path"].required


def test_public_student_loading_selects_native_student_and_explicit_assets(
    native_student: NativeDiffusionPipeline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"student-weights")
    calls = []

    def loader(cls: Any, model_id: str, **kwargs: Any) -> NativeDiffusionPipeline:
        calls.append((model_id, kwargs))
        return native_student

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", classmethod(loader))
    pipe = DreamDojoStudentPipeline.from_pretrained(
        {
            "student_checkpoint_path": checkpoint,
            "vae_path": "/weights/tokenizer.pth",
            "text_embeddings_path": "/weights/robot-text.pt",
        },
        device="cpu",
    )
    assert not calls
    pipe.configure_realtime(Image.new("RGB", (16, 16)), height=16, width=16)
    model_id, options = calls[0]
    assert model_id == "dreamdojo-student"
    assert options["checkpoint_overrides"]["transformer"] == str(checkpoint)
    assert options["checkpoint_overrides"]["vae"] == "/weights/tokenizer.pth"
    assert options["component_options"]["conditioner:main"]["text_embeddings_path"] == "/weights/robot-text.pt"


def test_long_causal_history_is_bounded_and_noisy_reads_leave_clean_keys_unchanged() -> None:
    model = tiny_student()
    caches: dict[int, CompactingKVCache] = {}
    context = torch.randn(1, 3, 8)
    with torch.inference_mode():
        for index in range(12):
            latent = torch.randn(1, 2, 1, 4, 4)
            action = None if index == 0 else torch.zeros(1, 4, 384)
            kwargs = {
                "action": action,
                "frame_index": index,
                "kv_cache": caches,
                "condition_mask": torch.zeros(1, 1, 1, 4, 4),
            }
            model.forward_frame(latent, torch.zeros(1), context, store_kv=True, **kwargs)
            saved = {key: tuple(value.clone() for value in cache.cached()) for key, cache in caches.items()}
            model.forward_frame(
                latent + 1,
                torch.ones(1),
                context,
                **{
                    **kwargs,
                    "frame_index": index + 1,
                    "action": torch.zeros(1, 4, 384),
                },
            )
            for key, cache in caches.items():
                assert len(cache.blocks) <= 3 and cache.length <= 12
                for value, expected in zip(cache.cached(), saved[key], strict=True):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_native_student_assembler_binds_all_component_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worldfoundry.base_models.diffusion_model import RuntimePolicy
    from worldfoundry.base_models.diffusion_model.loaders import NativeModuleLoader
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE

    class ToyWan(TinyVideoVAE, WanVideoVAE):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

    def load(self: Any, spec: Any, checkpoint: Any, policy: Any) -> Any:
        if spec.module_class is CausalActionCosmosTransformer:
            return tiny_student(text_dim=100352)
        assert spec.module_class is WanVideoVAE
        return ToyWan()

    embedding_path = tmp_path / "text.pt"
    torch.save(torch.zeros(1, 2, 100352), embedding_path)
    monkeypatch.setattr(NativeModuleLoader, "load", load)
    pipeline = NativeDiffusionPipeline.from_pretrained(
        "dreamdojo-student",
        policy=RuntimePolicy(device=torch.device("cpu"), dtype=torch.float32),
        checkpoint_overrides={"transformer": str(tmp_path / "student.pt"), "vae": str(tmp_path / "tokenizer.pth")},
        component_options={"conditioner:main": {"text_embeddings_path": str(embedding_path)}},
    )
    session = DreamDojoStudentSession(pipeline)
    session.configure(Image.new("RGB", (16, 16)), height=16, width=16)
    assert session.generate(torch.zeros(12, 384))["video_chunk"].frames.shape == (1, 3, 13, 16, 16)


def test_student_ema_mapping_discards_training_networks_and_restores_real_forward() -> None:
    from worldfoundry.base_models.diffusion_model.models.denoisers.dreamdojo import convert_dreamdojo_state_dict

    model = tiny_student()
    restored = tiny_student()
    source = {f"net_ema.{key}": value for key, value in model.state_dict().items()}
    source["net_fake_score.weight"] = torch.zeros(1)
    source["net_discriminator_head.weight"] = torch.zeros(1)
    restored.load_state_dict(convert_dreamdojo_state_dict(source), strict=True)
    latent, context = torch.randn(1, 2, 1, 4, 4), torch.randn(1, 3, 8)
    with torch.inference_mode():
        outputs = [
            network.forward_frame(
                latent,
                torch.zeros(1),
                context,
                action=None,
                frame_index=0,
                kv_cache={},
                condition_mask=torch.zeros(1, 1, 1, 4, 4),
            )
            for network in (model, restored)
        ]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
