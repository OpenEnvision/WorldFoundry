from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import (  # noqa: E402
    LTXModel,
    LTXModelType,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.data.video_bucketing import VideoLatentGeometry  # noqa: E402
from worldfoundry.training.data.video_cache import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
)
from worldfoundry.training.engine.ltx import (  # noqa: E402
    LTXFlowMatchingObjective,
    LTXTimestepSamplingConfig,
    apply_ltx_tuning,
    audit_ltx_lora_targets,
    build_ltx_single_device_session,
    ltx_latent_normalization,
    sample_ltx_sigmas,
)
from worldfoundry.training.engine.ltx import sft as ltx_sft  # noqa: E402
from worldfoundry.training.models.ltx import LTXTrainAdapter  # noqa: E402
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402


class _TinyLTXModule(nn.Module):
    def __init__(self, model_type: LTXModelType = LTXModelType.VideoOnly) -> None:
        super().__init__()
        self.velocity_model = LTXModel(
            model_type=model_type,
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            cross_attention_dim=16,
            positional_embedding_max_pos=[20, 64, 64],
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_cross_attention_dim=8,
            audio_positional_embedding_max_pos=[20],
        )
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(mean=0.0, std=0.02)
        self.last_video = None

    def forward(self, *args, **kwargs):
        self.last_video = kwargs["video"]
        return self.velocity_model(*args, **kwargs)


def _adapter(
    seed: int,
    *,
    model_type: LTXModelType = LTXModelType.VideoOnly,
    first_frame_probability: float = 0.0,
    causal_positions: bool = True,
    discrete_timesteps: bool = False,
    per_sample: bool = True,
) -> tuple[LTXTrainAdapter, _TinyLTXModule]:
    torch.manual_seed(seed)
    module = _TinyLTXModule(model_type)
    adapter = LTXTrainAdapter(
        SimpleNamespace(model=module),
        expected_latent_channels=4,
        first_frame_conditioning_probability=first_frame_probability,
        per_sample_first_frame_conditioning=per_sample,
        causal_positions=causal_positions,
        discrete_timesteps=discrete_timesteps,
    )
    return adapter, module


def _batch(*, explicit_first_frame: bool = True) -> TrainingBatch:
    generator = torch.Generator().manual_seed(31)
    conditions: dict[str, torch.Tensor] = {
        "clean_latents": torch.randn(2, 4, 2, 2, 2, generator=generator),
        "video_context": torch.randn(2, 3, 16, generator=generator),
        "context_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    if explicit_first_frame:
        mask = torch.zeros(2, 1, 2, 2, 2, dtype=torch.bool)
        mask[0, :, 0] = True
        conditions["latent_conditioning_mask"] = mask
    return TrainingBatch(
        sample_ids=("first", "second"),
        prompts=("first prompt", "second prompt"),
        conditions=conditions,
        metadata={
            "target_num_frames": 9,
            "target_height": 64,
            "target_width": 64,
            "target_fps": 24.0,
        },
    )


def _objective_batch(adapter: LTXTrainAdapter, sigmas: torch.Tensor) -> ObjectiveBatch:
    prepared = adapter.prepare_batch(_batch())
    generator = torch.Generator().manual_seed(37)
    model_input = torch.randn(prepared.clean_latents.shape, generator=generator)
    return ObjectiveBatch(
        sample_ids=prepared.sample_ids,
        model_input=model_input,
        target=torch.zeros_like(model_input),
        sigmas=sigmas,
        timesteps=sigmas,
        conditioning=prepared.conditioning,
        loss_mask=prepared.loss_mask,
    )


def test_ltx_timestep_samplers_match_released_trainers() -> None:
    generator = torch.Generator().manual_seed(12345)
    current = sample_ltx_sigmas(
        8,
        2048,
        config=LTXTimestepSamplingConfig(stretch=True),
        device=torch.device("cpu"),
        generator=generator,
    )
    # LTX-2 packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py
    expected_current = torch.tensor(
        [
            0.9368942379951477,
            0.18771769106388092,
            0.4028981924057007,
            0.6735058426856995,
            0.6281424760818481,
            0.3269442319869995,
            0.0954303964972496,
            0.934267520904541,
        ]
    )
    torch.testing.assert_close(current, expected_current, rtol=0.0, atol=0.0)

    generator.manual_seed(12345)
    legacy = sample_ltx_sigmas(
        8,
        2048,
        config=LTXTimestepSamplingConfig(stretch=False),
        device=torch.device("cpu"),
        generator=generator,
    )
    # LTX-Video-Trainer src/ltxv_trainer/timestep_samplers.py
    expected_legacy = torch.tensor(
        [
            0.9395621418952942,
            0.36507076025009155,
            0.5300776958465576,
            0.3257908821105957,
            0.7028018832206726,
            0.47183388471603394,
            0.6454576849937439,
            0.9375478625297546,
        ]
    )
    torch.testing.assert_close(legacy, expected_legacy, rtol=0.0, atol=0.0)


def test_ltx_adapter_uses_clean_condition_tokens_and_native_positions() -> None:
    adapter, module = _adapter(41)
    objective_batch = _objective_batch(adapter, torch.tensor([0.1234, 0.6]))
    clean = objective_batch.conditioning["ltx_clean_conditioning_latents"]
    prediction = adapter.forward_train(objective_batch)
    video = module.last_video

    assert video is not None
    assert prediction.shape == (2, 4, 2, 2, 2)
    assert video.latent.shape == (2, 8, 4)
    torch.testing.assert_close(video.latent[0, :4], adapter.patchifier.patchify(clean)[0, :4])
    torch.testing.assert_close(video.timesteps[0], torch.tensor([0.0] * 4 + [0.1234] * 4))
    torch.testing.assert_close(video.timesteps[1], torch.full((8,), 0.6))
    assert objective_batch.loss_mask is not None
    assert objective_batch.loss_mask[0, :, 0].count_nonzero() == 0
    assert objective_batch.conditioning["context_mask"].dtype == torch.int64

    expected_first_time_bounds = torch.tensor([[0.0, 1.0 / 24.0]] * 4)
    expected_second_time_bounds = torch.tensor([[1.0 / 24.0, 9.0 / 24.0]] * 4)
    torch.testing.assert_close(video.positions[0, 0, :4], expected_first_time_bounds)
    torch.testing.assert_close(video.positions[0, 0, 4:], expected_second_time_bounds)

    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_ltx_video_profile_rounds_timesteps_and_uses_legacy_coordinates() -> None:
    adapter, module = _adapter(
        43,
        first_frame_probability=1.0,
        causal_positions=False,
        discrete_timesteps=True,
        per_sample=False,
    )
    prepared = adapter.prepare_batch(_batch(explicit_first_frame=False))
    model_input = torch.randn_like(prepared.clean_latents)
    batch = ObjectiveBatch(
        sample_ids=prepared.sample_ids,
        model_input=model_input,
        target=torch.zeros_like(model_input),
        sigmas=torch.tensor([0.1236, 0.6004]),
        timesteps=torch.tensor([0.1236, 0.6004]),
        conditioning=prepared.conditioning,
        loss_mask=prepared.loss_mask,
    )
    adapter.forward_train(batch)
    video = module.last_video

    assert video is not None
    torch.testing.assert_close(video.sigma, torch.tensor([0.124, 0.6]))
    torch.testing.assert_close(video.timesteps[:, :4], torch.zeros(2, 4))
    torch.testing.assert_close(video.timesteps[0, 4:], torch.full((4,), 0.124))
    torch.testing.assert_close(video.positions[0, 0, :4], torch.tensor([[0.0, 8.0 / 24.0]] * 4))
    torch.testing.assert_close(video.positions[0, 0, 4:], torch.tensor([[8.0 / 24.0, 16.0 / 24.0]] * 4))


def test_ltx_loss_reduces_each_sample_before_batch_mean() -> None:
    adapter, _ = _adapter(45)
    objective = LTXFlowMatchingObjective(LTXTimestepSamplingConfig())
    original = _objective_batch(adapter, torch.tensor([0.2, 0.7]))
    target = torch.zeros_like(original.target)
    prediction = torch.ones_like(target)
    prediction[0] *= 2.0
    mask = torch.ones_like(target[:, :1])
    mask[0].zero_()
    mask[0, :, 0, 0, 0] = 1.0
    batch = ObjectiveBatch(
        sample_ids=original.sample_ids,
        model_input=original.model_input,
        target=target,
        sigmas=original.sigmas,
        timesteps=original.timesteps,
        conditioning=original.conditioning,
        loss_mask=mask,
    )

    result = objective.compute_loss(prediction, batch)

    torch.testing.assert_close(result.loss, torch.tensor(2.5))
    torch.testing.assert_close(result.metrics["loss_numerator"], torch.tensor(5.0))
    torch.testing.assert_close(result.metrics["loss_denominator"], torch.tensor(2.0))


def _recipe(
    *,
    tuning: str = "full",
    model_recipe: str = "ltx-2-i2v",
    param_dtype: str = "float32",
    backend: str = "single",
) -> TrainingRecipe:
    tuning_section: dict[str, object] = {"mode": tuning}
    if tuning == "lora":
        tuning_section.update({"preset": "ltx-attention", "rank": 2, "alpha": 2})
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": "ltx-native-training", "output_dir": "unused"},
            "model": {"recipe": model_recipe},
            "tuning": tuning_section,
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "max_latent_tokens_per_microbatch": 8,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"num_workers": 0, "pin_memory": False},
            },
            "objective": {
                "type": "flow-matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "shifted-logit-normal",
                "options": {"first_frame_conditioning_probability": 0.0},
            },
            "optimizer": {"type": "adamw", "learning_rate": 1.0e-3},
            "runtime": {"param_dtype": param_dtype, "reduce_dtype": "float32"},
            "distributed": {"backend": backend},
        }
    )


def _cache(root: Path, *, model_recipe: str = "ltx-2-i2v") -> VideoCachedDataset:
    store = VideoCacheStore(root)
    source = VideoCacheProvenance(
        media_uri="ltx.mp4",
        prompt="a tiny moving object",
        model_recipe=model_recipe,
        codec={"name": "ltx-video-vae"},
        conditioner={"name": "ltx-gemma-connector"},
        tokenizer={"name": "gemma-tokenizer"},
        conditioning_inputs={"task": "i2v"},
        safety_audit={"safe": True},
        frame_sampling={"mode": "head"},
        spatial_transform={"mode": "identity"},
        latent_normalization=ltx_latent_normalization(model_recipe),
        task="i2v",
        conditioning_layout="t5-sequence" if model_recipe == "ltx-video-i2v" else "gemma-sequence",
        aspect_bin="1:1",
        source_num_frames=9,
        source_height=64,
        source_width=64,
        source_fps=24.0,
        target_num_frames=9,
        target_height=64,
        target_width=64,
        target_fps=24.0,
        latent_geometry=VideoLatentGeometry(32, 32, 8, "first-frame"),
    )
    generator = torch.Generator().manual_seed(47)
    entry = store.write_sample(
        sample_id="ltx",
        provenance=source,
        clean_latents=torch.randn(4, 2, 2, 2, generator=generator),
        conditioning={
            "video_context": torch.randn(3, 16, generator=generator),
            "context_mask": torch.ones(3, dtype=torch.int64),
        },
        conditioning_layouts={
            "video_context": "sequence-features",
            "context_mask": "sequence",
        },
        latent_loss_mask=torch.ones(1, 2, 2, 2),
        valid_latent_mask=torch.ones(1, 2, 2, 2, dtype=torch.bool),
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def test_ltx_session_reuses_shared_video_flow_engine_and_updates(tmp_path: Path) -> None:
    adapter, _ = _adapter(53)
    before = {name: value.detach().clone() for name, value in adapter.trainable_module.named_parameters()}
    session = build_ltx_single_device_session(
        recipe=_recipe(),
        adapter=adapter,
        dataset=_cache(tmp_path / "cache"),
        output_dir=tmp_path / "run",
        fused_adamw=False,
    )

    summary = session.run(max_steps=1, seed=59)

    assert summary.changed_parameter_tensors > 0
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in adapter.trainable_module.named_parameters()
    )
    assert session.data_identity["cache_contract"]["context_features"] == 16
    assert session.data_identity["token_sampler"]["world_size"] == 1


def test_ltx_video_attention_lora_targets_are_exact() -> None:
    pytest.importorskip("peft")
    adapter, _ = _adapter(61)
    audit = audit_ltx_lora_targets(adapter.trainable_module)

    assert audit.block_count == 1
    assert len(audit.module_names) == 8
    assert all(".attn1." in name or ".attn2." in name for name in audit.module_names)
    assert all(name.endswith(("to_q", "to_k", "to_v", "to_out.0")) for name in audit.module_names)

    application = apply_ltx_tuning(_recipe(tuning="lora"), adapter)

    assert application is not None
    assert set(application.targeted_module_names) == set(audit.module_names)
    assert all("lora_" in name for name in application.trainable_parameter_names)

    prediction = adapter.forward_train(_objective_batch(adapter, torch.tensor([0.2, 0.7])))
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in application.model.parameters() if parameter.requires_grad)


def test_ltx2_lora_uses_fp32_optimizer_storage_with_bfloat16_compute(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    recipe = _recipe(tuning="lora", param_dtype="bfloat16")
    adapter, _ = _adapter(62)
    adapter.trainable_module.to(dtype=torch.bfloat16)

    session = build_ltx_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=_cache(tmp_path / "cache"),
        output_dir=tmp_path / "run",
        fused_adamw=False,
    )

    assert session.engine.autocast_dtype is torch.bfloat16
    assert {parameter.dtype for parameter in adapter.trainable_module.parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in session.engine.parameters} == {torch.float32}
    assert {parameter.dtype for group in session.engine.optimizer.param_groups for parameter in group["params"]} == {
        torch.float32
    }


def test_ltx2_fsdp2_applies_fp32_lora_storage_before_sharding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("peft")
    recipe = _recipe(tuning="lora", param_dtype="bfloat16", backend="fsdp2")
    adapter, _ = _adapter(63)
    adapter.trainable_module.to(dtype=torch.bfloat16)
    distributed_context = object()
    built_session = object()

    def build_session(**kwargs):
        from worldfoundry.training.distributed.fsdp import _precision_island_modules

        assert kwargs["distributed_context"] is distributed_context
        assert kwargs["recipe"].runtime.param_dtype == "bfloat16"
        assert kwargs["master_parameter_dtype"] is torch.float32
        application = kwargs["tuning_factory"](kwargs["recipe"], kwargs["adapter"])
        assert application is not None
        assert {parameter.dtype for parameter in kwargs["adapter"].trainable_module.parameters()} == {torch.float32}
        assert not _precision_island_modules(
            kwargs["adapter"].trainable_module,
            param_dtype=torch.bfloat16,
            master_parameter_dtype=kwargs["master_parameter_dtype"],
        )
        return built_session

    monkeypatch.setattr(ltx_sft, "build_cached_video_flow_fsdp2_session", build_session)

    result = ltx_sft.build_ltx_fsdp2_session(
        recipe=recipe,
        adapter=adapter,
        dataset=_cache(tmp_path / "cache"),
        distributed_context=distributed_context,
        output_dir=tmp_path / "run",
        fused_adamw=False,
    )

    assert result is built_session


def test_legacy_ltx_lora_preserves_released_bfloat16_storage(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    recipe = _recipe(
        tuning="lora",
        model_recipe="ltx-video-i2v",
        param_dtype="bfloat16",
    )
    adapter, _ = _adapter(
        64,
        causal_positions=False,
        discrete_timesteps=True,
        per_sample=False,
    )
    adapter.trainable_module.to(dtype=torch.bfloat16)

    session = build_ltx_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=_cache(tmp_path / "cache", model_recipe="ltx-video-i2v"),
        output_dir=tmp_path / "run",
        fused_adamw=False,
    )

    assert session.engine.autocast_dtype is torch.bfloat16
    frozen_dtypes = {
        parameter.dtype for parameter in adapter.trainable_module.parameters() if not parameter.requires_grad
    }
    trainable_dtypes = {
        parameter.dtype for parameter in adapter.trainable_module.parameters() if parameter.requires_grad
    }
    assert frozen_dtypes == {torch.bfloat16}
    assert trainable_dtypes == {torch.float32}


def test_ltx_lora_export_matches_released_single_file_format(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    pytest.importorskip("peft")
    adapter, _ = _adapter(63)
    application = apply_ltx_tuning(_recipe(tuning="lora"), adapter)

    assert application is not None
    artifact = application.export_adapter(tmp_path / "ltx-lora")
    state = load_file(artifact.path / "lora_weights.safetensors")

    assert artifact.file_size_bytes.keys() == {"lora_weights.safetensors"}
    assert len(state) == 2 * len(application.targeted_module_names)
    assert all(name.startswith("diffusion_model.") for name in state)
    assert all("base_model.model." not in name for name in state)
    assert all(value.dtype == torch.bfloat16 for value in state.values())


def test_ltx_video_lora_has_no_dead_audio_targets_on_audio_video_model() -> None:
    peft = pytest.importorskip("peft")
    adapter, _ = _adapter(67, model_type=LTXModelType.AudioVideo)
    audit = audit_ltx_lora_targets(adapter.trainable_module)

    assert audit.block_count == 1
    assert len(audit.module_names) == 8
    assert all(".attn1." in name or ".attn2." in name for name in audit.module_names)
    assert not any(
        marker in name
        for name in audit.module_names
        for marker in ("audio_attn", "audio_to_video_attn", "video_to_audio_attn")
    )

    application = apply_ltx_tuning(_recipe(tuning="lora"), adapter)

    assert application is not None
    prediction = adapter.forward_train(_objective_batch(adapter, torch.tensor([0.2, 0.7])))
    prediction.square().mean().backward()
    trainable = {name: parameter for name, parameter in application.model.named_parameters() if parameter.requires_grad}
    exported = peft.get_peft_model_state_dict(application.model)

    assert len(exported) == len(trainable)
    assert all(parameter.grad is not None for parameter in trainable.values())
    assert all(".attn1." in name or ".attn2." in name for name in exported)
    assert not any(
        marker in name for name in exported for marker in ("audio_attn", "audio_to_video_attn", "video_to_audio_attn")
    )
