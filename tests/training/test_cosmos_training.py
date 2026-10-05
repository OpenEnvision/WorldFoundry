from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.contracts import (
    DenoiserInput,
    MultiModalDenoiserOutput,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos2 import Cosmos2Denoiser
from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos2p5 import Cosmos25Denoiser
from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos3 import Cosmos3JointDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.model import (
    Cosmos25Transformer3DModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.model import (
    Cosmos3OmniTransformer,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch
from worldfoundry.training.engine.cosmos import (
    COSMOS3_DMD2_FLOW_SIGMAS,
    COSMOS3_LORA_PRESET,
    COSMOS3_NANO_VISION_SFT_PRESET,
    COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL,
    COSMOS_PREDICT25_DMD2_FLOW_SIGMAS,
    COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES,
    COSMOS_PREDICT_LORA_PRESET,
    COSMOS_PREDICT_LOSS_SCALE,
    Cosmos3VideoDMD2PredictionAdapter,
    Cosmos3VisionFlowMatchingObjective,
    CosmosDMD2DiscriminatorHead,
    CosmosDMD2GuidanceAdapter,
    CosmosFlowDMD2PredictionAdapter,
    CosmosPredictFlowMatchingObjective,
    apply_cosmos_tuning,
    audit_cosmos_lora_targets,
    cosmos3_dmd2_schedule,
    cosmos_predict25_dmd2_schedule,
    trigflow_time_to_flow_sigma,
)
from worldfoundry.training.models.cosmos import (
    Cosmos3TrainAdapter,
    CosmosPredict2TrainAdapter,
    CosmosPredict25TrainAdapter,
)
from worldfoundry.training.objectives.flow_matching import FlowMatchingConfig, FlowMatchingObjective
from worldfoundry.training.recipes.spec import TrainingRecipe


def _predict_model(*, layers: int = 2) -> Cosmos25Transformer3DModel:
    return Cosmos25Transformer3DModel(
        in_channels=3,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=layers,
        mlp_ratio=2.0,
        text_in_channels=4,
        text_embed_dim=24,
        adaln_lora_dim=4,
        max_size=(2, 4, 4),
        patch_size=(1, 1, 1),
        rope_scale=(1.0, 1.0, 1.0),
        use_crossattn_projection=True,
        concat_padding_mask=True,
    )


def _predict_conditioning(batch_size: int) -> dict[str, torch.Tensor | float]:
    return {
        "context": torch.randn(batch_size, 3, 4),
        "condition_latents": torch.zeros(batch_size, 2, 1, 2, 2),
        "condition_mask": torch.zeros(batch_size, 1, 1, 2, 2),
        "condition_indicator": torch.zeros(batch_size, 1, 1, 1, 1),
        "padding_mask": torch.zeros(batch_size, 1, 2, 2),
        "fps": 16.0,
    }


def _objective_batch(
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    conditioning: dict[str, object],
    *,
    noise: torch.Tensor | None = None,
) -> ObjectiveBatch:
    return ObjectiveBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(int(latents.shape[0]))),
        model_input=latents,
        target=torch.zeros_like(latents),
        sigmas=sigmas,
        timesteps=sigmas,
        conditioning=conditioning,
        noise=noise,
        metadata={"prediction_type": "flow_velocity"},
    )


def test_predict2_adapter_matches_released_rf_preconditioning() -> None:
    torch.manual_seed(7)
    model = _predict_model()
    denoiser = Cosmos2Denoiser(model)
    adapter = CosmosPredict2TrainAdapter(
        denoiser,
        None,
        None,
        expected_latent_channels=2,
    )
    normalized_state = torch.randn(2, 2, 1, 2, 2)
    sigmas = torch.tensor([0.25, 0.6])
    conditioning = _predict_conditioning(2)
    velocity = adapter.forward_model(
        _objective_batch(normalized_state, sigmas, conditioning),
        training=False,
    )

    edm_sigmas = sigmas / (1.0 - sigmas)
    clean = denoiser(
        DenoiserInput(
            latents=normalized_state / (1.0 - sigmas[:, None, None, None, None]),
            timestep=edm_sigmas,
            next_timestep=torch.zeros_like(edm_sigmas),
            conditioning=conditioning,
            step_index=0,
            total_steps=1000,
        )
    ).sample
    expected = (normalized_state - clean) / sigmas[:, None, None, None, None]
    torch.testing.assert_close(velocity, expected, atol=2.0e-5, rtol=2.0e-5)


def test_predict25_flow_training_supports_per_sample_timesteps_and_backward() -> None:
    torch.manual_seed(11)
    model = _predict_model()
    adapter = CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(model),
        None,
        None,
        expected_latent_channels=2,
    )
    clean = torch.randn(2, 2, 1, 2, 2)
    objective = CosmosPredictFlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("left", "right"),
            prompts=("left", "right"),
            conditions={"clean_latents": clean, **_predict_conditioning(2)},
        )
    )
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(13))
    observed_timesteps: list[torch.Tensor] = []
    handle = model.time_embed.register_forward_pre_hook(
        lambda _module, args: observed_timesteps.append(args[1].detach().clone())
    )
    try:
        prediction = adapter.forward_train(batch)
    finally:
        handle.remove()
    result = objective.compute_loss(prediction, batch)
    result.loss.backward()
    assert prediction.shape == clean.shape
    torch.testing.assert_close(observed_timesteps[0], batch.sigmas)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_predict25_lora_condition_frames_keep_the_sampled_training_timestep() -> None:
    model = _predict_model()
    adapter = CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(model),
        None,
        None,
        expected_latent_channels=2,
    )
    clean = torch.randn(1, 2, 2, 2, 2)
    conditions = _predict_conditioning(1)
    conditions["condition_latents"] = clean
    conditions["condition_mask"] = torch.tensor([[[[[1.0]], [[0.0]]]]]).expand(1, 1, 2, 2, 2)
    conditions["condition_indicator"] = torch.tensor([[[[[1.0]], [[0.0]]]]])
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("sample",),
            prompts=("sample",),
            conditions={"clean_latents": clean, **conditions},
        )
    )
    assert prepared.conditioning["conditional_frame_timestep"] == -1.0
    batch = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform")).corrupt(
        prepared,
        generator=torch.Generator().manual_seed(23),
    )
    observed_timesteps: list[torch.Tensor] = []
    handle = model.time_embed.register_forward_pre_hook(
        lambda _module, args: observed_timesteps.append(args[1].detach().clone())
    )
    try:
        adapter.forward_train(batch)
    finally:
        handle.remove()
    torch.testing.assert_close(observed_timesteps[0], batch.sigmas)


def test_cosmos3_waver_sampler_matches_released_continuous_rf_formula() -> None:
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="waver", flow_shift=3.0))
    generator = torch.Generator().manual_seed(117)
    actual = objective.sample_sigmas(4, device=torch.device("cpu"), generator=generator)

    uniform = torch.rand(4, generator=torch.Generator().manual_seed(117))
    waver = 1.0 - uniform - 1.29 * (torch.cos(torch.pi * 0.5 * uniform).square() - 1.0 + uniform)
    expected = 3.0 * waver / (1.0 + 2.0 * waver)
    torch.testing.assert_close(actual, expected)


def test_predict25_text_dropout_is_per_sample_and_generator_deterministic() -> None:
    context = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2) + 1.0
    batch = ObjectiveBatch(
        sample_ids=("a", "b", "c", "d"),
        model_input=torch.zeros(4, 1),
        target=torch.zeros(4, 1),
        sigmas=torch.full((4,), 0.5),
        timesteps=torch.full((4,), 0.5),
        conditioning={"context": context, "negative_context": -context},
    )
    objective = CosmosPredictFlowMatchingObjective(conditioning_dropout_probability=0.5)
    actual = objective._apply_text_dropout(  # noqa: SLF001 - released dropout formula contract.
        batch,
        generator=torch.Generator().manual_seed(43),
    )
    draws = torch.rand(4, generator=torch.Generator().manual_seed(43))
    keep = (draws >= 0.5).reshape(4, 1, 1)
    torch.testing.assert_close(actual.conditioning["context"], context * keep)
    torch.testing.assert_close(actual.conditioning["negative_context"], -context)
    assert int(actual.metadata["conditioning_dropped_samples"]) == int((~keep).sum())


def test_released_dmd2_grids_and_trigflow_conversion() -> None:
    converted = tuple(trigflow_time_to_flow_sigma(value) for value in COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES)
    assert converted == pytest.approx(COSMOS_PREDICT25_DMD2_FLOW_SIGMAS)
    assert cosmos_predict25_dmd2_schedule().sigmas == COSMOS_PREDICT25_DMD2_FLOW_SIGMAS
    assert cosmos3_dmd2_schedule().sigmas == COSMOS3_DMD2_FLOW_SIGMAS
    assert COSMOS_DMD2_GENERATOR_UPDATE_INTERVAL == 5


def test_predict25_objective_is_unscaled_velocity_mse() -> None:
    prediction = torch.tensor([[[2.0, -1.0]]])
    target = torch.tensor([[[0.0, 1.0]]])
    batch = ObjectiveBatch(
        sample_ids=("sample",),
        model_input=torch.zeros_like(prediction),
        target=target,
        sigmas=torch.tensor([0.5]),
        timesteps=torch.tensor([0.5]),
        conditioning={},
    )
    result = CosmosPredictFlowMatchingObjective().compute_loss(prediction, batch)
    torch.testing.assert_close(result.loss, (prediction - target).square().mean())
    assert COSMOS_PREDICT_LOSS_SCALE == 1.0


def test_predict_condition_frames_are_zero_error_with_full_tensor_denominator() -> None:
    clean = torch.zeros(1, 2, 3, 1, 1)
    objective = CosmosPredictFlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform"),
        conditional_frame_probabilities=(0.0, 0.0, 1.0),
    )
    prepared = PreparedBatch(
        sample_ids=("sample",),
        clean_latents=clean,
        conditioning={
            "condition_latents": clean,
            "condition_mask": torch.zeros(1, 1, 3, 1, 1),
            "condition_indicator": torch.zeros(1, 1, 3, 1, 1),
        },
    )
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(31))
    mask = batch.conditioning["condition_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask[:, :, :2].all() and not mask[:, :, 2:].any()
    prediction = batch.target + (1.0 - mask)
    result = objective.compute_loss(prediction, batch)
    torch.testing.assert_close(result.loss, torch.tensor(COSMOS_PREDICT_LOSS_SCALE / 3.0))
    assert float(result.metrics["loss_denominator"]) == pytest.approx(float(clean.numel()))


def test_predict25_dmd2_guidance_preserves_trigflow_fake_score_weight() -> None:
    torch.manual_seed(17)
    adapter = CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(_predict_model()),
        None,
        None,
        expected_latent_channels=2,
    )
    prediction = CosmosFlowDMD2PredictionAdapter(adapter, checkpoint_identity="fake-score")
    guidance = CosmosDMD2GuidanceAdapter(
        prediction,
        checkpoint_identity="fake-score",
        trigflow_denoising_weight=True,
    )
    clean = torch.randn(2, 2, 1, 2, 2)
    noise = torch.randn_like(clean)
    levels = torch.tensor([0.25, 0.8])
    noisy = prediction.add_noise(clean, noise, levels)
    conditioning = _predict_conditioning(2)
    velocity = prediction.predict_velocity(
        noisy,
        levels,
        sample_ids=("left", "right"),
        conditioning=conditioning,
        training=True,
    )
    expected = (velocity - (noise - clean)).float().square().flatten(1).mean(1)
    expected = expected * (levels.square() + (1.0 - levels).square())
    actual = guidance.denoising_loss_per_sample(
        clean,
        noisy,
        noise,
        levels,
        sample_ids=("left", "right"),
        conditioning=conditioning,
        training=True,
    )
    torch.testing.assert_close(actual, expected)


def test_predict25_discriminator_consumes_selected_dit_features() -> None:
    torch.manual_seed(19)
    adapter = CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(_predict_model()),
        None,
        None,
        expected_latent_channels=2,
    )
    prediction = CosmosFlowDMD2PredictionAdapter(adapter, checkpoint_identity="fake-score")
    head = CosmosDMD2DiscriminatorHead(model_channels=24, num_branches=2)
    guidance = CosmosDMD2GuidanceAdapter(
        prediction,
        checkpoint_identity="fake-score",
        discriminator=head,
        intermediate_feature_ids=(0, 1),
    )
    logits = guidance.discriminator_logits(
        torch.randn(2, 2, 1, 2, 2),
        torch.tensor([0.2, 0.7]),
        sample_ids=("left", "right"),
        conditioning=_predict_conditioning(2),
        training=True,
    )
    assert logits.shape == (2, 1)
    logits.sum().backward()
    assert head.final_linear.weight.grad is not None


class _JointModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(1, 1),))
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self.gradient_checkpointing = bool(enable)


class _JointDenoiser:
    def __init__(self) -> None:
        self.model = _JointModule()
        self.timesteps: list[torch.Tensor] = []
        self.states: list[object] = []
        self.input_ids: list[torch.Tensor] = []

    def __call__(self, model_input) -> MultiModalDenoiserOutput:
        self.timesteps.append(model_input.timestep.detach().clone())
        self.states.append(model_input.modalities["video"])
        self.input_ids.append(model_input.conditioning["input_ids"].detach().clone())
        return MultiModalDenoiserOutput(
            samples={name: state.latent * self.model.scale for name, state in model_input.modalities.items()}
        )


class _CapturingCosmos3Denoiser:
    def __init__(self, model: Cosmos3OmniTransformer) -> None:
        self.model = model
        self.inner = Cosmos3JointDenoiser(model)
        self.timesteps: list[torch.Tensor] = []

    def __call__(self, model_input) -> MultiModalDenoiserOutput:
        self.timesteps.append(model_input.timestep.detach().clone())
        return self.inner(model_input)


def test_cosmos3_video_sft_batches_joint_sequences_per_rank_and_backpropagates() -> None:
    denoiser = _JointDenoiser()
    adapter = Cosmos3TrainAdapter(denoiser, expected_latent_channels=2)
    clean = torch.randn(2, 2, 2, 2, 2)
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("left", "right"),
            prompts=("left", "right"),
            conditions={
                "clean_latents": clean,
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "empty_input_ids": torch.tensor([[9, 9], [9, 9]]),
            },
        )
    )
    objective = Cosmos3VisionFlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform"),
        conditioning_config={0: 1.0},
        conditioning_dropout=0.0,
    )
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(23))
    prediction = adapter.forward_train(batch)
    result = objective.compute_loss(prediction, batch)
    result.loss.backward()
    assert prediction["video"].shape == clean.shape
    assert denoiser.model.scale.grad is not None
    torch.testing.assert_close(
        torch.stack(denoiser.timesteps),
        batch.sigmas * 1000.0,
    )


def test_cosmos3_native_omni_adapter_masked_forward_and_backward() -> None:
    model = Cosmos3OmniTransformer(
        hidden_size=12,
        head_dim=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=24,
        num_hidden_layers=1,
        latent_channel=2,
        latent_patch_size=1,
        patch_latent_dim=2,
        vocab_size=32,
        rope_scaling={"mrope_section": [1, 1, 1]},
    )
    denoiser = _CapturingCosmos3Denoiser(model)
    adapter = Cosmos3TrainAdapter(
        denoiser,
        expected_latent_channels=2,
        gradient_checkpointing=True,
    )
    recipe = TrainingRecipe.from_file(Path(__file__).parents[2] / "configs/training/cosmos3_nano_vision_sft.yaml")
    apply_cosmos_tuning(recipe, adapter)
    clean = torch.randn(1, 2, 2, 1, 1)
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("sample",),
            prompts=("sample",),
            conditions={
                "clean_latents": clean,
                "input_ids": torch.tensor([[1, 2, 3]]),
                "empty_input_ids": torch.tensor([[4, 5]]),
            },
        )
    )
    objective = Cosmos3VisionFlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform"),
        conditioning_config={1: 1.0},
        conditioning_dropout=0.0,
    )
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(29))
    layer_calls = 0

    def count_layer_calls(_module, _inputs, _output) -> None:
        nonlocal layer_calls
        layer_calls += 1

    handle = model.layers[0].register_forward_hook(count_layer_calls)
    try:
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            prediction = adapter.forward_train(batch)
            result = objective.compute_loss(prediction, batch)
            result.loss.backward()
    finally:
        handle.remove()

    assert prediction["video"].shape == clean.shape
    torch.testing.assert_close(denoiser.timesteps[0], batch.sigmas[0] * 1000.0)
    assert model.proj_out.weight.grad is not None
    assert model.proj_in.weight.grad is not None
    assert model.time_embedder.linear_1.weight.grad is not None
    assert model.layers[0].self_attn.add_q_proj.weight.grad is not None
    assert model.layers[0].mlp_moe_gen.gate_proj.weight.grad is not None
    assert model.layers[0].self_attn.to_q.weight.grad is None
    assert layer_calls == 2


def test_cosmos3_step_rng_selects_official_task_mix_and_cfg_dropout() -> None:
    clean = torch.zeros(12, 2, 4, 1, 1)
    prepared = PreparedBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(12)),
        clean_latents={"video": clean},
        conditioning={
            "input_ids": torch.ones(12, 3, dtype=torch.long),
            "empty_input_ids": torch.zeros(12, 2, dtype=torch.long),
        },
    )
    objective = Cosmos3VisionFlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(5))

    counts = batch.conditioning["num_conditional_frames"]
    assert counts.tolist() == [1, 0, 2, 1, 2, 0, 0, 1, 0, 2, 0, 0]
    assert batch.conditioning["caption_dropout_mask"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    mask = batch.conditioning["denoise_masks"]["video"]
    torch.testing.assert_close(mask[:, 0, :, 0, 0].sum(1), 4.0 - counts.float())
    assert all(row[:count].eq(0).all() for row, count in zip(mask[:, 0, :, 0, 0], counts, strict=True))


def test_cosmos3_conditioned_frames_stay_clean_and_cfg_uses_cached_empty_ids() -> None:
    denoiser = _JointDenoiser()
    adapter = Cosmos3TrainAdapter(denoiser, expected_latent_channels=2)
    clean = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3, 1, 1)
    prepared = adapter.prepare_batch(
        TrainingBatch(
            sample_ids=("conditioned", "unconditional"),
            prompts=("conditioned", "unconditional"),
            conditions={
                "clean_latents": clean,
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "empty_input_ids": torch.tensor([[8, 9], [8, 9]]),
            },
        )
    )
    objective = Cosmos3VisionFlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform"),
        conditioning_config={1: 1.0},
        conditioning_dropout=1.0,
    )
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(41))
    adapter.forward_train(batch)

    for index, state in enumerate(denoiser.states):
        torch.testing.assert_close(state.latent[:, :, :1], clean[index : index + 1, :, :1])
        torch.testing.assert_close(state.latent[:, :, 1:], batch.model_input["video"][index : index + 1, :, 1:])
        assert state.denoise_mask[:, :, :1].eq(0).all()
        assert state.denoise_mask[:, :, 1:].eq(1).all()
    assert [ids.tolist() for ids in denoiser.input_ids] == [[8, 9], [8, 9]]


def test_cosmos3_condition_error_is_zero_but_valid_elements_stay_in_denominator() -> None:
    clean = torch.zeros(1, 2, 3, 1, 1)
    valid = torch.ones_like(clean)
    prepared = PreparedBatch(
        sample_ids=("sample",),
        clean_latents={"video": clean},
        conditioning={
            "input_ids": torch.tensor([[1, 2]]),
            "empty_input_ids": torch.tensor([[3]]),
        },
        loss_mask={"video": valid},
    )
    objective = Cosmos3VisionFlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler="uniform"),
        conditioning_config={1: 1.0},
        conditioning_dropout=0.0,
    )
    batch = objective.corrupt(prepared, generator=torch.Generator().manual_seed(43))
    target = batch.target["video"]
    prediction = {"video": target + 1.0}
    result = objective.compute_loss(prediction, batch)

    torch.testing.assert_close(result.loss, torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(result.metrics["loss_numerator"], torch.tensor(4.0))
    torch.testing.assert_close(result.metrics["loss_denominator"], torch.tensor(6.0))


def test_cosmos3_sft_does_not_claim_unreleased_joint_modality_training() -> None:
    adapter = Cosmos3TrainAdapter(_JointDenoiser(), expected_latent_channels=2)
    video = torch.randn(1, 2, 2, 2, 2)
    sound = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match="only the video modality"):
        adapter.prepare_batch(
            TrainingBatch(
                sample_ids=("sample",),
                prompts=("sample",),
                conditions={
                    "clean_latents": {"video": video, "sound": sound},
                    "input_ids": torch.tensor([[1, 2, 3]]),
                },
            )
        )


def test_cosmos3_video_dmd2_role_uses_native_joint_adapter() -> None:
    adapter = Cosmos3TrainAdapter(_JointDenoiser(), expected_latent_channels=2)
    prediction = Cosmos3VideoDMD2PredictionAdapter(adapter, checkpoint_identity="cosmos3-fake-score")
    noisy = torch.randn(2, 2, 2, 2, 2)
    clean = prediction.predict_clean(
        noisy,
        torch.tensor([0.5, 0.25]),
        sample_ids=("left", "right"),
        conditioning={
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "denoise_masks": {"video": torch.ones_like(noisy)},
        },
        training=True,
    )
    assert clean.shape == noisy.shape


def test_cosmos_family_lora_audits_match_released_targets() -> None:
    predict = audit_cosmos_lora_targets(_predict_model(), COSMOS_PREDICT_LORA_PRESET)
    assert predict.block_count == 2
    assert len(predict.module_names) == 20

    cosmos3 = Cosmos3OmniTransformer(
        hidden_size=12,
        head_dim=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=24,
        num_hidden_layers=2,
        latent_channel=2,
        latent_patch_size=1,
        patch_latent_dim=2,
        vocab_size=32,
        rope_scaling={"mrope_section": [1, 1, 1]},
    )
    generation = audit_cosmos_lora_targets(cosmos3, COSMOS3_LORA_PRESET)
    assert generation.block_count == 2
    assert len(generation.module_names) == 8
    assert all(".self_attn.add_" in name or name.endswith(".to_add_out") for name in generation.module_names)


def test_cosmos3_nano_formal_profile_selects_the_released_optimizer_keys() -> None:
    model = Cosmos3OmniTransformer(
        hidden_size=12,
        head_dim=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=24,
        num_hidden_layers=2,
        latent_channel=2,
        latent_patch_size=1,
        patch_latent_dim=2,
        vocab_size=32,
        rope_scaling={"mrope_section": [1, 1, 1]},
    )
    denoiser = _JointDenoiser()
    denoiser.model = model
    adapter = Cosmos3TrainAdapter(denoiser, expected_latent_channels=2)
    recipe = TrainingRecipe.from_file(Path(__file__).parents[2] / "configs/training/cosmos3_nano_vision_sft.yaml")

    assert recipe.tuning.preset == COSMOS3_NANO_VISION_SFT_PRESET
    assert apply_cosmos_tuning(recipe, adapter) is None
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        "moe_gen" in name
        or ".self_attn.add_q_proj." in name
        or ".self_attn.add_k_proj." in name
        or ".self_attn.add_v_proj." in name
        or ".self_attn.to_add_out." in name
        or ".self_attn.norm_added_q." in name
        or ".self_attn.norm_added_k." in name
        or name.startswith("time_embedder.")
        or name.startswith("proj_in.")
        or name.startswith("proj_out.")
        for name in trainable
    )
    assert any(name.startswith("proj_in.") for name in trainable)
    assert any(name.startswith("proj_out.") for name in trainable)
    assert any("self_attn.add_q_proj" in name for name in trainable)
    assert any("self_attn.norm_added_q" in name for name in trainable)
    assert not any("self_attn.to_q" in name for name in trainable)
    assert not any("input_layernorm.weight" in name for name in trainable)


def test_gen3c_is_not_a_cosmos_training_recipe() -> None:
    from worldfoundry.training.engine.cosmos import sft

    assert "gen3c-cosmos1-7b" not in sft._SUPPORTED_RECIPES
    assert math.isclose(COSMOS_PREDICT25_DMD2_TRIGFLOW_TIMES[0], math.pi / 2.0)
