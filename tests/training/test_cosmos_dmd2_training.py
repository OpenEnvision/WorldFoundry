from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos2p5 import Cosmos25Denoiser
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.model import (
    Cosmos25Transformer3DModel,
)
from worldfoundry.cli.training import register_training_subparser
from worldfoundry.training.api.contracts import PreparedBatch
from worldfoundry.training.engine.cosmos.dmd2 import (
    COSMOS_PREDICT25_DMD2_PRETRAINED_REVISION,
    COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE,
    _cosmos_predict25_dmd2_scheduler,
    _prepare_trainable_backbone,
    _validate_official_recipe,
    cosmos_predict25_dmd2_lr_multiplier,
    cosmos_predict25_dmd2_pretrained_checkpoint,
)
from worldfoundry.training.engine.cosmos.dmd2_data import cosmos_predict25_dmd2_batch
from worldfoundry.training.engine.cosmos.dmd2_roles import (
    CosmosDMD2DiscriminatorHead,
    CosmosDMD2GuidanceAdapter,
    CosmosFlowDMD2PredictionAdapter,
)
from worldfoundry.training.engine.cosmos.sft import apply_cosmos_tuning
from worldfoundry.training.models.cosmos import CosmosPredict25TrainAdapter
from worldfoundry.training.post_training.distillation.dmd.objective import FewStepSchedule
from worldfoundry.training.post_training.distillation.dmd2.contracts import DMD2TrainingBatch
from worldfoundry.training.post_training.distillation.dmd2.math import (
    dmd2_distribution_gradient,
    dmd2_proxy_loss_per_sample,
)
from worldfoundry.training.post_training.distillation.dmd2.objective import (
    DMD2Config,
    NativeDMD2LossAdapter,
    dmd2_teacher_guidance,
    sample_dmd2_score_levels,
)
from worldfoundry.training.recipes import PostTrainingRecipe

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/post_training/cosmos_predict25_2b_dmd2.yaml"


def test_cosmos_engine_facade_is_a_lightweight_fresh_import() -> None:
    script = """
import sys
import worldfoundry.training.engine.cosmos
assert 'worldfoundry.training.engine.cosmos.dmd2' not in sys.modules
assert 'worldfoundry.base_models.diffusion_model.models.encoders.cosmos2p5' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def _model(*, layers: int = 2) -> Cosmos25Transformer3DModel:
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


def _adapter(model: Cosmos25Transformer3DModel) -> CosmosPredict25TrainAdapter:
    return CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(model),
        None,
        None,
        expected_latent_channels=2,
    )


def _conditioning(batch_size: int) -> dict[str, object]:
    return {
        "context": torch.randn(batch_size, 3, 4),
        "negative_context": torch.randn(batch_size, 3, 4),
        "condition_latents": torch.zeros(batch_size, 2, 1, 2, 2),
        "condition_mask": torch.zeros(batch_size, 1, 1, 2, 2),
        "condition_indicator": torch.zeros(batch_size, 1, 1, 1, 1),
        "padding_mask": torch.zeros(batch_size, 1, 2, 2),
        "fps": 16.0,
    }


def _batch(batch_size: int = 2) -> DMD2TrainingBatch:
    conditioning = _conditioning(batch_size)
    unconditional = dict(conditioning)
    unconditional["context"] = conditioning["negative_context"]
    return DMD2TrainingBatch(
        sample_ids=tuple(f"generated-{index}" for index in range(batch_size)),
        real_sample_ids=tuple(f"real-{index}" for index in range(batch_size)),
        real_latents=torch.randn(batch_size, 2, 1, 2, 2),
        conditioning=conditioning,
        unconditional_conditioning=unconditional,
        real_conditioning=conditioning,
    )


def test_cosmos_dmd2_profile_is_the_released_t2v_discriminator_loop() -> None:
    recipe = PostTrainingRecipe.from_file(PROFILE)
    algorithm = _validate_official_recipe(recipe)

    assert algorithm.update_mode == "alternating"
    assert algorithm.rollout_noise_mode == "shared-initial"
    assert algorithm.student_step_sampling == "rank-shared"
    assert algorithm.shared_adversarial_score_input is True
    assert algorithm.distribution_matching_dtype == "float64"
    assert algorithm.distribution_matching_weight == 2.0
    assert recipe.data.options["conditional_frame_probabilities"] == [0.6, 0.2, 0.2]
    bucket = recipe.data.options["video_buckets"][0]
    assert (bucket["num_frames"], bucket["height"], bucket["width"]) == (93, 704, 1280)
    assert recipe.data.max_latent_tokens_per_microbatch == 24 * 88 * 160
    assert recipe.data.options["decode"]["frame_sampling"] == "seeded-random-contiguous"
    assert recipe.data.options["decode"]["frame_sampling_seed"] == 42
    assert recipe.data.options["decode"]["spatial_transform"] == "direct-resize"
    assert recipe.data.options["decode"]["interpolation"] == "bilinear"
    assert recipe.distributed.dp_shard == "auto"


def test_cosmos_dmd2_uses_the_released_pretrained_teacher_checkpoint() -> None:
    checkpoint = cosmos_predict25_dmd2_pretrained_checkpoint()

    assert checkpoint.repo_id == "nvidia/Cosmos-Predict2.5-2B"
    assert checkpoint.revision == COSMOS_PREDICT25_DMD2_PRETRAINED_REVISION
    assert checkpoint.files == (COSMOS_PREDICT25_DMD2_PRETRAINED_WEIGHT_FILE,)


def test_cosmos_dmd2_shifted_uniform_score_levels_match_fixed_rng_formula() -> None:
    recipe = PostTrainingRecipe.from_file(PROFILE)
    config = DMD2Config.from_recipe(recipe.algorithm)
    reference = torch.zeros(3, 1)
    actual = sample_dmd2_score_levels(
        reference,
        config,
        generator=torch.Generator().manual_seed(73),
    )
    uniform = torch.rand((3,), generator=torch.Generator().manual_seed(73))
    expected = 5.0 * uniform.double() / (1.0 + 4.0 * uniform.double())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cosmos_dmd2_cfg_and_proxy_weights_match_author_parameterization() -> None:
    conditional = torch.tensor([2.0, 4.0])
    unconditional = torch.tensor([1.0, -1.0])
    torch.testing.assert_close(
        dmd2_teacher_guidance(conditional, unconditional, 4.0),
        conditional + 3.0 * (conditional - unconditional),
    )

    generated = torch.tensor([[0.5, -0.25]], requires_grad=True)
    field, normalizer = dmd2_distribution_gradient(
        generated.detach(),
        torch.tensor([[0.4, -0.7]]),
        torch.tensor([[0.2, -0.3]]),
        normalization_axes=(1,),
        normalization_epsilon=1.0e-5,
        calculation_dtype="float64",
    )
    assert field.dtype is torch.float64
    assert normalizer.dtype is torch.float64
    (2.0 * dmd2_proxy_loss_per_sample(generated, field, calculation_dtype="float64").mean()).backward()
    torch.testing.assert_close(generated.grad, field.float(), rtol=0, atol=2.0e-8)


def test_cosmos_dmd2_linear_scheduler_matches_released_key_steps() -> None:
    assert cosmos_predict25_dmd2_lr_multiplier(0) == pytest.approx(1.0e-6)
    assert cosmos_predict25_dmd2_lr_multiplier(99) == pytest.approx((0.99 - 1.0e-6) / 100.0 * 99 + 1.0e-6)
    assert cosmos_predict25_dmd2_lr_multiplier(100) == pytest.approx(0.99)
    midpoint = 200_050
    assert cosmos_predict25_dmd2_lr_multiplier(midpoint) == pytest.approx(0.695)

    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW((parameter,), lr=1.0e-6)
    scheduler = _cosmos_predict25_dmd2_scheduler(optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-12)
    assert "last_epoch" in scheduler.state_dict()


def test_cosmos_dmd2_keeps_fp32_master_parameters_and_adamw_state_with_bf16_compute() -> None:
    recipe = PostTrainingRecipe.from_file(PROFILE)
    assert recipe.runtime.param_dtype == "bfloat16"

    student = _adapter(_model().to(dtype=torch.bfloat16))
    fake_score = _adapter(_model().to(dtype=torch.bfloat16))
    apply_cosmos_tuning(recipe, student)
    apply_cosmos_tuning(recipe, fake_score)
    _prepare_trainable_backbone(student)
    _prepare_trainable_backbone(fake_score)

    discriminator = CosmosDMD2DiscriminatorHead(24, 2).float()
    fake_prediction = CosmosFlowDMD2PredictionAdapter(
        fake_score,
        checkpoint_identity="guidance",
        autocast_dtype=torch.bfloat16,
    )
    guidance = CosmosDMD2GuidanceAdapter(
        fake_prediction,
        checkpoint_identity="guidance",
        discriminator=discriminator,
        intermediate_feature_ids=(0, 1),
        trigflow_denoising_weight=True,
    )
    student_parameters = tuple(
        parameter for parameter in student.trainable_module.parameters() if parameter.requires_grad
    )
    guidance_parameters = tuple(parameter for parameter in guidance.module.parameters() if parameter.requires_grad)
    assert student_parameters and guidance_parameters
    assert all(parameter.dtype is torch.float32 for parameter in student_parameters)
    assert all(parameter.dtype is torch.float32 for parameter in guidance_parameters)
    assert fake_prediction.autocast_dtype is torch.bfloat16
    assert guidance.autocast_dtype is torch.bfloat16

    student_optimizer = torch.optim.AdamW(student_parameters, lr=1.0e-6)
    guidance_optimizer = torch.optim.AdamW(guidance_parameters, lr=2.0e-7)
    for optimizer, parameters in (
        (student_optimizer, student_parameters),
        (guidance_optimizer, guidance_parameters),
    ):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for parameter in parameters:
            state = optimizer.state[parameter]
            assert state["exp_avg"].dtype is torch.float32
            assert state["exp_avg_sq"].dtype is torch.float32

    _, logits = guidance.predict_clean_and_logits(
        torch.randn(2, 2, 1, 2, 2),
        torch.tensor([0.2, 0.7]),
        sample_ids=("left", "right"),
        conditioning=_conditioning(2),
        training=True,
    )
    assert logits.dtype is torch.bfloat16


def test_cosmos_dmd2_data_bridge_uses_positive_and_negative_context() -> None:
    clean = torch.randn(2, 2, 3, 1, 1)
    positive = torch.randn(2, 4, 6)
    negative = torch.randn(2, 4, 6)
    batch = cosmos_predict25_dmd2_batch(
        PreparedBatch(
            sample_ids=("left", "right"),
            clean_latents=clean,
            conditioning={"context": positive, "negative_context": negative},
        ),
        conditional_frame_probabilities=(0.0, 0.0, 1.0),
        seed=5,
    )
    torch.testing.assert_close(batch.conditioning["context"], positive)
    torch.testing.assert_close(batch.unconditional_conditioning["context"], negative)
    indicator = batch.conditioning["condition_indicator"]
    assert isinstance(indicator, torch.Tensor)
    assert bool(indicator[:, :, :2].all()) and not bool(indicator[:, :, 2:].any())


def test_cosmos_fused_guidance_shares_one_fake_backbone_forward() -> None:
    torch.manual_seed(83)
    student_model = _model()
    teacher_model = _model().requires_grad_(False)
    fake_model = _model()
    student = CosmosFlowDMD2PredictionAdapter(_adapter(student_model), checkpoint_identity="student")
    teacher = CosmosFlowDMD2PredictionAdapter(_adapter(teacher_model), checkpoint_identity="teacher")
    fake = CosmosFlowDMD2PredictionAdapter(_adapter(fake_model), checkpoint_identity="guidance")
    guidance = CosmosDMD2GuidanceAdapter(
        fake,
        checkpoint_identity="guidance",
        discriminator=CosmosDMD2DiscriminatorHead(24, 2),
        intermediate_feature_ids=(0, 1),
        trigflow_denoising_weight=True,
    )
    config = DMD2Config(
        schedule=FewStepSchedule((math.pi / 2,), (1.0,)),
        normalization_axes=(1, 2, 3, 4),
        score_min_sigma=0.0,
        score_max_sigma=1.0,
        score_flow_shift=5.0,
        teacher_guidance_scale=4.0,
        normalization_epsilon=1.0e-5,
        score_sampling="continuous",
        normalization_reference="generated-clean",
        shared_adversarial_score_input=True,
        distribution_matching_weight=2.0,
    )
    losses = NativeDMD2LossAdapter(student, teacher, guidance, config)
    calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = fake_model.register_forward_hook(count_forward)
    try:
        losses.generator_loss(_batch(), generator=torch.Generator().manual_seed(89))
        assert calls == 1
        losses.guidance_loss(_batch(), generator=torch.Generator().manual_seed(97))
        assert calls == 3
    finally:
        handle.remove()


def test_cosmos_fused_fake_score_loss_equals_author_trigflow_x0_weight() -> None:
    torch.manual_seed(101)
    model = _model()
    prediction = CosmosFlowDMD2PredictionAdapter(_adapter(model), checkpoint_identity="guidance")
    guidance = CosmosDMD2GuidanceAdapter(
        prediction,
        checkpoint_identity="guidance",
        discriminator=CosmosDMD2DiscriminatorHead(24, 2),
        intermediate_feature_ids=(0, 1),
        trigflow_denoising_weight=True,
    )
    clean = torch.randn(2, 2, 1, 2, 2)
    noise = torch.randn_like(clean)
    levels = torch.tensor([0.2, 0.7])
    noisy = prediction.add_noise(clean, noise, levels)
    conditioning = _conditioning(2)
    predicted_clean, _ = guidance.predict_clean_and_logits(
        noisy,
        levels,
        sample_ids=("left", "right"),
        conditioning=conditioning,
        training=True,
    )
    actual = guidance.denoising_loss_from_clean_per_sample(
        clean,
        predicted_clean,
        levels,
        conditioning=conditioning,
    )
    trig_time = torch.atan(levels / (1.0 - levels))
    expected = (
        ((clean - predicted_clean).float().square() / torch.sin(trig_time).square()[:, None, None, None, None])
        .flatten(1)
        .mean(1)
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_post_train_cli_dispatches_cosmos_dmd2(monkeypatch, tmp_path, capsys) -> None:
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    output_dir = tmp_path / "run"
    args = parser.parse_args(
        [
            "post-train",
            "--recipe",
            str(PROFILE),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--no-export-trained-artifact",
        ]
    )
    called: dict[str, object] = {}

    @dataclass(frozen=True)
    class Summary:
        initial_step: int = 0
        final_step: int = 2
        iterations: int = 2
        student_optimizer_steps: int = 1
        guidance_optimizer_steps: int = 1
        final_generator_loss: float = 0.0
        final_guidance_loss: float = 0.5

    class Run:
        world_size = 1
        is_coordinator = True

        def __init__(self) -> None:
            self.output_dir = output_dir

        def run(self, *, max_steps):
            called["max_steps"] = max_steps
            return Summary()

        def close(self):
            called["closed"] = True

    def materialize(recipe, **kwargs):
        called["algorithm"] = recipe.algorithm.type
        called.update(kwargs)
        return Run()

    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.training.engine.cosmos.dmd2",
        SimpleNamespace(materialize_cosmos_predict25_dmd2_training_run=materialize),
    )
    assert args.func(args) == 0
    assert called["algorithm"] == "dmd2"
    assert called["max_steps"] == 2
    assert called["closed"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"] == "dmd2"
    assert payload["summary"]["guidance_optimizer_steps"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 gate requires CUDA")
def test_cosmos_dmd2_fsdp_wraps_trainable_modules_before_optimizer(tmp_path: Path) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from worldfoundry.training.distributed.fsdp import apply_fsdp2
    from worldfoundry.training.distributed.parallel import ParallelPlan

    if dist.is_initialized():
        pytest.skip("test owns a world-size-one process group")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'process-group'}",
        rank=0,
        world_size=1,
    )
    try:
        recipe = PostTrainingRecipe.from_file(PROFILE)
        plan = ParallelPlan.resolve(recipe.distributed, world_size=1)
        mesh = plan.build_device_mesh("cuda")
        student = _adapter(_model().to(device="cuda", dtype=torch.bfloat16))
        application = apply_fsdp2(
            student,
            plan=plan,
            mesh=mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in student.trainable_module.parameters() if parameter.requires_grad),
            lr=1.0e-6,
        )
        fake = CosmosFlowDMD2PredictionAdapter(
            _adapter(_model().to(device="cuda", dtype=torch.bfloat16)),
            checkpoint_identity="guidance",
            autocast_dtype=torch.bfloat16,
        )
        guidance = CosmosDMD2GuidanceAdapter(
            fake,
            checkpoint_identity="guidance",
            discriminator=CosmosDMD2DiscriminatorHead(24, 2).to(device="cuda", dtype=torch.bfloat16),
            intermediate_feature_ids=(0, 1),
            trigflow_denoising_weight=True,
        )
        guidance_application = apply_fsdp2(
            guidance,
            plan=plan,
            mesh=mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        guidance_optimizer = torch.optim.AdamW(
            (parameter for parameter in guidance.module.parameters() if parameter.requires_grad),
            lr=2.0e-7,
        )
        assert application.parameter_mode == "trainable"
        assert guidance_application.parameter_mode == "trainable"
        assert all(isinstance(parameter, DTensor) for parameter in student.trainable_module.parameters())
        assert all(isinstance(parameter, DTensor) for group in optimizer.param_groups for parameter in group["params"])
        assert all(isinstance(parameter, DTensor) for parameter in guidance.module.parameters())
        assert all(
            isinstance(parameter, DTensor) for group in guidance_optimizer.param_groups for parameter in group["params"]
        )
    finally:
        dist.destroy_process_group()
