from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.diffusion_model.recipes.registry import (  # noqa: E402
    default_native_diffusion_registry,
)
from worldfoundry.cli.training_commands.common import (  # noqa: E402
    training_checkpoint_key,
    training_family,
)
from worldfoundry.training.engine.cosmos.sft import (  # noqa: E402
    cosmos_training_checkpoint_overrides,
)
from worldfoundry.training.engine.ltx.sft import build_ltx_flow_objective  # noqa: E402
from worldfoundry.training.objectives.flow_matching import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.optimization import (  # noqa: E402
    build_lr_scheduler,
    warmup_cosine_multiplier,
)
from worldfoundry.training.recipes import TrainingRecipe  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def _recipe(name: str) -> TrainingRecipe:
    return TrainingRecipe.from_file(_ROOT / "configs" / "training" / name)


def test_ltx_training_profile_uses_the_development_checkpoint_and_author_sampler() -> None:
    recipe = _recipe("ltx_2p3_video_lora.yaml")

    assert training_family(recipe.model.recipe) == "ltx"
    assert training_checkpoint_key(recipe.model.recipe) == "model"
    assert recipe.model.checkpoint.endswith("ltx-2.3-22b-dev.safetensors")
    assert recipe.tuning.preset == "ltx-attention"
    objective = build_ltx_flow_objective(recipe)
    assert objective.config.mode == "shifted-logit-normal"
    assert objective.config.stretch is True
    assert recipe.objective.options["first_frame_conditioning_probability"] == 0.5
    assert recipe.optimizer.weight_decay == pytest.approx(0.01)
    assert recipe.scheduler is not None
    assert recipe.scheduler.type == "linear"
    assert recipe.scheduler.total_steps == 2_000
    assert recipe.scheduler.start_factor == 1.0
    assert recipe.scheduler.end_factor == 0.1
    assert recipe.checkpoint.save_every_steps == 250
    assert recipe.distributed.dp_shard == "auto"
    assert recipe.data.options["decode"]["frame_sampling"] == "head"
    assert recipe.data.options["decode"]["resize_rounding"] == "floor"

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    reference_parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW((parameter,), lr=recipe.optimizer.learning_rate)
    reference_optimizer = torch.optim.AdamW(
        (reference_parameter,),
        lr=recipe.optimizer.learning_rate,
    )
    scheduler = build_lr_scheduler(optimizer, recipe.scheduler)
    reference_scheduler = torch.optim.lr_scheduler.LinearLR(
        reference_optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=2_000,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
    assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()
    for _ in range(2_001):
        optimizer.step()
        reference_optimizer.step()
        scheduler.step()
        reference_scheduler.step()
        assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()


def test_legacy_ltx_video_profile_matches_released_2b_lora_config() -> None:
    recipe = _recipe("ltx_video_2b_lora.yaml")

    assert recipe.model.recipe == "ltx-video-i2v"
    assert recipe.model.checkpoint.endswith("ltxv-2b-0.9.6-dev-04-25.safetensors")
    assert recipe.tuning.rank == 32
    assert recipe.tuning.alpha == 32
    assert recipe.objective.options["first_frame_conditioning_probability"] == pytest.approx(0.1)
    assert recipe.optimizer.learning_rate == pytest.approx(2.0e-4)
    assert recipe.optimizer.gradient_accumulation_steps == 1
    assert recipe.scheduler is not None
    assert recipe.scheduler.type == "linear"
    assert recipe.scheduler.total_steps == 1500
    assert recipe.runtime.param_dtype == "bfloat16"
    assert recipe.runtime.activation_checkpoint == "none"
    assert recipe.checkpoint.save_every_steps == 250
    assert recipe.data.options["decode"]["frame_sampling"] == "head"
    assert recipe.data.options["decode"]["resize_rounding"] == "floor"
    bucket = recipe.data.options["video_buckets"][0]
    assert (bucket["num_frames"], bucket["height"], bucket["width"]) == (89, 448, 768)
    assert recipe.data.max_latent_tokens_per_microbatch == 12 * 14 * 24


def test_cosmos_predict_training_profile_keeps_released_loss_and_condition_mix() -> None:
    recipe = _recipe("cosmos_predict2p5_2b_video_lora.yaml")

    assert training_family(recipe.model.recipe) == "cosmos"
    assert training_checkpoint_key(recipe.model.recipe) == "transformer"
    assert recipe.tuning.preset == "cosmos-predict-attention-mlp"
    assert recipe.objective.timestep_sampler == "logit-normal"
    assert recipe.objective.options["flow_shift"] == 5.0
    assert "num_train_timesteps" not in recipe.objective.options
    assert "loss_scale" not in recipe.objective.options
    assert recipe.objective.options["conditional_frame_probabilities"] == {
        0: 0.333,
        1: 0.333,
        2: 0.334,
    }
    bucket = recipe.data.options["video_buckets"][0]
    assert (bucket["num_frames"], bucket["height"], bucket["width"]) == (93, 704, 1280)
    assert recipe.data.max_latent_tokens_per_microbatch == 24 * 88 * 160
    assert recipe.data.options["decode"] == {
        "frame_sampling": "seeded-random-contiguous",
        "frame_sampling_seed": 42,
        "spatial_transform": "direct-resize",
        "interpolation": "bilinear",
        "value_range": "minus-one-one",
        "decoder_thread_type": "auto",
        "verify_manifest_frame_count": True,
        "verify_manifest_geometry": True,
        "fps_tolerance": 0.05,
    }
    assert recipe.data.tail_policy == "drop"
    assert recipe.data.options["num_workers"] == 4
    assert recipe.checkpoint.save_every_steps == 200
    assert recipe.distributed.dp_shard == "auto"
    assert recipe.objective.conditioning_dropout == 0.2
    assert recipe.optimizer.learning_rate == pytest.approx(2 ** (-14.5))
    assert recipe.optimizer.weight_decay == 0.001
    assert recipe.scheduler is not None
    assert recipe.scheduler.type == "warmup-cosine"
    assert recipe.scheduler.warmup_steps == 2_000
    assert recipe.scheduler.total_steps == 100_000
    assert recipe.scheduler.start_factor == 0.0
    assert recipe.scheduler.peak_factor == 0.5
    assert recipe.scheduler.end_factor == 0.2
    assert warmup_cosine_multiplier(0, recipe.scheduler) == 0.0
    assert warmup_cosine_multiplier(1_000, recipe.scheduler) == pytest.approx(0.25)
    assert warmup_cosine_multiplier(2_000, recipe.scheduler) == pytest.approx(0.5)
    assert warmup_cosine_multiplier(51_000, recipe.scheduler) == pytest.approx(0.35)
    assert warmup_cosine_multiplier(100_000, recipe.scheduler) == pytest.approx(0.2)

    native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
    overrides = cosmos_training_checkpoint_overrides(
        recipe,
        native_recipe,
        None,
        base_dir=_ROOT,
    )
    selected = overrides["transformer"]
    assert selected is native_recipe.checkpoints["transformer-pretrained"]
    assert selected.repo_id == "nvidia/Cosmos-Predict2.5-2B"
    assert selected.files == ("base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",)

    flow_options = dict(recipe.objective.options)
    flow_options.pop("conditional_frame_probabilities")
    objective = FlowMatchingObjective(
        FlowMatchingConfig(timestep_sampler=recipe.objective.timestep_sampler, **flow_options)
    )
    generator = torch.Generator().manual_seed(67)
    actual = objective.sample_sigmas(4, device=torch.device("cpu"), generator=generator)
    unit = torch.sigmoid(torch.randn(4, generator=torch.Generator().manual_seed(67)))
    expected = 5.0 * unit / (1.0 + 4.0 * unit)
    torch.testing.assert_close(actual, expected)


def test_cosmos3_nano_profile_matches_released_vision_sft_core() -> None:
    recipe = _recipe("cosmos3_nano_vision_sft.yaml")

    bucket = recipe.data.options["video_buckets"][0]
    assert bucket["tasks"] == ["t2v", "i2v", "v2v"]
    assert recipe.model.options["use_system_prompt"] is False
    assert recipe.tuning.mode == "partial"
    assert recipe.tuning.preset == "cosmos3-nano-vision-sft"
    assert recipe.objective.conditioning_dropout == pytest.approx(0.1)
    assert recipe.objective.options["conditioning_config"] == {0: 0.7, 1: 0.2, 2: 0.1}
    assert recipe.data.max_latent_tokens_per_microbatch == 13 * 16 * 16
    assert recipe.optimizer.learning_rate == pytest.approx(1.0e-4)
    assert recipe.optimizer.betas == pytest.approx((0.9, 0.95))
    assert recipe.optimizer.epsilon == pytest.approx(1.0e-6)
    assert recipe.optimizer.max_grad_norm == pytest.approx(0.1)
    assert recipe.optimizer.gradient_accumulation_steps == 2
    assert recipe.scheduler is not None
    assert recipe.scheduler.type == "warmup-cosine"
    assert recipe.scheduler.warmup_steps == 50
    assert recipe.scheduler.total_steps == 1000
    assert warmup_cosine_multiplier(0, recipe.scheduler) == 0.0
    assert warmup_cosine_multiplier(50, recipe.scheduler) == 1.0
    assert warmup_cosine_multiplier(1000, recipe.scheduler) == 0.0
    assert recipe.runtime.activation_checkpoint == "full"
    assert recipe.runtime.compile is False
    assert recipe.distributed.dp_shard == "auto"
