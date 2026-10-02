from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.training.recipes import TRAINING_RECIPE_SCHEMA, TrainingRecipe


def _recipe_mapping() -> dict:
    return {
        "schema": TRAINING_RECIPE_SCHEMA,
        "execution_owner": "worldfoundry-native",
        "run": {"id": "SANA_DEMO", "output_dir": "runs/sana-demo"},
        "model": {"recipe": "sana-600m-512px", "checkpoint": "default"},
        "tuning": {
            "mode": "lora",
            "preset": "sana-attention",
            "rank": 8,
            "alpha": 16,
        },
        "data": {
            "manifest": "data/train.jsonl",
            "split": "train",
            "shuffle": True,
            "shuffle_seed": 7,
            "tail_policy": "drop",
        },
        "objective": {"type": "flow_matching", "timestep_sampler": "logit_normal"},
        "optimizer": {"type": "adamw", "learning_rate": 1.0e-4},
        "scheduler": {
            "type": "warmup_cosine",
            "warmup_steps": 100,
            "total_steps": 1000,
            "start_factor": 0.0,
            "peak_factor": 0.5,
            "end_factor": 0.2,
        },
        "distributed": {
            "backend": "fsdp2",
            "dp_shard": "auto",
            "cp": 1,
            "tp": 1,
            "timeout_seconds": 73,
        },
        "checkpoint": {"save_every_steps": 10, "async": True},
    }


def test_training_recipe_is_canonical_and_round_trips() -> None:
    recipe = TrainingRecipe.from_mapping(_recipe_mapping())
    restored = TrainingRecipe.from_mapping(recipe.to_dict())

    assert recipe.run.id == "sana-demo"
    assert recipe.objective.type == "flow-matching"
    assert recipe.data.tail_policy == "drop"
    assert recipe.scheduler is not None
    assert recipe.scheduler.type == "warmup-cosine"
    assert recipe.scheduler.warmup_steps == 100
    assert recipe.distributed.timeout_seconds == 73
    assert restored == recipe


@pytest.mark.parametrize("timeout", (0, -1, True))
def test_training_recipe_rejects_invalid_distributed_timeout(timeout: object) -> None:
    payload = _recipe_mapping()
    payload["distributed"]["timeout_seconds"] = timeout

    with pytest.raises((TypeError, ValueError), match="distributed.timeout_seconds"):
        TrainingRecipe.from_mapping(payload)


def test_training_recipe_rejects_unknown_fields() -> None:
    payload = _recipe_mapping()
    payload["optimizer"]["typo_learning_rate"] = 1.0

    with pytest.raises(ValueError, match="unknown fields"):
        TrainingRecipe.from_mapping(payload)


def test_training_recipe_rejects_string_booleans() -> None:
    payload = _recipe_mapping()
    payload["data"]["shuffle"] = "false"

    with pytest.raises(TypeError, match="data.shuffle must be a bool"):
        TrainingRecipe.from_mapping(payload)


def test_training_recipe_canonicalizes_prediction_type_to_contract_spelling() -> None:
    payload = _recipe_mapping()
    payload["objective"]["prediction_type"] = "flow-velocity"

    recipe = TrainingRecipe.from_mapping(payload)

    assert recipe.objective.prediction_type == "flow_velocity"
    assert recipe.to_dict()["objective"]["prediction_type"] == "flow_velocity"


def test_training_recipe_can_disable_gradient_clipping() -> None:
    payload = _recipe_mapping()
    payload["optimizer"]["max_grad_norm"] = None

    recipe = TrainingRecipe.from_mapping(payload)

    assert recipe.optimizer.max_grad_norm is None
    assert TrainingRecipe.from_mapping(recipe.to_dict()) == recipe


def test_training_recipe_rejects_external_execution_fields() -> None:
    payload = _recipe_mapping()
    payload["provider"] = {
        "name": "unirl",
    }

    with pytest.raises(ValueError, match="unknown fields.*provider"):
        TrainingRecipe.from_mapping(payload)

    payload.pop("provider")
    payload["execution_owner"] = "unirl"
    with pytest.raises(ValueError, match="external training loops are unsupported"):
        TrainingRecipe.from_mapping(payload)


def test_checked_in_sana_recipe_parses() -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]

    recipe = TrainingRecipe.from_file(root / "configs/training/sana_600m_flow_lora.yaml")

    assert recipe.schema == TRAINING_RECIPE_SCHEMA
    assert recipe.model.recipe == "sana-600m-512px"
    assert recipe.checkpoint.async_save is True


@pytest.mark.parametrize(
    ("filename", "backend"),
    (
        ("wan_1p3b_flow_lora_single_device.yaml", "single"),
        ("wan_1p3b_flow_lora.yaml", "fsdp2"),
    ),
)
def test_checked_in_wan_recipes_bind_video_cache_and_training_math(
    filename: str,
    backend: str,
) -> None:
    pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]

    recipe = TrainingRecipe.from_file(root / "configs/training" / filename)

    assert recipe.model.recipe == "wan2.1-t2v-1.3b"
    assert recipe.tuning.preset == "wan-attention"
    assert recipe.distributed.backend == backend
    assert recipe.runtime.activation_checkpoint == "full"
    assert recipe.objective.conditioning_dropout == 0.0
    assert recipe.objective.options["flow_shift"] == 1.0
    assert recipe.objective.options["num_train_timesteps"] == 1000
    assert recipe.data.tail_policy == "pad"
    assert len(recipe.data.options["video_buckets"]) == 3
