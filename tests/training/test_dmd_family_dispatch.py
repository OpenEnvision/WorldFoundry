from __future__ import annotations

import pytest

from worldfoundry.cli.training_commands.common import training_family
from worldfoundry.cli.training_commands.handlers.post_train import (
    _DMD_STRATEGY_FACTORIES,
    _resolve_dmd_strategy,
)


def test_training_family_maps_new_flow_velocity_recipes() -> None:
    assert training_family("wan2.1-t2v-1.3b") == "wan"
    assert training_family("wan2.1-vace") == "wan_vace"
    assert training_family("Wan-VACE") == "wan_vace"
    assert training_family("step-video-t2v") == "step_video"
    assert training_family("stepvideo") == "step_video"
    assert training_family("vchitect-2-t2v") == "vchitect"
    assert training_family("sana-600m-512") == "sana"


def test_training_family_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="does not support"):
        training_family("some-unregistered-model")


def test_resolve_dmd_strategy_returns_wan_strategy() -> None:
    pytest.importorskip("torch")
    try:
        strategy = _resolve_dmd_strategy("wan")
    except ImportError as error:
        # The Wan strategy transitively imports the FSDP2 engine; skip when the
        # installed torch lacks the composable FSDP2 API (environment version
        # skew, unrelated to the dispatch logic under test).
        pytest.skip(f"FSDP2 engine unavailable in this torch build: {error}")
    assert strategy.family == "wan"


def test_resolve_dmd_strategy_rejects_family_without_cache_path() -> None:
    # A flow-velocity training adapter alone is not a DMD strategy: the family
    # additionally needs its precompute cache contract and run container.
    assert "step_video" not in _DMD_STRATEGY_FACTORIES
    with pytest.raises(ValueError, match="no strategy for family 'step_video'"):
        _resolve_dmd_strategy("step_video")


def test_wan_dmd_strategy_validates_recipe_id() -> None:
    pytest.importorskip("torch")
    try:
        from worldfoundry.training.engine.wan.dmd import WanDMDStrategy
    except ImportError as error:
        pytest.skip(f"FSDP2 engine unavailable in this torch build: {error}")
    from worldfoundry.training.recipes import PostTrainingRecipe

    mapping = {
        "schema": "worldfoundry-post-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "dmd-test", "output_dir": "runs/dmd-test"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "model-digest"},
        "tuning": {"mode": "full"},
        "export": {"format": "safetensors"},
        "data": {
            "manifest": "data/train.jsonl",
            "cache": "data/cache",
            "shuffle": False,
            "max_latent_tokens_per_microbatch": 4096,
            "tail_policy": "drop",
        },
        "algorithm": {
            "type": "dmd",
            "student_timesteps": [1000, 750, 500],
            "student_sigmas": [1.0, 0.75, 0.5],
            "real_score_checkpoint": "teacher-digest",
            "fake_score_checkpoint": "critic-digest",
        },
        "optimizer": {"type": "adamw", "learning_rate": 2.0e-6, "max_grad_norm": 0.7},
        "fake_score_optimizer": {"type": "adamw", "learning_rate": 4.0e-6, "max_grad_norm": 0.9},
        "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
        "distributed": {"backend": "single"},
    }
    strategy = WanDMDStrategy()
    recipe = PostTrainingRecipe.from_mapping(mapping)
    assert strategy.validate_recipe(recipe).type == "dmd"

    bad = dict(mapping)
    bad["model"] = {"recipe": "step-video-t2v", "checkpoint": "model-digest"}
    with pytest.raises(ValueError, match="wan2.1-t2v-1.3b"):
        strategy.validate_recipe(PostTrainingRecipe.from_mapping(bad))
