from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from collections.abc import Callable
from pathlib import Path

import pytest

from worldfoundry.training.data.loader import build_stateful_dataloader
from worldfoundry.training.data.sampler import (
    DeterministicDistributedSampler,
    SamplerStateMismatchError,
)
from worldfoundry.training.engine.hunyuan_video import (
    validate_hunyuan_video_flow_policy_recipe,
)
from worldfoundry.training.engine.ltx import validate_ltx_flow_policy_recipe
from worldfoundry.training.engine.video_policy import (
    materialize_video_flow_policy_training_run,
    resolve_video_flow_policy_prompt_batch_size,
    validate_video_flow_policy_prompt_population,
)
from worldfoundry.training.engine.wan22 import validate_wan22_flow_policy_recipe
from worldfoundry.training.recipes import PostTrainingRecipe

_ROOT = Path(__file__).resolve().parents[2]


class _Dataset:
    def __init__(self, size: int) -> None:
        self.values = tuple(range(size))
        self.sample_ids = tuple(f"prompt-{index}" for index in self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


def _recipe(filename: str) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_file(_ROOT / "configs" / "post_training" / filename)


@pytest.mark.parametrize(
    ("filename", "global_prompt_batch_size"),
    (
        ("wan22_t2v_a14b_flow_grpo.yaml", 48),
        ("hunyuan_video_flow_grpo.yaml", 8),
        ("hunyuan_video_1p5_flow_grpo.yaml", 48),
        ("ltx_2_video_flow_grpo.yaml", 8),
        ("ltx_2p3_video_flow_grpo.yaml", 8),
        ("ltx_2p3_av_flow_grpo.yaml", 8),
    ),
)
def test_video_flow_policy_presets_declare_exact_global_prompt_batches(
    filename: str,
    global_prompt_batch_size: int,
) -> None:
    recipe = _recipe(filename)

    assert recipe.data.options["global_prompt_batch_size"] == global_prompt_batch_size
    assert resolve_video_flow_policy_prompt_batch_size(recipe, world_size=1) == global_prompt_batch_size


@pytest.mark.parametrize(
    ("filename", "world_size", "local_prompt_batch_size"),
    (
        ("wan22_t2v_a14b_flow_grpo.yaml", 1, 48),
        ("wan22_t2v_a14b_flow_grpo.yaml", 2, 24),
        ("wan22_t2v_a14b_flow_grpo.yaml", 3, 16),
        ("wan22_t2v_a14b_flow_grpo.yaml", 12, 4),
        ("hunyuan_video_flow_grpo.yaml", 1, 8),
        ("hunyuan_video_flow_grpo.yaml", 2, 4),
        ("hunyuan_video_flow_grpo.yaml", 4, 2),
        ("hunyuan_video_flow_grpo.yaml", 8, 1),
    ),
)
def test_video_flow_policy_global_batch_scales_over_compatible_world_sizes(
    filename: str,
    world_size: int,
    local_prompt_batch_size: int,
) -> None:
    recipe = _recipe(filename)

    assert resolve_video_flow_policy_prompt_batch_size(recipe, world_size=world_size) == local_prompt_batch_size


def test_video_flow_policy_global_batch_can_be_overridden_for_the_active_topology() -> None:
    mapping = _recipe("hunyuan_video_flow_grpo.yaml").to_dict()
    mapping["data"]["options"]["global_prompt_batch_size"] = 12
    recipe = PostTrainingRecipe.from_mapping(mapping)

    assert resolve_video_flow_policy_prompt_batch_size(recipe, world_size=3) == 4
    assert resolve_video_flow_policy_prompt_batch_size(recipe, world_size=6) == 2


@pytest.mark.parametrize(
    ("filename", "validate"),
    (
        ("wan22_t2v_a14b_flow_grpo.yaml", validate_wan22_flow_policy_recipe),
        ("hunyuan_video_flow_grpo.yaml", validate_hunyuan_video_flow_policy_recipe),
        ("ltx_2_video_flow_grpo.yaml", validate_ltx_flow_policy_recipe),
    ),
)
def test_video_family_plans_reject_invalid_global_prompt_batches(
    filename: str,
    validate: Callable[[PostTrainingRecipe], object],
) -> None:
    mapping = _recipe(filename).to_dict()
    mapping["data"]["options"]["global_prompt_batch_size"] = 0

    with pytest.raises(ValueError, match="global_prompt_batch_size"):
        validate(PostTrainingRecipe.from_mapping(mapping))


def test_video_flow_policy_rejects_rank_local_prompt_batch_configuration() -> None:
    mapping = _recipe("wan22_t2v_a14b_flow_grpo.yaml").to_dict()
    mapping["data"]["options"]["prompt_batch_size"] = 1

    with pytest.raises(ValueError, match="not rank-local prompt_batch_size"):
        resolve_video_flow_policy_prompt_batch_size(
            PostTrainingRecipe.from_mapping(mapping),
            world_size=2,
        )


@pytest.mark.parametrize(
    ("filename", "world_size"),
    (
        ("hunyuan_video_flow_grpo.yaml", 3),
        ("wan22_t2v_a14b_flow_grpo.yaml", 5),
    ),
)
def test_video_flow_policy_rejects_non_divisible_global_batches_before_loading(
    filename: str,
    world_size: int,
) -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_video_flow_policy_prompt_batch_size(
            _recipe(filename),
            world_size=world_size,
        )


def test_video_flow_policy_rejects_uneven_epoch_shards() -> None:
    mapping = _recipe("hunyuan_video_flow_grpo.yaml").to_dict()
    mapping["data"]["tail_policy"] = "uneven"

    with pytest.raises(ValueError, match="tail_policy='drop' or 'pad'"):
        resolve_video_flow_policy_prompt_batch_size(
            PostTrainingRecipe.from_mapping(mapping),
            world_size=2,
        )


@pytest.mark.parametrize("tail_policy", ("drop", "pad"))
@pytest.mark.parametrize("world_size", (1, 2, 3, 6))
def test_video_flow_policy_requires_one_complete_global_prompt_batch(
    tail_policy: str,
    world_size: int,
) -> None:
    mapping = _recipe("hunyuan_video_flow_grpo.yaml").to_dict()
    mapping["data"]["tail_policy"] = tail_policy
    mapping["data"]["options"]["global_prompt_batch_size"] = 12
    recipe = PostTrainingRecipe.from_mapping(mapping)

    with pytest.raises(ValueError, match="at least one complete global prompt batch"):
        validate_video_flow_policy_prompt_population(
            recipe,
            prompt_count=11,
            world_size=world_size,
        )


@pytest.mark.parametrize("tail_policy", ("drop", "pad"))
@pytest.mark.parametrize("world_size", (1, 2, 3, 6))
def test_valid_prompt_population_never_duplicates_ids_within_a_rank_batch(
    tail_policy: str,
    world_size: int,
) -> None:
    mapping = _recipe("hunyuan_video_flow_grpo.yaml").to_dict()
    mapping["data"]["tail_policy"] = tail_policy
    mapping["data"]["options"]["global_prompt_batch_size"] = 12
    recipe = PostTrainingRecipe.from_mapping(mapping)
    dataset = _Dataset(17)
    local_batch_size = validate_video_flow_policy_prompt_population(
        recipe,
        prompt_count=len(dataset),
        world_size=world_size,
    )

    for rank in range(world_size):
        indices = tuple(
            DeterministicDistributedSampler(
                dataset,
                shuffle=False,
                rank=rank,
                world_size=world_size,
                tail_policy=tail_policy,
                local_batch_size=local_batch_size,
            )
        )
        for start in range(0, len(indices), local_batch_size):
            batch = indices[start : start + local_batch_size]
            assert len(batch) == local_batch_size
            assert len(set(batch)) == len(batch)


def test_insufficient_prompt_population_fails_before_model_materialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import worldfoundry.training.engine.video_policy as video_policy

    mapping = _recipe("hunyuan_video_flow_grpo.yaml").to_dict()
    mapping["data"]["options"]["global_prompt_batch_size"] = 8
    mapping["distributed"] = {"backend": "single"}
    mapping["run"]["output_dir"] = str(tmp_path / "run")
    recipe = PostTrainingRecipe.from_mapping(mapping)
    model_materialized = False

    def load_conditioning(*args, **kwargs):
        del args, kwargs
        return object(), _Dataset(7)

    def materialize_roles(*args, **kwargs):
        del args, kwargs
        nonlocal model_materialized
        model_materialized = True
        raise AssertionError("model loading must not start")

    class Evaluator:
        def evaluate(self, requests):
            del requests
            return ()

    monkeypatch.setattr(video_policy, "_load_conditioning_dataset", load_conditioning)
    monkeypatch.setattr(video_policy, "materialize_video_flow_policy_roles", materialize_roles)

    with pytest.raises(ValueError, match="at least one complete global prompt batch"):
        materialize_video_flow_policy_training_run(
            recipe,
            base_dir=tmp_path,
            device="cpu",
            reward_evaluator=Evaluator(),
        )

    assert model_materialized is False
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("tail_policy", "expected_steps"),
    (("drop", 1), ("pad", 2)),
)
def test_batch_aligned_sampler_emits_exact_synchronized_global_rollouts(
    tail_policy: str,
    expected_steps: int,
) -> None:
    dataset = _Dataset(53)
    world_size = 3
    global_prompt_batch_size = 48
    local_prompt_batch_size = global_prompt_batch_size // world_size
    rank_batches = []
    for rank in range(world_size):
        sampler = DeterministicDistributedSampler(
            dataset,
            shuffle=False,
            rank=rank,
            world_size=world_size,
            tail_policy=tail_policy,
            local_batch_size=local_prompt_batch_size,
        )
        indices = list(sampler)
        rank_batches.append(
            tuple(
                indices[start : start + local_prompt_batch_size]
                for start in range(0, len(indices), local_prompt_batch_size)
            )
        )

    assert {len(batches) for batches in rank_batches} == {expected_steps}
    for logical_rollout in zip(*rank_batches, strict=True):
        assert {len(local_batch) for local_batch in logical_rollout} == {local_prompt_batch_size}
        assert sum(map(len, logical_rollout)) == global_prompt_batch_size


def test_batch_aligned_stateful_loader_resumes_at_the_next_complete_rollout() -> None:
    pytest.importorskip("torchdata")
    dataset = _Dataset(53)

    def build():
        sampler = DeterministicDistributedSampler(
            dataset,
            seed=31,
            shuffle=True,
            rank=1,
            world_size=3,
            tail_policy="pad",
            local_batch_size=16,
        )
        return build_stateful_dataloader(dataset, sampler, batch_size=16)

    loader = build()
    iterator = iter(loader)
    first = next(iterator).tolist()
    state = loader.state_dict()
    expected_tail = [batch.tolist() for batch in iterator]

    restored = build()
    restored.load_state_dict(state)

    assert len(first) == 16
    assert all(len(batch) == 16 for batch in expected_tail)
    assert [batch.tolist() for batch in restored] == expected_tail


def test_sampler_resume_binds_the_local_batch_alignment() -> None:
    dataset = _Dataset(53)
    aligned = DeterministicDistributedSampler(
        dataset,
        rank=0,
        world_size=3,
        tail_policy="pad",
        local_batch_size=16,
    )
    state = aligned.state_dict()

    with pytest.raises(SamplerStateMismatchError, match="local_batch_size"):
        DeterministicDistributedSampler(
            dataset,
            rank=0,
            world_size=3,
            tail_policy="pad",
            local_batch_size=8,
        ).load_state_dict(state)


def test_sampler_reads_pre_alignment_state_as_the_default_single_sample_batch() -> None:
    dataset = _Dataset(7)
    sampler = DeterministicDistributedSampler(
        dataset,
        rank=0,
        world_size=1,
        tail_policy="pad",
    )
    state = sampler.state_dict()
    state.pop("local_batch_size")

    restored = DeterministicDistributedSampler(
        dataset,
        rank=0,
        world_size=1,
        tail_policy="pad",
    )
    restored.load_state_dict(state)

    assert restored.state_dict()["local_batch_size"] == 1
