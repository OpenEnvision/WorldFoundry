from __future__ import annotations

import json

import pytest

from worldfoundry.training.data import (
    DeterministicDistributedSampler,
    SamplerStateMismatchError,
    distributed_context_from_environment,
)


class _Dataset:
    def __init__(self, size: int) -> None:
        self.size = size
        self.sample_ids = tuple(f"sample-{index}" for index in range(size))

    def __len__(self) -> int:
        return self.size


def test_sampler_is_deterministic_and_disjoint_across_drop_policy_ranks() -> None:
    dataset = _Dataset(12)
    samplers = [
        DeterministicDistributedSampler(dataset, seed=17, rank=rank, world_size=3, tail_policy="drop")
        for rank in range(3)
    ]
    orders = [list(sampler) for sampler in samplers]

    assert all(len(order) == 4 for order in orders)
    assert set().union(*(set(order) for order in orders)) == set(range(12))
    assert sum(len(order) for order in orders) == len(set().union(*(set(order) for order in orders)))
    assert orders[0] == list(
        DeterministicDistributedSampler(dataset, seed=17, rank=0, world_size=3, tail_policy="drop")
    )


def test_sampler_state_round_trip_resumes_at_the_next_unseen_sample() -> None:
    dataset = _Dataset(10)
    sampler = DeterministicDistributedSampler(dataset, seed=5, rank=0, world_size=2)
    iterator = iter(sampler)
    consumed = [next(iterator), next(iterator)]
    state = json.loads(json.dumps(sampler.state_dict()))
    expected_remainder = list(iterator)

    restored = DeterministicDistributedSampler(dataset, seed=5, rank=0, world_size=2)
    restored.load_state_dict(state)

    assert state["position"] == len(consumed)
    assert state["next_sample_id"] == dataset.sample_ids[expected_remainder[0]]
    assert list(restored) == expected_remainder


def test_sampler_rejects_dataset_and_topology_changes_for_exact_resume() -> None:
    state = DeterministicDistributedSampler(_Dataset(8), rank=0, world_size=2).state_dict()

    changed_samples = _Dataset(8)
    changed_samples.sample_ids = tuple(f"other-{index}" for index in range(8))
    with pytest.raises(SamplerStateMismatchError, match="sample_ids"):
        DeterministicDistributedSampler(changed_samples, rank=0, world_size=2).load_state_dict(state)

    with pytest.raises(SamplerStateMismatchError, match="world_size"):
        DeterministicDistributedSampler(_Dataset(8), rank=0, world_size=1).load_state_dict(state)


def test_failed_sampler_restore_does_not_mutate_active_cursor() -> None:
    dataset = _Dataset(8)
    sampler = DeterministicDistributedSampler(dataset, seed=9, rank=0, world_size=2)
    iterator = iter(sampler)
    next(iterator)
    active_state = sampler.state_dict()
    invalid_state = dict(active_state)
    invalid_state["epoch"] = 3
    invalid_state["next_sample_id"] = "not-the-reconstructed-sample"

    with pytest.raises(SamplerStateMismatchError, match="next sample identity"):
        sampler.load_state_dict(invalid_state)

    assert sampler.state_dict() == active_state


def test_sampler_tail_policies_are_explicit() -> None:
    dataset = _Dataset(5)
    drop = [
        list(DeterministicDistributedSampler(dataset, shuffle=False, rank=rank, world_size=2, tail_policy="drop"))
        for rank in range(2)
    ]
    pad = [
        list(DeterministicDistributedSampler(dataset, shuffle=False, rank=rank, world_size=2, tail_policy="pad"))
        for rank in range(2)
    ]
    uneven = [
        list(DeterministicDistributedSampler(dataset, shuffle=False, rank=rank, world_size=2, tail_policy="uneven"))
        for rank in range(2)
    ]

    assert drop == [[0, 2], [1, 3]]
    assert pad == [[0, 2, 4], [1, 3, 0]]
    assert uneven == [[0, 2, 4], [1, 3]]


def test_exhausted_state_advances_only_when_the_next_iterator_is_consumed() -> None:
    dataset = _Dataset(4)
    sampler = DeterministicDistributedSampler(dataset, shuffle=False)

    assert list(sampler) == [0, 1, 2, 3]
    assert sampler.state_dict()["epoch"] == 0
    assert sampler.state_dict()["position"] == 4
    assert list(sampler) == [0, 1, 2, 3]
    assert sampler.epoch == 1


def test_distributed_context_requires_a_complete_valid_pair() -> None:
    assert distributed_context_from_environment({}) == (0, 1)
    assert distributed_context_from_environment({"RANK": "2", "WORLD_SIZE": "4"}) == (2, 4)
    with pytest.raises(ValueError, match="both"):
        distributed_context_from_environment({"RANK": "0"})
