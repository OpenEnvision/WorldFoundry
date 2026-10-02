from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdata")

from worldfoundry.training.data import (  # noqa: E402
    DeterministicDistributedSampler,
    build_stateful_dataloader,
)


class _NumberDataset:
    def __init__(self, size: int) -> None:
        self.values = tuple(range(size))
        self.sample_ids = tuple(f"sample-{index}" for index in self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


def _sampler(dataset: _NumberDataset) -> DeterministicDistributedSampler:
    return DeterministicDistributedSampler(
        dataset,
        seed=29,
        shuffle=True,
        rank=0,
        world_size=1,
        tail_policy="uneven",
    )


def test_stateful_loader_resumes_at_the_exact_next_batch() -> None:
    dataset = _NumberDataset(11)
    loader = build_stateful_dataloader(
        dataset,
        _sampler(dataset),
        batch_size=3,
        worker_seed=41,
    )
    iterator = iter(loader)
    first = next(iterator).tolist()
    state = loader.state_dict()
    expected_tail = [batch.tolist() for batch in iterator]

    restored_dataset = _NumberDataset(11)
    restored = build_stateful_dataloader(
        restored_dataset,
        _sampler(restored_dataset),
        batch_size=3,
        worker_seed=41,
    )
    restored.load_state_dict(state)

    assert len(first) == 3
    assert [batch.tolist() for batch in restored] == expected_tail


def test_stateful_loader_rejects_ambiguous_worker_and_dataset_configuration() -> None:
    dataset = _NumberDataset(5)
    other = _NumberDataset(5)
    sampler = _sampler(dataset)

    with pytest.raises(ValueError, match="exact dataset"):
        build_stateful_dataloader(other, sampler, batch_size=2)
    with pytest.raises(ValueError, match="prefetch_factor requires"):
        build_stateful_dataloader(dataset, sampler, batch_size=2, prefetch_factor=2)
    with pytest.raises(ValueError, match="persistent_workers requires"):
        build_stateful_dataloader(dataset, sampler, batch_size=2, persistent_workers=True)
    with pytest.raises(ValueError, match="multiprocessing_context requires"):
        build_stateful_dataloader(dataset, sampler, batch_size=2, multiprocessing_context="spawn")
    with pytest.raises(ValueError, match="multiprocessing_context must be"):
        build_stateful_dataloader(
            dataset,
            sampler,
            batch_size=2,
            num_workers=1,
            multiprocessing_context="unsupported",
        )
