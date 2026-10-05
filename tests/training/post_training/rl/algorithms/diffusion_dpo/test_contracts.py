from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.rl.algorithms.diffusion_dpo import (  # noqa: E402
    DiffusionDPOBatch,
)


def _batch(*, pair_ids: tuple[str, ...] = ("first", "first", "second", "second")):
    return DiffusionDPOBatch(
        batch_id="batch-1",
        sample_ids=("chosen-a", "rejected-a", "chosen-b", "rejected-b"),
        pair_ids=pair_ids,
        clean_latents=torch.zeros(4, 2),
        conditioning={"context": torch.ones(4, 1)},
    )


def test_batch_requires_adjacent_unique_chosen_rejected_pairs() -> None:
    batch = _batch()

    assert batch.batch_size == 4
    assert batch.pair_count == 2
    with pytest.raises(ValueError, match="immediately followed"):
        _batch(pair_ids=("first", "second", "first", "second"))
    with pytest.raises(ValueError, match="exactly one"):
        _batch(pair_ids=("same", "same", "same", "same"))
