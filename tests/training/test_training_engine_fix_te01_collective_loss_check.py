"""TE-01 regression: FSDP2 loss finiteness must be a collective decision.

The engine used to raise ``FloatingPointError`` from a rank-local
``isfinite`` check.  When only one rank saw a NaN/inf loss it left the
training loop while its peers entered the FSDP2 backward reduce-scatter,
hanging the job until the NCCL watchdog fired.  The fixed check reduces a
finite flag with ``ReduceOp.MIN`` over the data-parallel group first, so
every rank reaches the same verdict and raises (or proceeds) together.

The two-process gloo test exercises the exact expression used by
``FSDP2TrainEngine.train_accumulation``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _worker(rank: int, world_size: int, rendezvous: str, output_dir: str) -> None:
    import torch.distributed as dist

    from worldfoundry.training.engine.fsdp import _reduced

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        verdicts: dict[str, bool] = {}

        # Case 1: only rank 1 produces a NaN loss (data-dependent event).
        loss = torch.tensor(float("nan")) if rank == 1 else torch.tensor(0.25)
        loss_is_finite = torch.isfinite(loss.detach()).all()
        verdicts["mixed_finite"] = bool(_reduced(loss_is_finite.to(torch.float32), dist.ReduceOp.MIN))

        # Case 2: every rank is finite.
        loss = torch.tensor(1.0 + rank)
        loss_is_finite = torch.isfinite(loss.detach()).all()
        verdicts["all_finite"] = bool(_reduced(loss_is_finite.to(torch.float32), dist.ReduceOp.MIN))

        torch.save(verdicts, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is unavailable",
)
def test_loss_finiteness_verdict_is_identical_on_every_rank(tmp_path: Path) -> None:
    world_size = 2
    rendezvous = tmp_path / "gloo-rendezvous"
    output_dir = tmp_path / "rank-results"
    output_dir.mkdir()

    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, str(rendezvous), str(output_dir)),
        nprocs=world_size,
        join=True,
    )

    results = [torch.load(output_dir / f"rank-{rank}.pt", weights_only=True) for rank in range(world_size)]
    for verdicts in results:
        # One non-finite rank forces the same abort verdict everywhere, so no
        # rank abandons the collective backward while peers keep waiting.
        assert verdicts["mixed_finite"] is False
        assert verdicts["all_finite"] is True
