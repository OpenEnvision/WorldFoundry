from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.post_training.distillation.adversarial_diffusion import (  # noqa: E402
    ADDLossResult,
    ADDTrainingBatch,
    NativeADDTrainEngine,
)


class _DistributedLossAdapter:
    def __init__(self, student, discriminator) -> None:
        self.student = student
        self.discriminator = discriminator

    def loss_denominator(self, batch, *, role):
        if role not in {"generator", "discriminator"}:
            raise ValueError(role)
        return batch.batch_size

    def generator_loss(self, batch, *, generator=None):
        del generator
        prediction = self.student(batch.clean_latents).flatten()
        loss = prediction.square().mean()
        return ADDLossResult(
            loss=loss,
            metrics={
                "loss_denominator": torch.tensor(
                    float(batch.batch_size),
                    device=prediction.device,
                )
            },
        )

    def discriminator_loss(self, batch, *, generator=None):
        del generator
        values = batch.real_images.flatten(1)
        prediction = self.discriminator(values).flatten()
        loss = (prediction - 1.0).square().mean()
        return ADDLossResult(
            loss=loss,
            metrics={
                "loss_denominator": torch.tensor(
                    float(batch.batch_size),
                    device=prediction.device,
                )
            },
        )


def _worker(rank: int, world_size: int, rendezvous: str, output_dir: str) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        student_raw = torch.nn.Linear(1, 1, bias=False)
        discriminator_raw = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            student_raw.weight.fill_(0.5)
            discriminator_raw.weight.fill_(0.25)
        student = DistributedDataParallel(student_raw)
        discriminator = DistributedDataParallel(discriminator_raw)
        teacher = torch.nn.Identity()
        decoder = torch.nn.Identity()
        feature = torch.nn.Identity()
        adapter = _DistributedLossAdapter(student, discriminator)
        student_optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1)
        engine = NativeADDTrainEngine(
            student_module=student,
            teacher_module=teacher,
            decoder_module=decoder,
            discriminator_module=discriminator,
            discriminator_feature_module=feature,
            loss_adapter=adapter,
            student_optimizer=student_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            student_max_grad_norm=100.0,
            discriminator_max_grad_norm=100.0,
        )
        values = torch.tensor([[1.0]]) if rank == 0 else torch.tensor([[2.0], [3.0], [4.0]])
        batch = ADDTrainingBatch(
            sample_ids=tuple(f"rank-{rank}-{index}" for index in range(values.shape[0])),
            clean_latents=values,
            real_images=values[:, :, None, None],
            conditioning={},
            discriminator_conditioning={},
        )

        result = engine.train_step(batch)

        torch.save(
            {
                "student": student_raw.weight.detach().clone(),
                "discriminator": discriminator_raw.weight.detach().clone(),
                "generator_loss": result.generator_loss,
                "discriminator_loss": result.discriminator_loss,
                "state": engine.state_dict(),
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is unavailable",
)
def test_add_ddp_uses_global_sample_weighting_at_arbitrary_world_size(tmp_path: Path) -> None:
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
    for result in results:
        torch.testing.assert_close(result["student"], torch.tensor([[-0.25]]), rtol=0, atol=1.0e-7)
        torch.testing.assert_close(
            result["discriminator"],
            torch.tensor([[0.375]]),
            rtol=0,
            atol=1.0e-7,
        )
        assert result["state"]["data_parallel_size"] == world_size
        assert result["state"]["global_step"] == 1
    torch.testing.assert_close(results[0]["student"], results[1]["student"], rtol=0, atol=0)
    torch.testing.assert_close(
        results[0]["discriminator"],
        results[1]["discriminator"],
        rtol=0,
        atol=0,
    )
