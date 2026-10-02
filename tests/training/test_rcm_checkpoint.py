from __future__ import annotations

from pathlib import Path

import torch
from test_rcm_runtime import _engine

from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.post_training.distillation.rcm import (
    NativeRCMTrainingSession,
    RCMTrainingBatch,
)


class _StatefulRCMLoader:
    def __init__(self) -> None:
        self.cursor = 0

    def __iter__(self) -> _StatefulRCMLoader:
        return self

    def __next__(self) -> RCMTrainingBatch:
        value = float(self.cursor + 1) / 10.0
        self.cursor += 1
        return RCMTrainingBatch(
            sample_ids=(f"sample-{self.cursor}",),
            clean_latents=torch.tensor([[value, value + 0.2]]),
            conditioning={},
            unconditional_conditioning={},
        )

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.cursor = int(state_dict["cursor"])


def _checkpointable_rcm():
    engine, _, _, _ = _engine(accumulation=2)
    loader = _StatefulRCMLoader()
    progress = TrainingProgress()
    generator = torch.Generator().manual_seed(37)
    model = torch.nn.ModuleDict(
        {
            "student": engine.student_module,
            "fake_score": engine.fake_score_module,
        }
    )
    state = TrainingState(
        model=model,
        optimizer=(engine.student_optimizer, engine.fake_score_optimizer),
        engine=engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={
            "algorithm": "rcm",
            "gradient_accumulation_steps": engine.gradient_accumulation_steps,
        },
    )
    return engine, loader, progress, generator, model, state


def test_rcm_compound_dcp_restores_both_roles_rng_loader_and_phase(tmp_path: Path) -> None:
    engine, loader, progress, generator, model, state = _checkpointable_rcm()
    session = NativeRCMTrainingSession(engine, loader, progress)
    session.run(max_steps=4, generator=generator)
    manager = TrainingCheckpointer(tmp_path / "rcm-checkpoints")
    artifact = manager.save(state)

    expected_summary = session.run(max_steps=2, generator=generator)
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_generator = generator.get_state().clone()

    (
        restored_engine,
        restored_loader,
        restored_progress,
        restored_generator,
        restored_model,
        restored_state,
    ) = _checkpointable_rcm()
    manager.load(restored_state, artifact.path)
    actual_summary = NativeRCMTrainingSession(
        restored_engine,
        restored_loader,
        restored_progress,
    ).run(max_steps=2, generator=restored_generator)

    assert restored_progress.optimizer_steps == 6
    assert restored_progress.microbatches_seen == 12
    assert restored_loader.cursor == 12
    assert actual_summary.student_optimizer_steps == expected_summary.student_optimizer_steps
    assert actual_summary.fake_score_optimizer_steps == expected_summary.fake_score_optimizer_steps
    assert torch.equal(restored_generator.get_state(), expected_generator)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
