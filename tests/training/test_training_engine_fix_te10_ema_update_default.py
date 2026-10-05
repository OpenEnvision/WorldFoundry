"""TE-10 regression: shared video builders default to per-optimizer-step EMA.

``build_cached_video_flow_*_session`` used to default ``ema_update`` to
``"microbatch"``.  Under gradient accumulation that updates the EMA after
every microbatch backward -- absorbing parameters that have not changed yet
N-1 times per optimizer step -- and inflates counter-based decay schedules
(PowerEMA/LitEma ``num_updates``).  The default is now ``"optimizer-step"``;
families that need Lightning author parity (LVDM) opt in to ``"microbatch"``
explicitly, which the existing ``test_lvdm_ema_training`` suite pins.

These tests check the new defaults and pin the engine-level semantics of
both modes without requiring the torchdata-backed loader stack.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import ObjectiveBatch, PreparedBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.engine import SingleDeviceTrainEngine  # noqa: E402
from worldfoundry.training.engine.video_flow import (  # noqa: E402
    _training_callbacks,
    build_cached_video_flow_fsdp2_session,
    build_cached_video_flow_single_device_session,
)
from worldfoundry.training.objectives import FlowMatchingConfig, FlowMatchingObjective  # noqa: E402


class _TinyAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes: tuple[type, ...] = ()

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0],
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


class _CountingEma:
    def __init__(self) -> None:
        self.num_updates = 0

    def __call__(self, module: torch.nn.Module) -> None:
        del module
        self.num_updates += 1


def _microbatches() -> tuple[TrainingBatch, TrainingBatch]:
    values = torch.tensor(
        [
            [[[[0.0, 0.5], [1.0, -0.5]]]],
            [[[[0.25, -0.25], [0.75, -0.75]]]],
        ]
    )
    return tuple(
        TrainingBatch(
            sample_ids=(("a", "b")[index],),
            prompts=(("first", "second")[index],),
            pixel_values=values[index : index + 1],
        )
        for index in range(2)
    )


def _accumulate_once(ema_update: str) -> _CountingEma:
    adapter = _TinyAdapter()
    ema = _CountingEma()
    train_batch_end, optimizer_step_end = _training_callbacks(
        adapter.trainable_module,
        ema=ema,
        ema_update=ema_update,
        lr_scheduler=None,
    )
    engine = SingleDeviceTrainEngine(
        adapter,
        FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform")),
        torch.optim.SGD(adapter.trainable_module.parameters(), lr=0.1),
        train_batch_end=train_batch_end,
        optimizer_step_end=optimizer_step_end,
    )
    engine.train_accumulation(_microbatches(), generator=torch.Generator().manual_seed(11))
    return ema


def test_shared_video_builders_default_to_optimizer_step_ema() -> None:
    for builder in (
        build_cached_video_flow_single_device_session,
        build_cached_video_flow_fsdp2_session,
    ):
        default = inspect.signature(builder).parameters["ema_update"].default
        assert default == "optimizer-step", builder.__name__


def test_optimizer_step_mode_updates_ema_once_per_applied_step() -> None:
    ema = _accumulate_once("optimizer-step")
    assert ema.num_updates == 1


def test_microbatch_mode_keeps_lightning_per_batch_parity() -> None:
    ema = _accumulate_once("microbatch")
    assert ema.num_updates == 2
