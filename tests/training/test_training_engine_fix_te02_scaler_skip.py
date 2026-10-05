"""TE-02 regression: fp16 GradScaler overflow must skip the step, not abort.

The engine used to call ``clip_grad_norm_(error_if_nonfinite=True)`` between
``scaler.unscale_`` and ``scaler.step``/``scaler.update``.  A single gradient
overflow therefore raised before the scaler could skip the step and lower the
scale, permanently killing fp16 training.  These CPU tests pin the corrected
semantics:

* active ``torch.amp.GradScaler`` + non-finite gradients -> no exception, the
  optimizer step is skipped, the scale backs off, and training self-heals;
* no scaler (bf16/fp32) and duck-typed/disabled scalers keep the original
  fail-stop ``RuntimeError`` from gradient clipping.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import ObjectiveBatch, PreparedBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.engine import SingleDeviceTrainEngine  # noqa: E402
from worldfoundry.training.objectives import FlowMatchingConfig, FlowMatchingObjective  # noqa: E402


class _InfiniteGradient(torch.autograd.Function):
    """Identity forward whose backward emits an overflowed (inf) gradient."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return torch.full_like(grad_output, float("inf"))


class _TinyAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes: tuple[type, ...] = ()

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.overflow_gradients = False

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0],
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        prediction = self.trainable_module(batch.model_input)
        if self.overflow_gradients:
            return _InfiniteGradient.apply(prediction)
        return prediction


class _DuckTypedScaler:
    """Legacy-style fake without get_scale()/is_enabled(); no skip support."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _raw_batch() -> TrainingBatch:
    return TrainingBatch(
        sample_ids=("a", "b"),
        prompts=("first", "second"),
        pixel_values=torch.tensor(
            [
                [[[[0.0, 0.5], [1.0, -0.5]]]],
                [[[[0.25, -0.25], [0.75, -0.75]]]],
            ]
        ),
    )


def _microbatches() -> tuple[TrainingBatch, TrainingBatch]:
    raw = _raw_batch()
    assert isinstance(raw.pixel_values, torch.Tensor)
    return tuple(
        TrainingBatch(
            sample_ids=(raw.sample_ids[index],),
            prompts=(raw.prompts[index],),
            pixel_values=raw.pixel_values[index : index + 1],
        )
        for index in range(2)
    )


def _engine(adapter: _TinyAdapter, **kwargs) -> tuple[SingleDeviceTrainEngine, torch.optim.SGD, list[int]]:
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = torch.optim.SGD(adapter.trainable_module.parameters(), lr=0.1)
    optimizer_step_ends: list[int] = []
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        optimizer_step_end=lambda: optimizer_step_ends.append(1),
        **kwargs,
    )
    return engine, optimizer, optimizer_step_ends


def test_scaler_overflow_skips_step_lowers_scale_and_self_heals() -> None:
    adapter = _TinyAdapter()
    engine, _, optimizer_step_ends = _engine(adapter)
    engine.grad_scaler = torch.amp.GradScaler("cpu", init_scale=64.0)
    before = adapter.trainable_module.weight.detach().clone()

    adapter.overflow_gradients = True
    result = engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(11))

    assert result.skipped is True
    assert result.metrics["optimizer_step_skipped"] is True
    assert "grad_norm" not in result.metrics
    torch.testing.assert_close(adapter.trainable_module.weight.detach(), before, rtol=0, atol=0)
    assert engine.grad_scaler.get_scale() == 32.0
    assert engine.global_step == 1
    assert not engine.is_poisoned
    # Iteration-aligned callbacks (scheduler/EMA) still fire on skipped steps.
    assert len(optimizer_step_ends) == 1

    adapter.overflow_gradients = False
    recovered = engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(13))

    assert recovered.skipped is False
    assert bool(torch.isfinite(recovered.metrics["grad_norm"]))
    assert "optimizer_step_skipped" not in recovered.metrics
    assert not torch.equal(adapter.trainable_module.weight.detach(), before)
    assert engine.grad_scaler.get_scale() == 32.0
    assert engine.global_step == 2
    assert len(optimizer_step_ends) == 2


def test_scaler_overflow_skips_accumulated_step_without_poisoning() -> None:
    adapter = _TinyAdapter()
    engine, _, _ = _engine(adapter)
    engine.grad_scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    before = adapter.trainable_module.weight.detach().clone()

    adapter.overflow_gradients = True
    result = engine.train_accumulation(
        _microbatches(),
        generator=torch.Generator().manual_seed(19),
    )

    assert result.skipped is True
    assert result.metrics["optimizer_step_skipped"] is True
    assert "grad_norm" not in result.metrics
    assert result.metrics["microbatch_count"].item() == 2
    torch.testing.assert_close(adapter.trainable_module.weight.detach(), before, rtol=0, atol=0)
    assert engine.grad_scaler.get_scale() == 8.0
    assert engine.global_step == 1
    assert not engine.is_poisoned

    adapter.overflow_gradients = False
    recovered = engine.train_accumulation(
        _microbatches(),
        generator=torch.Generator().manual_seed(23),
    )

    assert recovered.skipped is False
    assert bool(torch.isfinite(recovered.metrics["grad_norm"]))
    assert not torch.equal(adapter.trainable_module.weight.detach(), before)


def test_without_scaler_non_finite_gradients_still_fail_stop() -> None:
    adapter = _TinyAdapter()
    engine, _, optimizer_step_ends = _engine(adapter)
    assert engine.grad_scaler is None
    before = adapter.trainable_module.weight.detach().clone()

    adapter.overflow_gradients = True
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(29))

    torch.testing.assert_close(adapter.trainable_module.weight.detach(), before, rtol=0, atol=0)
    assert engine.global_step == 0
    assert not engine.is_poisoned
    assert optimizer_step_ends == []


def test_duck_typed_scaler_keeps_legacy_fail_stop_clip() -> None:
    adapter = _TinyAdapter()
    engine, _, _ = _engine(adapter)
    engine.grad_scaler = _DuckTypedScaler()

    adapter.overflow_gradients = True
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(31))

    assert engine.global_step == 0
    assert not engine.is_poisoned


def test_disabled_real_scaler_keeps_legacy_fail_stop_clip() -> None:
    adapter = _TinyAdapter()
    engine, _, _ = _engine(adapter)
    engine.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)

    adapter.overflow_gradients = True
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(37))

    assert engine.global_step == 0
    assert not engine.is_poisoned
