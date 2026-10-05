from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import ObjectiveBatch, PreparedBatch, TrainingBatch  # noqa: E402
from worldfoundry.training.engine import SingleDeviceTrainEngine, build_adamw  # noqa: E402
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


class _FailAfterMutationSGD(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("failure after optimizer mutation")


class _StateGateSGD(torch.optim.SGD):
    def __init__(self, params, *, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.engine: SingleDeviceTrainEngine | None = None
        self.state_error: RuntimeError | None = None

    def step(self, closure=None):
        assert self.engine is not None
        try:
            self.engine.state_dict()
        except RuntimeError as error:
            self.state_error = error
        return super().step(closure)


class _ScaledLoss:
    def __init__(self, loss: torch.Tensor, events: list[str]) -> None:
        self.loss = loss
        self.events = events

    def backward(self) -> None:
        self.events.append("backward")
        self.loss.backward()


class _RecordingGradScaler:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def scale(self, loss: torch.Tensor) -> _ScaledLoss:
        self.events.append("scale")
        return _ScaledLoss(loss, self.events)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        self.events.append("unscale")

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.events.append("scaler-step")
        optimizer.step()

    def update(self) -> None:
        self.events.append("scaler-update")


class _FailingForwardAdapter(_TinyAdapter):
    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self.forward_calls = 0
        self.fail_on_call: int | None = fail_on_call

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        self.forward_calls += 1
        if self.forward_calls == self.fail_on_call:
            raise RuntimeError("intentional pre-step forward failure")
        return super().forward_train(batch)


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


def test_single_device_engine_applies_a_finite_clipped_optimizer_step() -> None:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = build_adamw(
        adapter.trainable_module.parameters(),
        learning_rate=1.0e-2,
        fused="auto",
    )
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        max_grad_norm=0.5,
    )
    before = adapter.trainable_module.weight.detach().clone()

    result = engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(11))

    assert result.loss.requires_grad
    assert bool(torch.isfinite(result.loss))
    assert bool(torch.isfinite(result.metrics["grad_norm"]))
    assert result.metrics["global_step"].item() == 1
    assert engine.state_dict()["global_step"] == 1
    assert not torch.equal(adapter.trainable_module.weight.detach(), before)
    assert optimizer.defaults["fused"] is False


def test_single_device_engine_can_observe_gradient_norm_without_clipping() -> None:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = torch.optim.SGD(adapter.trainable_module.parameters(), lr=1.0e-2)
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        max_grad_norm=None,
    )

    result = engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(11))

    assert bool(torch.isfinite(result.metrics["grad_norm"]))
    assert result.diagnostics["max_grad_norm"] is None


def test_single_device_engine_accumulates_microbatches_with_one_optimizer_step() -> None:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = build_adamw(
        adapter.trainable_module.parameters(),
        learning_rate=1.0e-2,
        fused=False,
    )
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    microbatches = _microbatches()
    before = adapter.trainable_module.weight.detach().clone()

    result = engine.train_accumulation(
        microbatches,
        generator=torch.Generator().manual_seed(19),
    )

    assert result.sample_count == 2
    assert result.metrics["microbatch_count"].item() == 2
    assert result.metrics["global_step"].item() == 1
    assert result.diagnostics["gradient_accumulation"] == "token-weighted"
    assert engine.global_step == 1
    assert not torch.equal(adapter.trainable_module.weight.detach(), before)


@pytest.mark.parametrize("accumulate", [False, True], ids=["single-step", "accumulation"])
def test_grad_scaler_unscales_before_clipping_and_steps_after_clipping(
    monkeypatch: pytest.MonkeyPatch,
    accumulate: bool,
) -> None:
    engine_module = importlib.import_module("worldfoundry.training.engine.single_device")
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = torch.optim.SGD(adapter.trainable_module.parameters(), lr=0.1)
    events: list[str] = []
    original_clip = engine_module.clip_grad_norm_

    def observed_clip(*args, **kwargs):
        events.append("clip")
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(engine_module, "clip_grad_norm_", observed_clip)
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    engine.grad_scaler = _RecordingGradScaler(events)

    if accumulate:
        engine.train_accumulation(
            _microbatches(),
            generator=torch.Generator().manual_seed(20),
        )
        expected_prefix = ["scale", "backward", "scale", "backward"]
    else:
        engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(20))
        expected_prefix = ["scale", "backward"]

    assert events == [*expected_prefix, "unscale", "clip", "scaler-step", "scaler-update"]


def test_train_batch_end_runs_before_and_after_the_accumulated_optimizer_step() -> None:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = torch.optim.SGD(adapter.trainable_module.parameters(), lr=0.1)
    observed: list[torch.Tensor] = []
    optimizer_steps: list[int] = []
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        train_batch_end=lambda: observed.append(adapter.trainable_module.weight.detach().clone()),
        optimizer_step_end=lambda: optimizer_steps.append(engine.global_step),
    )
    before = adapter.trainable_module.weight.detach().clone()

    engine.train_accumulation(
        _microbatches(),
        generator=torch.Generator().manual_seed(23),
    )

    assert len(observed) == 2
    assert optimizer_steps == [1]
    torch.testing.assert_close(observed[0], before, rtol=0, atol=0)
    torch.testing.assert_close(observed[1], adapter.trainable_module.weight, rtol=0, atol=0)
    assert not torch.equal(observed[0], observed[1])


@pytest.mark.parametrize("accumulate", [False, True], ids=["single-step", "accumulation"])
def test_optimizer_failure_after_parameter_mutation_poisons_engine(accumulate: bool) -> None:
    adapter = _TinyAdapter()
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = _FailAfterMutationSGD(adapter.trainable_module.parameters(), lr=0.1)
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    before = adapter.trainable_module.weight.detach().clone()

    with pytest.raises(RuntimeError, match="failure after optimizer mutation"):
        if accumulate:
            engine.train_accumulation(
                _microbatches(),
                generator=torch.Generator().manual_seed(29),
            )
        else:
            engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(29))

    assert not torch.equal(adapter.trainable_module.weight.detach(), before)
    assert engine.is_poisoned
    assert engine.global_step == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.train_step(_raw_batch())
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.train_accumulation(_microbatches())
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.load_state_dict(
            {
                "schema": "worldfoundry-training-engine-single",
                "global_step": 0,
            }
        )


def test_state_dict_is_blocked_while_optimizer_commit_is_in_progress() -> None:
    adapter = _TinyAdapter()
    optimizer = _StateGateSGD(adapter.trainable_module.parameters(), lr=0.1)
    engine = SingleDeviceTrainEngine(adapter, FlowMatchingObjective(), optimizer)
    optimizer.engine = engine

    engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(31))

    assert optimizer.state_error is not None
    assert "active training step" in str(optimizer.state_error)
    assert not engine.is_poisoned
    assert engine.global_step == 1


def test_single_step_pre_commit_failure_clears_gradients_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_module = importlib.import_module("worldfoundry.training.engine.single_device")
    adapter = _TinyAdapter()
    optimizer = torch.optim.SGD(adapter.trainable_module.parameters(), lr=0.1)
    engine = SingleDeviceTrainEngine(adapter, FlowMatchingObjective(), optimizer)
    real_clip_grad_norm = engine_module.clip_grad_norm_
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("intentional pre-step gradient failure")
        return real_clip_grad_norm(*args, **kwargs)

    monkeypatch.setattr(engine_module, "clip_grad_norm_", fail_once)

    with pytest.raises(RuntimeError, match="intentional pre-step gradient failure"):
        engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(37))

    assert not engine.is_poisoned
    assert engine.global_step == 0
    assert all(parameter.grad is None for parameter in engine.parameters)

    result = engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(41))
    assert result.metrics["global_step"].item() == 1


def test_accumulation_pre_commit_failure_clears_partial_gradients_and_allows_retry() -> None:
    adapter = _FailingForwardAdapter(fail_on_call=2)
    optimizer = torch.optim.SGD(adapter.trainable_module.parameters(), lr=0.1)
    engine = SingleDeviceTrainEngine(adapter, FlowMatchingObjective(), optimizer)

    with pytest.raises(RuntimeError, match="intentional pre-step forward failure"):
        engine.train_accumulation(
            _microbatches(),
            generator=torch.Generator().manual_seed(43),
        )

    assert not engine.is_poisoned
    assert engine.global_step == 0
    assert all(parameter.grad is None for parameter in engine.parameters)

    adapter.fail_on_call = None
    result = engine.train_accumulation(
        _microbatches(),
        generator=torch.Generator().manual_seed(47),
    )
    assert result.metrics["global_step"].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_single_device_engine_runs_bfloat16_forward_with_fused_cuda_adamw() -> None:
    adapter = _TinyAdapter()
    adapter.trainable_module.to("cuda")
    objective = FlowMatchingObjective(
        FlowMatchingConfig(
            timestep_sampler="logit_normal",
            num_train_timesteps=1000,
            flow_shift=3.0,
        )
    )
    optimizer = build_adamw(
        adapter.trainable_module.parameters(),
        learning_rate=1.0e-2,
        fused="auto",
    )
    engine = SingleDeviceTrainEngine(
        adapter,
        objective,
        optimizer,
        autocast_dtype=torch.bfloat16,
    )
    raw = _raw_batch()
    batch = TrainingBatch(
        sample_ids=raw.sample_ids,
        prompts=raw.prompts,
        pixel_values=raw.pixel_values.to("cuda"),
    )

    result = engine.train_step(
        batch,
        generator=torch.Generator(device="cuda").manual_seed(23),
    )

    assert bool(torch.isfinite(result.loss))
    assert result.diagnostics["autocast_dtype"] == "torch.bfloat16"
    assert result.diagnostics["grad_scaling"] is False
    assert engine.grad_scaler is None
    assert optimizer.defaults["fused"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_single_device_fp16_compute_keeps_fp32_master_parameters_and_scales_gradients() -> None:
    adapter = _TinyAdapter()
    adapter.trainable_module.to(device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(adapter.trainable_module.parameters(), lr=0.01)
    engine = SingleDeviceTrainEngine(
        adapter,
        FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform")),
        optimizer,
        autocast_dtype=torch.float16,
    )
    raw = _raw_batch()
    batch = TrainingBatch(
        sample_ids=raw.sample_ids,
        prompts=raw.prompts,
        pixel_values=raw.pixel_values.to("cuda"),
    )

    result = engine.train_step(
        batch,
        generator=torch.Generator(device="cuda").manual_seed(24),
    )

    assert all(parameter.dtype is torch.float32 for parameter in engine.parameters)
    assert engine.grad_scaler is not None
    assert engine.grad_scaler.state_dict()["scale"] > 0
    assert result.diagnostics["autocast_dtype"] == "torch.float16"
    assert result.diagnostics["grad_scaling"] is True


def test_single_device_engine_rejects_optimizer_parameter_drift() -> None:
    adapter = _TinyAdapter()
    other = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(other.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="optimizer parameter audit"):
        SingleDeviceTrainEngine(
            adapter,
            FlowMatchingObjective(),
            optimizer,
        )


def test_single_device_engine_state_is_strict_and_round_trips() -> None:
    adapter = _TinyAdapter()
    optimizer = build_adamw(adapter.trainable_module.parameters(), learning_rate=1.0e-3, fused=False)
    engine = SingleDeviceTrainEngine(adapter, FlowMatchingObjective(), optimizer)

    engine.load_state_dict(
        {
            "schema": "worldfoundry-training-engine-single",
            "global_step": 17,
        }
    )

    assert engine.global_step == 17
    with pytest.raises(ValueError, match="state fields"):
        engine.load_state_dict({"schema": "worldfoundry-training-engine-single"})
    assert engine.global_step == 17
    with pytest.raises(ValueError, match="unsupported engine state schema"):
        engine.load_state_dict(
            {
                "schema": "not-the-active-schema",
                "global_step": 3,
            }
        )
    assert engine.global_step == 17
    with pytest.raises(TypeError, match="global_step must be an integer"):
        engine.load_state_dict(
            {
                "schema": "worldfoundry-training-engine-single",
                "global_step": 3.5,
            }
        )
    assert engine.global_step == 17

    result = engine.train_step(_raw_batch(), generator=torch.Generator().manual_seed(53))
    assert result.metrics["global_step"].item() == 18
