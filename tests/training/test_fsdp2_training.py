from __future__ import annotations

import importlib
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.api import (  # noqa: E402
    ObjectiveBatch,
    PreparedBatch,
    TrainingBatch,
)
from worldfoundry.training.checkpoint import (  # noqa: E402
    PendingTrainingCheckpoint,
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.distributed import (  # noqa: E402
    DistributedTrainingContext,
    FSDP2Application,
    ParallelPlan,
    apply_fsdp2,
)
from worldfoundry.training.engine import (  # noqa: E402
    FSDP2TrainEngine,
    SingleDeviceTrainEngine,
    build_adamw,
    trainable_parameters,
)
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.recipes import DistributedSpec  # noqa: E402
from worldfoundry.training.tuning import load_full_model  # noqa: E402


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.adapter_down = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.adapter_up = torch.nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.adapter_up(self.adapter_down(value.float())).to(value.dtype)
        return torch.nn.functional.silu(self.projection(value) + residual)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.output = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return self.output(value)


class _Adapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes = (_Block,)

    def __init__(self, device: torch.device) -> None:
        self.trainable_module = _Model().to(device=device, dtype=torch.bfloat16)
        self.trainable_module.requires_grad_(False)
        for block in self.trainable_module.blocks:
            block.adapter_down.float()
            block.adapter_up.float()
            block.adapter_down.requires_grad_(True)
            block.adapter_up.requires_grad_(True)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        reference = next(self.trainable_module.parameters())
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0].to(
                device=reference.device,
                dtype=torch.bfloat16,
            ),
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


def test_fsdp2_master_storage_dtype_is_not_classified_as_a_forward_precision_island() -> None:
    from worldfoundry.training.distributed.fsdp import _precision_island_modules

    model = _Model().float().requires_grad_(False)
    for block in model.blocks:
        block.adapter_down.requires_grad_(True)
        block.adapter_up.requires_grad_(True)

    assert _precision_island_modules(
        model,
        param_dtype=torch.bfloat16,
        master_parameter_dtype=None,
    )
    assert not _precision_island_modules(
        model,
        param_dtype=torch.bfloat16,
        master_parameter_dtype=torch.float32,
    )


class _FP32Adapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes = (_Block,)

    def __init__(self, device: torch.device) -> None:
        self.trainable_module = _Model().to(device=device, dtype=torch.float32)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0].to(
                device=next(self.trainable_module.parameters()).device,
                dtype=torch.float32,
            ),
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


class _StatefulCursor:
    def __init__(self) -> None:
        self.cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.cursor = int(state_dict["cursor"])


class _SyntheticFSDPRoot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.gradient_sync_enabled = True
        self.reshard_after_backward = True
        self.gradient_sync_calls: list[bool] = []
        self.reshard_calls: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)

    def set_requires_gradient_sync(self, enabled: bool) -> None:
        self.gradient_sync_enabled = enabled
        self.gradient_sync_calls.append(enabled)

    def set_reshard_after_backward(self, enabled: bool) -> None:
        self.reshard_after_backward = enabled
        self.reshard_calls.append(enabled)


class _SyntheticFSDPAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes: tuple[type, ...] = ()

    def __init__(self, *, fail_on_forward: int | None = None) -> None:
        self.trainable_module = _SyntheticFSDPRoot()
        self.forward_calls = 0
        self.fail_on_forward = fail_on_forward

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0],
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        self.forward_calls += 1
        if self.forward_calls == self.fail_on_forward:
            raise RuntimeError("intentional FSDP2 pre-step failure")
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


class _FailAfterMutationSGD(torch.optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        raise RuntimeError("failure after FSDP2 optimizer mutation")


def _batch(sample_id: str, offset: float) -> TrainingBatch:
    pixels = torch.tensor(
        [[[[[0.1 + offset, -0.3], [0.7, 0.2 - offset]]]]],
        dtype=torch.float32,
    )
    return TrainingBatch(
        sample_ids=(sample_id,),
        prompts=("cached",),
        pixel_values=pixels,
    )


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _local_parameter_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().to_local().clone() for name, parameter in module.named_parameters()}


def _synthetic_fsdp_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_on_forward: int | None = None,
    fail_after_optimizer_mutation: bool = False,
    max_grad_norm: float | None = 1.0,
    train_batch_end=None,
) -> tuple[FSDP2TrainEngine, _SyntheticFSDPAdapter]:
    engine_module = importlib.import_module("worldfoundry.training.engine.fsdp")
    monkeypatch.setattr(engine_module, "FSDPModule", _SyntheticFSDPRoot)
    monkeypatch.setattr(
        engine_module,
        "_reduced",
        lambda value, op=torch.distributed.ReduceOp.SUM: value.detach().clone(),
    )
    monkeypatch.setattr(engine_module.dist, "all_reduce", lambda *args, **kwargs: None)
    adapter = _SyntheticFSDPAdapter(fail_on_forward=fail_on_forward)
    optimizer_type = _FailAfterMutationSGD if fail_after_optimizer_mutation else torch.optim.SGD
    optimizer = optimizer_type(adapter.trainable_module.parameters(), lr=0.1)
    engine = object.__new__(FSDP2TrainEngine)
    SingleDeviceTrainEngine.__init__(
        engine,
        adapter,
        FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform")),
        optimizer,
        max_grad_norm=max_grad_norm,
        train_batch_end=train_batch_end,
    )
    engine.application = SimpleNamespace(
        to_dict=lambda: {"kind": "synthetic-fsdp2"},
        parallel_plan=SimpleNamespace(to_dict=lambda: {"backend": "fsdp2", "world_size": 1}),
    )
    engine.data_parallel_size = 1
    return engine, adapter


def test_fsdp2_engine_can_observe_gradient_norm_without_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _synthetic_fsdp_engine(monkeypatch, max_grad_norm=None)

    result = engine.train_accumulation(
        (_batch("first", 0.0), _batch("second", 0.1)),
        generator=torch.Generator().manual_seed(59),
    )

    assert bool(torch.isfinite(result.metrics["grad_norm"]))
    assert result.diagnostics["max_grad_norm"] is None


def test_fsdp2_train_batch_end_preserves_accumulation_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[torch.Tensor] = []
    adapter_holder: list[_SyntheticFSDPAdapter] = []

    def observe() -> None:
        observed.append(adapter_holder[0].trainable_module.projection.weight.detach().clone())

    engine, adapter = _synthetic_fsdp_engine(monkeypatch, train_batch_end=observe)
    adapter_holder.append(adapter)
    before = adapter.trainable_module.projection.weight.detach().clone()

    engine.train_accumulation(
        (_batch("first", 0.0), _batch("second", 0.1)),
        generator=torch.Generator().manual_seed(60),
    )

    assert len(observed) == 2
    torch.testing.assert_close(observed[0], before, rtol=0, atol=0)
    torch.testing.assert_close(observed[1], adapter.trainable_module.projection.weight, rtol=0, atol=0)
    assert not torch.equal(observed[0], observed[1])


def test_fsdp2_partial_optimizer_commit_poisons_and_restores_sync_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, adapter = _synthetic_fsdp_engine(
        monkeypatch,
        fail_after_optimizer_mutation=True,
    )
    before = adapter.trainable_module.projection.weight.detach().clone()

    with pytest.raises(RuntimeError, match="failure after FSDP2 optimizer mutation"):
        engine.train_accumulation(
            (_batch("first", 0.0), _batch("second", 0.1)),
            generator=torch.Generator().manual_seed(61),
        )

    assert not torch.equal(adapter.trainable_module.projection.weight.detach(), before)
    assert engine.is_poisoned
    assert adapter.trainable_module.gradient_sync_enabled is True
    assert adapter.trainable_module.reshard_after_backward is True
    assert adapter.trainable_module.gradient_sync_calls[-1] is True
    assert adapter.trainable_module.reshard_calls[-1] is True
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.train_step(_batch("blocked", 0.0))


def test_fsdp2_pre_commit_failure_clears_gradients_restores_sync_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, adapter = _synthetic_fsdp_engine(monkeypatch, fail_on_forward=2)

    with pytest.raises(RuntimeError, match="intentional FSDP2 pre-step failure"):
        engine.train_accumulation(
            (_batch("first", 0.0), _batch("second", 0.1)),
            generator=torch.Generator().manual_seed(67),
        )

    assert not engine.is_poisoned
    assert engine.global_step == 0
    assert all(parameter.grad is None for parameter in engine.parameters)
    assert adapter.trainable_module.gradient_sync_enabled is True
    assert adapter.trainable_module.reshard_after_backward is True

    adapter.fail_on_forward = None
    result = engine.train_accumulation(
        (_batch("first", 0.0), _batch("second", 0.1)),
        generator=torch.Generator().manual_seed(71),
    )
    assert result.metrics["global_step"].item() == 1
    assert adapter.trainable_module.gradient_sync_enabled is True
    assert adapter.trainable_module.reshard_after_backward is True


def test_fsdp2_state_load_validates_every_field_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _synthetic_fsdp_engine(monkeypatch)
    engine.global_step = 9

    with pytest.raises(ValueError, match="application differs"):
        engine.load_state_dict(
            {
                "schema": "worldfoundry-training-engine-fsdp2",
                "global_step": 3,
                "fsdp2_application": {"kind": "another-application"},
            }
        )
    assert engine.global_step == 9
    with pytest.raises(TypeError, match="global_step must be an integer"):
        engine.load_state_dict(
            {
                "schema": "worldfoundry-training-engine-fsdp2",
                "global_step": 3.5,
                "fsdp2_application": {"kind": "synthetic-fsdp2"},
            }
        )
    assert engine.global_step == 9

    engine.load_state_dict(
        {
            "schema": "worldfoundry-training-engine-fsdp2",
            "global_step": 3,
            "fsdp2_application": {"kind": "synthetic-fsdp2"},
        }
    )
    assert engine.global_step == 3


def test_fsdp2_application_distinguishes_trainable_and_frozen_roles() -> None:
    plan = ParallelPlan.resolve(
        DistributedSpec(backend="fsdp2", dp_shard="auto"),
        world_size=1,
    )
    frozen = FSDP2Application(
        parallel_plan=plan,
        block_module_names=("blocks.0",),
        block_class_names=("tests.Block",),
        parameter_names=("blocks.0.weight",),
        trainable_parameter_names=(),
        parameter_count=4,
        trainable_parameter_count=0,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        root_reshard_after_forward=False,
        parameter_mode="frozen-reference",
    )

    assert frozen.to_dict()["parameter_mode"] == "frozen-reference"
    assert frozen.to_dict()["parameter_mode"] == "frozen-reference"
    with pytest.raises(ValueError, match="cannot contain trainable"):
        FSDP2Application(
            parallel_plan=plan,
            block_module_names=("blocks.0",),
            block_class_names=("tests.Block",),
            parameter_names=("blocks.0.weight",),
            trainable_parameter_names=("blocks.0.weight",),
            parameter_count=4,
            trainable_parameter_count=4,
            param_dtype="bfloat16",
            reduce_dtype="float32",
            root_reshard_after_forward=False,
            parameter_mode="frozen-reference",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fsdp2_full_model_export_gathers_dtensors_and_strictly_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.tensor import DTensor

    from worldfoundry.training.engine.artifacts import export_full_model

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    monkeypatch.setenv("NCCL_DEBUG", "WARN")

    with DistributedTrainingContext(device_type="cuda") as context:
        adapter = _Adapter(context.device)
        plan = ParallelPlan.resolve(
            DistributedSpec(backend="fsdp2", dp_shard="auto"),
            world_size=context.world_size,
        )
        apply_fsdp2(
            adapter,
            plan=plan,
            mesh=plan.build_device_mesh(context.device.type),
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        assert all(isinstance(parameter, DTensor) for parameter in adapter.trainable_module.parameters())

        artifact = export_full_model(
            adapter.trainable_module,
            tmp_path / "full-model",
            distributed_context=context,
            role="test model",
            max_shard_size_bytes=128,
        )
        gathered = get_model_state_dict(
            adapter.trainable_module,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=False,
                strict=False,
            ),
        )

    assert gathered
    assert not any(isinstance(value, DTensor) for value in gathered.values())
    restored = _Adapter(torch.device("cpu")).trainable_module
    load_full_model(restored, artifact.path)
    for name, expected in gathered.items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fsdp2_application_engine_and_dcp_exact_next_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    torch.manual_seed(71)
    torch.cuda.manual_seed_all(73)

    with DistributedTrainingContext(device_type="cuda") as context:
        adapter = _Adapter(context.device)
        parameter_names_before = tuple(name for name, _ in adapter.trainable_module.named_parameters())
        plan = ParallelPlan.resolve(
            DistributedSpec(backend="fsdp2", dp_shard="auto"),
            world_size=context.world_size,
        )
        mesh = plan.build_device_mesh(context.device.type)
        application = apply_fsdp2(
            adapter,
            plan=plan,
            mesh=mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        assert application.block_module_names == ("blocks.0", "blocks.1")
        assert application.parameter_mode == "trainable"
        assert application.original_parameter_dtypes == ("bfloat16", "float32")
        assert application.precision_island_module_names == (
            "blocks.0.adapter_down:float32",
            "blocks.0.adapter_up:float32",
            "blocks.1.adapter_down:float32",
            "blocks.1.adapter_up:float32",
        )
        assert application.parameter_names == parameter_names_before
        assert application.parallel_plan == plan
        assert application.to_dict()["parallel_plan"] == plan.to_dict()
        assert all(
            isinstance(parameter, torch.distributed.tensor.DTensor)
            for parameter in adapter.trainable_module.parameters()
        )

        # The optimizer is intentionally constructed only after fully_shard.
        optimizer = build_adamw(
            trainable_parameters(adapter.trainable_module),
            learning_rate=0.01,
            fused=True,
        )
        objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
        engine = FSDP2TrainEngine(
            adapter,
            objective,
            optimizer,
            application=application,
            autocast_dtype=torch.bfloat16,
        )
        generator = torch.Generator(device=context.device).manual_seed(79)
        progress = TrainingProgress()
        dataloader = _StatefulCursor()
        state = TrainingState(
            model=adapter.trainable_module,
            optimizer=optimizer,
            engine=engine,
            dataloader=dataloader,
            objective_generator=generator,
            progress=progress,
            identity={
                "model": "tiny-fsdp2",
                "parallel_plan": plan.to_dict(),
                "fsdp2_application": application.to_dict(),
            },
            ignore_frozen_parameters=True,
        )

        first = engine.train_accumulation(
            (_batch("first", 0.0), _batch("second", 0.1)),
            generator=generator,
        )
        progress.record_step(
            microbatches=2,
            samples=first.sample_count,
            latent_tokens=first.latent_token_count,
        )
        assert first.sample_count == 2
        assert first.metrics["data_parallel_size"].item() == 1
        assert first.diagnostics["gradient_accumulation"] == "globally-token-weighted"
        manager = TrainingCheckpointer(tmp_path / "checkpoints")
        pending = manager.save(state, asynchronous=True)
        assert isinstance(pending, PendingTrainingCheckpoint)

        expected = engine.train_step(_batch("next", -0.1), generator=generator)
        progress.record_step(
            microbatches=1,
            samples=expected.sample_count,
            latent_tokens=expected.latent_token_count,
        )
        expected_parameters = _local_parameter_state(adapter.trainable_module)
        expected_loss = expected.loss.detach().clone()
        checkpoint = pending.wait()

        manager.load(state, checkpoint.path)
        actual = engine.train_step(_batch("next", -0.1), generator=generator)

        assert engine.global_step == 2
        assert progress.optimizer_steps == 1
        torch.testing.assert_close(actual.loss, expected_loss, rtol=0, atol=0)
        for name, value in _local_parameter_state(adapter.trainable_module).items():
            torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fsdp2_fp16_compute_keeps_fp32_shards_and_resumes_grad_scaler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    torch.manual_seed(83)
    torch.cuda.manual_seed_all(89)

    with DistributedTrainingContext(device_type="cuda") as context:
        adapter = _FP32Adapter(context.device)
        plan = ParallelPlan.resolve(
            DistributedSpec(backend="fsdp2", dp_shard="auto"),
            world_size=context.world_size,
        )
        application = apply_fsdp2(
            adapter,
            plan=plan,
            mesh=plan.build_device_mesh(context.device.type),
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            master_parameter_dtype=torch.float32,
        )
        optimizer = build_adamw(
            trainable_parameters(adapter.trainable_module),
            learning_rate=0.01,
            fused=True,
        )
        engine = FSDP2TrainEngine(
            adapter,
            FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform")),
            optimizer,
            application=application,
            autocast_dtype=torch.float16,
        )
        engine.grad_scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=128.0,
            growth_interval=1,
        )
        assert application.original_parameter_dtypes == ("float32",)
        assert application.precision_island_module_names == ()
        assert all(parameter.to_local().dtype is torch.float32 for parameter in engine.parameters)

        generator = torch.Generator(device=context.device).manual_seed(97)
        progress = TrainingProgress()
        state = TrainingState(
            model=adapter.trainable_module,
            optimizer=optimizer,
            engine=engine,
            dataloader=_StatefulCursor(),
            objective_generator=generator,
            progress=progress,
            identity={"model": "fp16-compute-fp32-master", "parallel_plan": plan.to_dict()},
            grad_scaler=engine.grad_scaler,
        )

        first = engine.train_step(_batch("first", 0.0), generator=generator)
        progress.record_step(
            microbatches=1,
            samples=first.sample_count,
            latent_tokens=first.latent_token_count,
        )
        manager = TrainingCheckpointer(tmp_path / "fp16-checkpoints")
        checkpoint = manager.save(state, asynchronous=False)

        expected = engine.train_step(_batch("next", 0.1), generator=generator)
        progress.record_step(
            microbatches=1,
            samples=expected.sample_count,
            latent_tokens=expected.latent_token_count,
        )
        expected_parameters = _local_parameter_state(adapter.trainable_module)
        expected_scaler = dict(engine.grad_scaler.state_dict())

        manager.load(state, checkpoint.path)
        actual = engine.train_step(_batch("next", 0.1), generator=generator)

        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        assert engine.grad_scaler.state_dict() == expected_scaler
        assert actual.diagnostics["grad_scaling"] is True
        for name, value in _local_parameter_state(adapter.trainable_module).items():
            torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
