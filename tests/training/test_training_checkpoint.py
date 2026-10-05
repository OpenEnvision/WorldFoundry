from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdata")

from worldfoundry.training.api import (  # noqa: E402
    ObjectiveBatch,
    PreparedBatch,
    TrainingBatch,
)
from worldfoundry.training.checkpoint import (  # noqa: E402
    IMMUTABLE_DTENSOR_ASYNC_STAGING,
    SYNCHRONOUS_DCP_STAGING,
    IncompleteTrainingCheckpointError,
    NamedStatefulCollection,
    PendingTrainingCheckpoint,
    TrainingCheckpointCompatibilityError,
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.checkpoint.state import (  # noqa: E402
    _cuda_rng_state_dict,
    _local_cuda_rng_state,
)
from worldfoundry.training.data import (  # noqa: E402
    DeterministicDistributedSampler,
    build_stateful_dataloader,
)
from worldfoundry.training.engine import (  # noqa: E402
    SingleDeviceTrainEngine,
    build_adamw,
    trainable_parameters,
)
from worldfoundry.training.objectives import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)


class _TinyAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes: tuple[type, ...] = ()

    def __init__(self, *, frozen_base: bool = False) -> None:
        layers: list[torch.nn.Module] = []
        if frozen_base:
            base = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)
            base.requires_grad_(False)
            layers.append(base)
        layers.extend(
            (
                torch.nn.Conv2d(1, 1, kernel_size=1, bias=True),
                torch.nn.Dropout2d(p=0.25),
            )
        )
        self.trainable_module = torch.nn.Sequential(*layers)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        assert isinstance(batch.pixel_values, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=batch.pixel_values[:, :, 0],
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


class _Dataset:
    def __init__(self) -> None:
        self.sample_ids = tuple(f"sample-{index}" for index in range(7))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> TrainingBatch:
        pixels = torch.tensor([[[[[0.1, 0.4], [0.7, -0.2]]]]]) + index * 0.03
        return TrainingBatch(
            sample_ids=(self.sample_ids[index],),
            prompts=("cached",),
            pixel_values=pixels,
        )


def _collate(values: list[TrainingBatch]) -> TrainingBatch:
    if len(values) != 1:
        raise ValueError("tiny checkpoint test expects batch size one")
    return values[0]


class _OptionalState:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: object) -> None:
        if not isinstance(state_dict, dict) or set(state_dict) != {"value"}:
            raise ValueError("optional state fields are invalid")
        self.value = int(state_dict["value"])


@dataclass
class _Stack:
    adapter: _TinyAdapter
    engine: SingleDeviceTrainEngine
    loader: object
    generator: torch.Generator
    progress: TrainingProgress
    state: TrainingState
    lr_scheduler: _OptionalState | None
    algorithm_state: _OptionalState | None


def _stack(
    *,
    identity_suffix: str = "a",
    environment_version: str = "torch-a",
    frozen_base: bool = False,
    lr_scheduler_value: int | None = None,
    algorithm_state_value: int | None = None,
) -> _Stack:
    torch.manual_seed(101)
    random.seed(202)
    adapter = _TinyAdapter(frozen_base=frozen_base)
    objective = FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform"))
    optimizer = build_adamw(
        trainable_parameters(adapter.trainable_module),
        learning_rate=0.01,
        fused=False,
    )
    engine = SingleDeviceTrainEngine(adapter, objective, optimizer)
    dataset = _Dataset()
    sampler = DeterministicDistributedSampler(
        dataset,
        seed=47,
        shuffle=True,
        rank=0,
        world_size=1,
        tail_policy="uneven",
    )
    loader = build_stateful_dataloader(
        dataset,
        sampler,
        batch_size=1,
        collate_fn=_collate,
        worker_seed=53,
    )
    generator = torch.Generator().manual_seed(307)
    progress = TrainingProgress()
    lr_scheduler = None if lr_scheduler_value is None else _OptionalState(lr_scheduler_value)
    algorithm_state = None if algorithm_state_value is None else _OptionalState(algorithm_state_value)
    state = TrainingState(
        model=adapter.trainable_module,
        optimizer=optimizer,
        engine=engine,
        dataloader=loader,
        objective_generator=generator,
        progress=progress,
        identity={
            "recipe": {"variant": identity_suffix},
            "sample_ids": list(dataset.sample_ids),
            "model_recipe": "sana-tiny",
            "prediction_type": "flow_velocity",
            "parallel_plan": {"backend": "single", "world_size": 1},
            "environment": {"torch": environment_version, "device_type": "cpu"},
        },
        ignore_frozen_parameters=frozen_base,
        lr_scheduler=lr_scheduler,
        algorithm_state=algorithm_state,
    )
    return _Stack(
        adapter,
        engine,
        loader,
        generator,
        progress,
        state,
        lr_scheduler,
        algorithm_state,
    )


def _step(stack: _Stack, iterator: object) -> tuple[TrainingBatch, object]:
    batch = next(iterator)
    result = stack.engine.train_step(batch, generator=stack.generator)
    stack.progress.record_step(
        microbatches=1,
        samples=result.sample_count,
        latent_tokens=result.latent_token_count,
    )
    return batch, result


def _parameter_state(stack: _Stack) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in stack.adapter.trainable_module.state_dict().items()}


def test_dcp_resume_restores_exact_next_data_rng_loss_and_update(tmp_path: Path) -> None:
    baseline = _stack()
    baseline_iterator = iter(baseline.loader)
    _step(baseline, baseline_iterator)
    _step(baseline, baseline_iterator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")

    artifact = manager.save(baseline.state)

    assert not isinstance(artifact, PendingTrainingCheckpoint)
    assert artifact.staging_strategy == SYNCHRONOUS_DCP_STAGING
    expected_python_random = random.random()
    expected_batch, expected_result = _step(baseline, baseline_iterator)
    expected_parameters = _parameter_state(baseline)
    expected_generator_state = baseline.generator.get_state().clone()

    restored = _stack()
    loaded = manager.load(restored.state)
    actual_python_random = random.random()
    restored_iterator = iter(restored.loader)
    actual_batch, actual_result = _step(restored, restored_iterator)

    assert loaded.path == artifact.path
    assert restored.engine.global_step == 3
    assert restored.progress.optimizer_steps == 3
    assert actual_python_random == expected_python_random
    assert actual_batch.sample_ids == expected_batch.sample_ids
    torch.testing.assert_close(actual_result.loss, expected_result.loss, rtol=0, atol=0)
    assert torch.equal(restored.generator.get_state(), expected_generator_state)
    for name, value in _parameter_state(restored).items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)


def test_async_dcp_save_commits_manifest_and_latest_pointer(tmp_path: Path) -> None:
    stack = _stack()
    iterator = iter(stack.loader)
    _step(stack, iterator)
    manager = TrainingCheckpointer(tmp_path / "checkpoints")

    pending = manager.save(stack.state, asynchronous=True)
    _step(stack, iterator)

    assert isinstance(pending, PendingTrainingCheckpoint)
    artifact = pending.wait()
    assert pending.wait() == artifact
    assert artifact.global_step == 1
    assert artifact.staging_strategy == IMMUTABLE_DTENSOR_ASYNC_STAGING
    assert (artifact.path / "_SUCCESS").is_file()
    manifest = json.loads((artifact.path / "checkpoint-manifest.json").read_text())
    assert manifest["staging_strategy"] == IMMUTABLE_DTENSOR_ASYNC_STAGING
    assert manager.inspect(artifact.path) == artifact
    restored = _stack()
    assert manager.load(restored.state, "latest").global_step == 1
    assert restored.progress.optimizer_steps == 1


def test_checkpoint_rejects_incomplete_modified_and_incompatible_state(tmp_path: Path) -> None:
    incomplete_stack = _stack()
    _step(incomplete_stack, iter(incomplete_stack.loader))
    incomplete_manager = TrainingCheckpointer(tmp_path / "incomplete")
    incomplete = incomplete_manager.save(incomplete_stack.state)
    assert not isinstance(incomplete, PendingTrainingCheckpoint)
    (incomplete.path / "_SUCCESS").unlink()
    with pytest.raises(IncompleteTrainingCheckpointError, match="no valid atomic commit"):
        incomplete_manager.load(_stack().state, incomplete.path)

    tampered_stack = _stack()
    _step(tampered_stack, iter(tampered_stack.loader))
    tampered_manager = TrainingCheckpointer(tmp_path / "tampered")
    tampered = tampered_manager.save(tampered_stack.state)
    assert not isinstance(tampered, PendingTrainingCheckpoint)
    payload = next(tampered.path.glob("*.distcp"))
    with payload.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(IncompleteTrainingCheckpointError, match="payload was modified"):
        tampered_manager.load(_stack().state, tampered.path)

    compatible_stack = _stack()
    _step(compatible_stack, iter(compatible_stack.loader))
    compatible_manager = TrainingCheckpointer(tmp_path / "incompatible")
    compatible = compatible_manager.save(compatible_stack.state)
    assert not isinstance(compatible, PendingTrainingCheckpoint)
    incompatible = _stack(identity_suffix="b")
    before = _parameter_state(incompatible)
    with pytest.raises(TrainingCheckpointCompatibilityError, match="identity differs"):
        compatible_manager.load(incompatible.state, compatible.path)
    for name, value in _parameter_state(incompatible).items():
        assert torch.equal(value, before[name])


def test_checkpoint_environment_drift_is_explicit_and_section_scoped(tmp_path: Path) -> None:
    source = _stack(environment_version="torch-old")
    _step(source, iter(source.loader))
    manager = TrainingCheckpointer(tmp_path / "environment-drift")
    artifact = manager.save(source.state)
    assert not isinstance(artifact, PendingTrainingCheckpoint)

    environment_drift = _stack(environment_version="torch-new")
    with pytest.raises(TrainingCheckpointCompatibilityError) as captured:
        manager.load(environment_drift.state, artifact.path)
    assert set(captured.value.section_differences) == {"environment"}
    assert '"environment"' in str(captured.value)
    assert captured.value.section_differences["environment"]["checkpoint"]["value"] == {
        "device_type": "cpu",
        "torch": "torch-old",
    }

    loaded = manager.load(
        environment_drift.state,
        artifact.path,
        allow_environment_drift=True,
    )
    assert loaded.path == artifact.path
    assert environment_drift.progress.optimizer_steps == 1

    semantic_drift = _stack(identity_suffix="different", environment_version="torch-new")
    with pytest.raises(TrainingCheckpointCompatibilityError) as semantic_error:
        manager.load(
            semantic_drift.state,
            artifact.path,
            allow_environment_drift=True,
        )
    assert set(semantic_error.value.section_differences) == {"recipe"}


@pytest.mark.parametrize(
    ("identity_shape", "recipe_schema", "difference_section"),
    (
        ("top-level", None, "distributed"),
        ("nested", "worldfoundry-training", "recipe"),
        ("nested", "worldfoundry-post-training", "recipe"),
    ),
)
def test_checkpoint_accepts_legacy_identity_without_default_distributed_timeout(
    identity_shape: str,
    recipe_schema: str | None,
    difference_section: str,
) -> None:
    stack = _stack()
    recipe_identity = {
        "schema": recipe_schema or "worldfoundry-training",
        "distributed": {"backend": "single", "timeout_seconds": 1800},
    }
    active_identity = (
        {"recipe": recipe_identity}
        if identity_shape == "nested"
        else {
            "schema": "worldfoundry-training-resume-identity",
            "distributed": recipe_identity["distributed"],
        }
    )
    stack.state.identity = MappingProxyType(active_identity)
    payload = stack.state.state_dict()
    payload["identity"] = json.loads(json.dumps(payload["identity"]))
    saved_distributed = (
        payload["identity"]["recipe"]["distributed"]
        if identity_shape == "nested"
        else payload["identity"]["distributed"]
    )
    saved_distributed.pop("timeout_seconds")

    stack.state.load_state_dict(payload)

    saved_distributed["timeout_seconds"] = 17
    with pytest.raises(TrainingCheckpointCompatibilityError) as captured:
        stack.state.load_state_dict(payload)
    assert set(captured.value.section_differences) == {difference_section}


def test_checkpoint_rejects_legacy_missing_timeout_for_nondefault_active_timeout() -> None:
    stack = _stack()
    active_identity = {
        "schema": "worldfoundry-training-resume-identity",
        "distributed": {"backend": "single", "timeout_seconds": 17},
    }
    stack.state.identity = MappingProxyType(active_identity)
    payload = stack.state.state_dict()
    payload["identity"] = json.loads(json.dumps(payload["identity"]))
    payload["identity"]["distributed"].pop("timeout_seconds")

    with pytest.raises(TrainingCheckpointCompatibilityError) as captured:
        stack.state.load_state_dict(payload)
    assert set(captured.value.section_differences) == {"distributed"}


def test_checkpoint_cuda_rng_is_rank_local_and_accepts_legacy_inventory(monkeypatch) -> None:
    states = {
        0: torch.arange(8, dtype=torch.uint8),
        1: torch.arange(8, dtype=torch.uint8) + 10,
        2: torch.arange(8, dtype=torch.uint8) + 20,
    }
    requested: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    def get_rng_state(device: int) -> torch.Tensor:
        requested.append(device)
        return states[device]

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_rng_state)

    saved = _cuda_rng_state_dict()

    assert requested == [1]
    assert set(saved) == {"device-0001"}
    assert torch.equal(saved["device-0001"], states[1])
    assert saved["device-0001"].data_ptr() != states[1].data_ptr()
    assert _local_cuda_rng_state(saved)[0] == 1

    legacy = {f"device-{index:04d}": state for index, state in states.items()}
    device_index, restored = _local_cuda_rng_state(legacy)
    assert device_index == 1
    assert torch.equal(restored, states[1])

    with pytest.raises(TrainingCheckpointCompatibilityError, match="inventory"):
        _local_cuda_rng_state({"device-0000": states[0], "device-0001": states[1]})


def test_checkpoint_refuses_to_overwrite_an_existing_step(tmp_path: Path) -> None:
    stack = _stack()
    manager = TrainingCheckpointer(tmp_path / "checkpoints")
    manager.save(stack.state)

    with pytest.raises(FileExistsError, match="already exists"):
        manager.save(stack.state)


def test_checkpoint_restores_optional_state_and_rejects_presence_drift(tmp_path: Path) -> None:
    source = _stack(lr_scheduler_value=17)
    _step(source, iter(source.loader))
    manager = TrainingCheckpointer(tmp_path / "optional-state")
    artifact = manager.save(source.state)
    assert not isinstance(artifact, PendingTrainingCheckpoint)

    restored = _stack(lr_scheduler_value=0)
    manager.load(restored.state, artifact.path)
    assert restored.lr_scheduler is not None
    assert restored.lr_scheduler.value == 17

    with pytest.raises(TrainingCheckpointCompatibilityError, match="presence differs.*lr_scheduler"):
        manager.load(_stack().state, artifact.path)


def test_checkpoint_restores_algorithm_state_and_named_component_inventory(tmp_path: Path) -> None:
    source = _stack(algorithm_state_value=23)
    _step(source, iter(source.loader))
    manager = TrainingCheckpointer(tmp_path / "algorithm-state")
    artifact = manager.save(source.state)
    assert not isinstance(artifact, PendingTrainingCheckpoint)
    assert artifact.optional_state_presence["algorithm_state"] is True

    restored = _stack(algorithm_state_value=0)
    manager.load(restored.state, artifact.path)
    assert restored.algorithm_state is not None
    assert restored.algorithm_state.value == 23

    first = _OptionalState(3)
    second = _OptionalState(7)
    collection = NamedStatefulCollection({"second": second, "first": first})
    saved = collection.state_dict()
    first.value = 0
    second.value = 0
    collection.load_state_dict(saved)
    assert collection.component_names == ("first", "second")
    assert (first.value, second.value) == (3, 7)

    incompatible = NamedStatefulCollection({"first": _OptionalState(0)})
    with pytest.raises(ValueError, match="inventory differs"):
        incompatible.load_state_dict(saved)


def test_adapter_only_checkpoint_strictly_restores_trainable_keys(tmp_path: Path) -> None:
    baseline = _stack(frozen_base=True)
    iterator = iter(baseline.loader)
    _step(baseline, iterator)
    manager = TrainingCheckpointer(tmp_path / "adapter-checkpoints")
    artifact = manager.save(baseline.state)
    assert not isinstance(artifact, PendingTrainingCheckpoint)

    expected_batch, expected_result = _step(baseline, iterator)
    expected_parameters = _parameter_state(baseline)
    restored = _stack(frozen_base=True)
    manager.load(restored.state, artifact.path)
    actual_batch, actual_result = _step(restored, iter(restored.loader))

    assert actual_batch.sample_ids == expected_batch.sample_ids
    torch.testing.assert_close(actual_result.loss, expected_result.loss, rtol=0, atol=0)
    for name, value in _parameter_state(restored).items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
