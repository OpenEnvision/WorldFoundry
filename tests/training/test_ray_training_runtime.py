from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.training.distributed.ray_runtime import (
    DevicePoolPlanner,
    RayDevicePool,
    RayDevicePoolConfig,
    RayRoleWorker,
    RayWorkerContext,
    RolloutPlacement,
)
from worldfoundry.training.distributed.rollout_runtime import (
    RayPostTrainingRuntime,
    RayPostTrainingRuntimeConfig,
    TrainerBinding,
)


class _AccumulatorRole:
    def __init__(self, *, context: RayWorkerContext, offset: int = 0) -> None:
        self.context = context
        self.offset = offset

    def add(self, value: int) -> tuple[int, int, int]:
        return self.context.rank, self.context.slot, value + self.offset


def _sync_role_factories():
    class SyncModel(nn.Module):
        def __init__(self, *, base: float, adapter: float) -> None:
            super().__init__()
            self.base = nn.Linear(2, 2, bias=False)
            self.lora_A = nn.Parameter(torch.full((2, 2), adapter))
            self.lora_B = nn.Parameter(torch.full((2, 2), adapter))
            with torch.no_grad():
                self.base.weight.fill_(base)

    class TrainerRole:
        def __init__(self, *, context: RayWorkerContext) -> None:
            self.context = context
            self.model = SyncModel(base=1.0, adapter=2.0)

        def fill(self, *, base: float | None = None, adapter: float | None = None) -> None:
            with torch.no_grad():
                if base is not None:
                    self.model.base.weight.fill_(base)
                if adapter is not None:
                    self.model.lora_A.fill_(adapter)
                    self.model.lora_B.fill_(adapter)

        def placement(self) -> tuple[int, int, int, int]:
            return (
                self.context.placement_group_index,
                self.context.bundle_index,
                self.context.device_id,
                self.context.slot,
            )

    class RolloutRole:
        def __init__(self, *, context: RayWorkerContext) -> None:
            self.context = context
            self.model = SyncModel(base=-1.0, adapter=-2.0)

        def state(self) -> dict[str, torch.Tensor]:
            return {name: value.clone() for name, value in self.model.state_dict().items()}

        def placement(self) -> tuple[int, int, int, int]:
            return (
                self.context.placement_group_index,
                self.context.bundle_index,
                self.context.device_id,
                self.context.slot,
            )

    class RewardRole:
        def __init__(self, *, context: RayWorkerContext) -> None:
            self.context = context

    return TrainerRole, RolloutRole, RewardRole


def test_device_pool_plans_separate_and_colocated_rollout_slabs() -> None:
    planner = DevicePoolPlanner(
        RayDevicePoolConfig(
            num_devices=10,
            devices_per_node=8,
            workers_per_device=3,
        )
    )
    trainer = planner.reserve("trainer", 4)
    rollout = planner.reserve("rollout", 3)
    colocated = planner.reserve(
        "fast-rollout",
        4,
        placement=RolloutPlacement.COLOCATE,
        colocate_with="trainer",
    )
    reward = planner.reserve(
        "reward",
        2,
        placement=RolloutPlacement.COLOCATE,
        colocate_with="trainer",
    )

    assert trainer.device_ids == (0, 1, 2, 3)
    assert rollout.device_ids == (4, 5, 6)
    assert rollout.placement is RolloutPlacement.SEPARATE
    assert colocated.device_ids == trainer.device_ids
    assert colocated.slot == 1
    assert reward.device_ids == (0, 1)
    assert reward.slot == 2


def test_local_ray_role_worker_invokes_caller_owned_role() -> None:
    context = RayWorkerContext(role="rollout", rank=1, world_size=2, device_id=3, slot=1)
    worker = RayRoleWorker(_AccumulatorRole, context, {"offset": 5})
    assert worker.invoke("add", (7,), {}) == (1, 1, 12)


def test_ray_device_pool_executes_cpu_colocated_workers() -> None:
    pytest.importorskip("ray")
    config = RayDevicePoolConfig(
        num_devices=1,
        devices_per_node=1,
        workers_per_device=2,
        cpus_per_worker=0.25,
        accelerator_resource="CPU",
    )
    with RayDevicePool(config) as pool:
        trainer = pool.reserve("trainer", 1)
        rollout = pool.reserve(
            "rollout",
            1,
            placement="colocate",
            colocate_with="trainer",
        )

        def factory(*, context, offset=0):
            return SimpleNamespace(add=lambda value: (context.rank, context.slot, value + offset))

        trainer_group = pool.create_worker_group(
            trainer,
            factory,
            factory_kwargs={"offset": 2},
        )
        rollout_group = pool.create_worker_group(
            rollout,
            factory,
            factory_kwargs={"offset": 10},
        )
        assert trainer_group.broadcast("add", 3) == ((0, 0, 5),)
        assert rollout_group.gather([rollout_group.submit("add", 3)]) == ((0, 1, 13),)


def test_external_trainer_binding_rejects_fake_colocate() -> None:
    _, rollout_factory, _ = _sync_role_factories()
    runtime = RayPostTrainingRuntime(
        RayPostTrainingRuntimeConfig(
            pool=RayDevicePoolConfig(
                num_devices=1,
                devices_per_node=1,
                workers_per_device=2,
                cpus_per_worker=0.25,
                accelerator_resource="CPU",
            ),
            rollout_devices=1,
            rollout_placement=RolloutPlacement.COLOCATE,
        )
    )
    with pytest.raises(ValueError, match="actor-hosted trainer"):
        runtime.setup(rollout_factory)


def test_actor_runtime_rejects_colocated_rollout_larger_than_trainer_group() -> None:
    with pytest.raises(ValueError, match="cannot exceed trainer_devices"):
        RayPostTrainingRuntimeConfig(
            pool=RayDevicePoolConfig(
                num_devices=2,
                devices_per_node=2,
                workers_per_device=2,
            ),
            trainer_devices=1,
            rollout_devices=2,
            rollout_placement=RolloutPlacement.COLOCATE,
            trainer_binding=TrainerBinding.ACTOR,
        )


def test_actor_runtime_places_separate_roles_and_syncs_full_then_lora_weights() -> None:
    ray = pytest.importorskip("ray")
    trainer_factory, rollout_factory, reward_factory = _sync_role_factories()
    runtime = RayPostTrainingRuntime(
        RayPostTrainingRuntimeConfig(
            pool=RayDevicePoolConfig(
                num_devices=3,
                devices_per_node=3,
                workers_per_device=1,
                cpus_per_worker=0.25,
                accelerator_resource="CPU",
            ),
            trainer_devices=1,
            rollout_devices=1,
            rollout_placement=RolloutPlacement.SEPARATE,
            reward_devices=1,
            reward_placement=RolloutPlacement.SEPARATE,
            trainer_binding=TrainerBinding.ACTOR,
            weight_bucket_bytes=16,
        )
    )
    try:
        runtime.setup(
            rollout_factory,
            trainer_factory=trainer_factory,
            reward_factory=reward_factory,
        )
        assert runtime.trainer_group is not None
        assert runtime.rollout_group is not None
        assert runtime.reward_group is not None
        trainer_place = runtime.trainer_group.broadcast("placement")[0]
        rollout_place = runtime.rollout_group.broadcast("placement")[0]
        assert trainer_place[:2] == (0, 0)
        assert rollout_place[:2] == (0, 1)
        assert runtime.reward_group.lease.device_ids == (2,)

        full_report = runtime.sync_rollout_weights(revision=0)
        assert full_report.transmitted and full_report.bucket_count > 1
        full_state = runtime.rollout_group.broadcast("state")[0]
        assert all(
            torch.equal(value, torch.full_like(value, 1.0 if "base" in name else 2.0))
            for name, value in full_state.items()
        )

        runtime.trainer_group.broadcast("fill", base=7.0, adapter=5.0)
        lora_report = runtime.sync_rollout_weights(revision=1, kind="lora")
        assert lora_report.kind.value == "lora"
        lora_state = runtime.rollout_group.broadcast("state")[0]
        assert torch.equal(lora_state["base.weight"], torch.ones_like(lora_state["base.weight"]))
        assert torch.equal(lora_state["lora_A"], torch.full_like(lora_state["lora_A"], 5.0))
        assert torch.equal(lora_state["lora_B"], torch.full_like(lora_state["lora_B"], 5.0))
        with pytest.raises(ray.exceptions.RayTaskError, match="not newer"):
            runtime.sync_rollout_weights(revision=1, kind="lora")
    finally:
        runtime.shutdown()


def test_actor_runtime_physically_colocates_roles_on_distinct_fractional_slots() -> None:
    pytest.importorskip("ray")
    trainer_factory, rollout_factory, reward_factory = _sync_role_factories()
    runtime = RayPostTrainingRuntime(
        RayPostTrainingRuntimeConfig(
            pool=RayDevicePoolConfig(
                num_devices=1,
                devices_per_node=1,
                workers_per_device=3,
                cpus_per_worker=0.25,
                accelerator_resource="CPU",
            ),
            trainer_devices=1,
            rollout_devices=1,
            rollout_placement=RolloutPlacement.COLOCATE,
            reward_devices=1,
            reward_placement=RolloutPlacement.COLOCATE,
            trainer_binding=TrainerBinding.ACTOR,
        )
    )
    try:
        runtime.setup(
            rollout_factory,
            trainer_factory=trainer_factory,
            reward_factory=reward_factory,
        )
        assert runtime.trainer_group is not None
        assert runtime.rollout_group is not None
        assert runtime.reward_group is not None
        trainer_place = runtime.trainer_group.broadcast("placement")[0]
        rollout_place = runtime.rollout_group.broadcast("placement")[0]
        assert trainer_place[:3] == rollout_place[:3] == (0, 0, 0)
        assert (trainer_place[3], rollout_place[3], runtime.reward_group.lease.slot) == (0, 1, 2)
        runtime.sync_rollout_weights(revision=0)
    finally:
        runtime.shutdown()
