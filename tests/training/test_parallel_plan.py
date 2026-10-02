from __future__ import annotations

import socket
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.distributed import (  # noqa: E402
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.recipes import DistributedSpec  # noqa: E402


def test_fsdp_parallel_plan_resolves_dynamic_world_size_and_named_dimensions() -> None:
    plan = ParallelPlan.resolve(
        DistributedSpec(
            backend="fsdp2",
            dp_replicate=2,
            dp_shard="auto",
            cp=1,
            tp=2,
        ),
        world_size=16,
    )

    assert plan.mesh_shape == (2, 4, 1, 2)
    assert plan.data_parallel_size == 8
    assert plan.to_dict()["mesh_dim_names"] == [
        "dp_replicate",
        "dp_shard",
        "cp",
        "tp",
    ]


def test_parallel_plan_rejects_geometry_that_does_not_match_the_launch() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ParallelPlan.resolve(
            DistributedSpec(
                backend="fsdp2",
                dp_replicate=2,
                dp_shard="auto",
                cp=2,
                tp=1,
            ),
            world_size=6,
        )
    with pytest.raises(ValueError, match="multiply to world_size"):
        ParallelPlan.resolve(
            DistributedSpec(
                backend="fsdp2",
                dp_replicate=1,
                dp_shard=2,
                cp=1,
                tp=1,
            ),
            world_size=4,
        )


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def test_world_size_one_context_builds_the_named_cpu_mesh(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    plan = ParallelPlan.resolve(
        DistributedSpec(backend="fsdp2", dp_shard="auto"),
        world_size=1,
    )

    with DistributedTrainingContext(device_type="cpu") as context:
        mesh = plan.build_device_mesh(context.device.type)
        fsdp_mesh = plan.fsdp_mesh(mesh)

        assert tuple(mesh.mesh.shape) == (1, 1, 1, 1)
        assert mesh.mesh_dim_names == ("dp_replicate", "dp_shard", "cp", "tp")
        assert tuple(fsdp_mesh.mesh.shape) == (1,)

    assert not torch.distributed.is_initialized()


@pytest.mark.parametrize("supports_device_id", (False, True))
def test_cuda_context_passes_device_id_only_when_supported(monkeypatch, supports_device_id: bool) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    selected_devices: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    calls: list[dict[str, object]] = []

    if supports_device_id:

        def init_process_group(*, backend, init_method, rank, world_size, timeout, device_id=None):
            calls.append(
                {
                    "backend": backend,
                    "init_method": init_method,
                    "rank": rank,
                    "world_size": world_size,
                    "timeout": timeout,
                    "device_id": device_id,
                }
            )

    else:

        def init_process_group(*, backend, init_method, rank, world_size, timeout):
            calls.append(
                {
                    "backend": backend,
                    "init_method": init_method,
                    "rank": rank,
                    "world_size": world_size,
                    "timeout": timeout,
                }
            )

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)

    context = DistributedTrainingContext(device_type="cuda", timeout_seconds=17)

    assert selected_devices == [0]
    assert calls[0]["timeout"] == timedelta(seconds=17)
    if supports_device_id:
        assert calls[0]["device_id"] == torch.device("cuda", 0)
    else:
        assert "device_id" not in calls[0]
    context.close()
