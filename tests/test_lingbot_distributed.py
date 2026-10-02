from __future__ import annotations

import torch

from worldfoundry.synthesis.visual_generation.lingbot_world.lingbot_world_runtime import distributed


def _enable_distributed(monkeypatch, backend: str) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(distributed.dist, "barrier", lambda **kwargs: calls.append(kwargs))
    return calls


def test_nccl_barrier_receives_explicit_cuda_device(monkeypatch) -> None:
    calls = _enable_distributed(monkeypatch, "nccl")

    distributed.distributed_barrier(torch.device("cuda:3"))

    assert calls == [{"device_ids": [3]}]


def test_gloo_barrier_does_not_receive_cuda_device(monkeypatch) -> None:
    calls = _enable_distributed(monkeypatch, "gloo")

    distributed.distributed_barrier(torch.device("cuda:2"))

    assert calls == [{}]


def test_barrier_is_skipped_without_process_group(monkeypatch) -> None:
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    distributed.distributed_barrier(torch.device("cuda:0"))
