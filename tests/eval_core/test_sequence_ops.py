from __future__ import annotations

import torch

import worldfoundry.core.distributed.sequence_parallel.ops as sequence_ops


def test_all_to_all_many_returns_inputs_without_distributed_runtime(monkeypatch) -> None:
    values = (torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
    monkeypatch.setattr(sequence_ops, "_distributed_ready", lambda: False)

    outputs = sequence_ops.all_to_all_many(values, scatter_dim=1, gather_dim=0)

    assert all(output is value for output, value in zip(outputs, values, strict=True))


def test_all_to_all_many_destination_pack_matches_rank_ordered_exchange(monkeypatch) -> None:
    world_size = 2
    destination_rank = 0
    source_values = tuple(
        tuple(
            torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)
            + source_rank * 1_000
            + tensor_index * 100
            for tensor_index in range(3)
        )
        for source_rank in range(world_size)
    )

    def packed_for_source(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        packed = values[0].new_empty(world_size, 2, 3, 1, 3, 2)
        for index, value in enumerate(values):
            packed[:, :, index].copy_(value.movedim(2, 0).unflatten(0, (world_size, 2)))
        return packed

    source_packs = tuple(packed_for_source(values) for values in source_values)

    def fake_all_to_all_single(output: torch.Tensor, _input: torch.Tensor, *, group=None) -> None:
        del group
        for source_rank in range(world_size):
            output[source_rank].copy_(source_packs[source_rank][destination_rank])

    monkeypatch.setattr(sequence_ops, "_distributed_ready", lambda: True)
    monkeypatch.setattr(sequence_ops.dist, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(sequence_ops.dist, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setenv("WORLDFOUNDRY_FUSED_QKV_A2A_MAX_MB", "512")

    outputs = sequence_ops.all_to_all_many(source_values[destination_rank], scatter_dim=-2, gather_dim=-3)
    expected = tuple(
        torch.cat(
            [values[tensor_index].chunk(world_size, dim=2)[destination_rank] for values in source_values],
            dim=1,
        )
        for tensor_index in range(3)
    )

    assert all(output.is_contiguous() for output in outputs)
    assert all(torch.equal(output, reference) for output, reference in zip(outputs, expected, strict=True))


def test_all_to_all_many_falls_back_for_incompatible_shapes(monkeypatch) -> None:
    values = (torch.randn(1, 2, 4), torch.randn(1, 3, 4))
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(sequence_ops, "_distributed_ready", lambda: True)
    monkeypatch.setattr(sequence_ops.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        sequence_ops,
        "all_to_all",
        lambda value, scatter_dim, gather_dim, group=None, **kwargs: calls.append(value) or value,
    )

    outputs = sequence_ops.all_to_all_many(values, scatter_dim=2, gather_dim=1)

    assert all(output is value for output, value in zip(outputs, values, strict=True))
    assert calls == list(values)
