# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Lazy payload and bounded grouped-transform contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import torch

from worldfoundry.core.model_loading.checkpoints.transforms import (
    TensorGroupTransform,
    TensorTransform,
    TransformedStateDict,
    checkpoint_tensor_metadata,
)


class _TrackedState(Mapping[str, torch.Tensor]):
    def __init__(self) -> None:
        self.reads: list[str] = []
        self.tensors = {"first": torch.arange(4.0), "second": torch.ones(4)}

    def __getitem__(self, key: str) -> torch.Tensor:
        self.reads.append(key)
        return self.tensors[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)


def test_transformed_state_dict_membership_and_iteration_do_not_read_payloads() -> None:
    source = _TrackedState()
    converted = TransformedStateDict(
        source,
        {"renamed": "first", "sum": TensorTransform(("first", "second"), torch.add)},
    )
    assert "renamed" in converted
    assert "missing" not in converted
    assert list(converted) == ["renamed", "sum"]
    assert len(converted) == 2
    assert source.reads == []
    torch.testing.assert_close(converted["sum"], torch.arange(4.0) + 1, rtol=0, atol=0)
    assert source.reads == ["first", "second"]
    assert converted["renamed"] is source.tensors["first"]


def test_grouped_transform_reads_and_packs_once_then_releases_outputs() -> None:
    source = _TrackedState()
    calls: list[torch.Tensor] = []

    def pack(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        calls.append(value)
        return value + 1, value + 2, value + 3

    converted = TransformedStateDict(
        source,
        {str(index): TensorGroupTransform(("first",), pack, index) for index in range(3)},
    )
    for index in (2, 0, 1):
        torch.testing.assert_close(converted[str(index)], source.tensors["first"] + index + 1, rtol=0, atol=0)
    assert source.reads == ["first"]
    assert len(calls) == 1
    assert converted._group_values == ()
    converted["0"]
    assert source.reads == ["first", "first"]


def test_grouped_transform_retains_only_the_current_group() -> None:
    source = _TrackedState()

    def split(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return value[:2], value[2:]

    converted = TransformedStateDict(
        source,
        {f"{key}.{index}": TensorGroupTransform((key,), split, index) for key in source for index in range(2)},
    )
    converted["first.0"]
    converted["second.0"]
    converted["first.1"]
    assert source.reads == ["first", "second", "first"]
    assert len(converted._group_values) == 2


@pytest.mark.parametrize("suffix", ["safetensors", "pt"])
@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_disk_map_tensor_metadata_follows_renames_and_lookup_dtype_without_payload_reads(
    tmp_path: Path,
    suffix: str,
    dtype: torch.dtype | None,
) -> None:
    from safetensors.torch import save_file

    from worldfoundry.core.vram.disk_map import DiskMap

    checkpoint = tmp_path / f"weights.{suffix}"
    state = {"stored": torch.ones(4, 8, dtype=torch.float8_e4m3fn)}
    if suffix == "safetensors":
        save_file(state, str(checkpoint))
    else:
        torch.save(state, checkpoint)
    source = DiskMap(
        str(checkpoint),
        "cpu",
        torch_dtype=dtype,
        state_dict_converter=lambda state: {"public": state["stored"]},
    )
    converted = TransformedStateDict(source, {"renamed": "public"})
    assert checkpoint_tensor_metadata(converted, "renamed") == (dtype or torch.float8_e4m3fn, (4, 8))
    assert source.num_params == 0


def test_computed_transform_metadata_reflects_its_output() -> None:
    source = _TrackedState()
    converted = TransformedStateDict(source, {"sum": TensorTransform(("first", "second"), torch.add)})
    assert checkpoint_tensor_metadata(converted, "sum") == (torch.float32, (4,))
    assert source.reads == ["first", "second"]
