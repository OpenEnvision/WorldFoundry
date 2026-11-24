# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA communicator operation selection without loading NCCL or initializing CUDA."""

from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch
from torch.distributed import ReduceOp

from worldfoundry.core.distributed.sequence_parallel.device_communicators.cuda_communicator import CudaCommunicator
from worldfoundry.core.distributed.sequence_parallel.device_communicators.pynccl_wrapper import ncclRedOpTypeEnum

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ["default", "none", "sum", "max"])
@pytest.mark.parametrize("fallback", [False, True])
def test_all_reduce_operation_and_out_of_place_fallback(
    operation: Literal["default", "none", "sum", "max"],
    fallback: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.arange(4, dtype=torch.float32)
    original = value.clone()
    expected_op = ReduceOp.MAX if operation == "max" else ReduceOp.SUM
    nccl_calls: list[ReduceOp.RedOpType] = []
    torch_calls: list[ReduceOp.RedOpType] = []
    group = object()

    def nccl_all_reduce(tensor: torch.Tensor, op: ReduceOp.RedOpType) -> torch.Tensor | None:
        assert tensor is value
        assert ncclRedOpTypeEnum.from_torch(op) == ncclRedOpTypeEnum.from_torch(expected_op)
        nccl_calls.append(op)
        return None if fallback else tensor.clone()

    def torch_all_reduce(tensor: torch.Tensor, *, group: object, op: ReduceOp.RedOpType) -> None:
        assert group is communicator.device_group
        assert tensor.data_ptr() != value.data_ptr()
        torch_calls.append(op)
        tensor.add_(1)

    communicator = object.__new__(CudaCommunicator)
    cast(Any, communicator).pynccl_comm = SimpleNamespace(all_reduce=nccl_all_reduce)
    cast(Any, communicator).device_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", torch_all_reduce)

    if operation == "default":
        output = communicator.all_reduce(value)
    else:
        output = communicator.all_reduce(value, op=None if operation == "none" else expected_op)

    assert nccl_calls == [expected_op]
    assert torch_calls == ([expected_op] if fallback else [])
    assert output.data_ptr() != value.data_ptr()
    torch.testing.assert_close(value, original)
    torch.testing.assert_close(output, original + 1 if fallback else original)
