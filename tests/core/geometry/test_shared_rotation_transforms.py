# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Exact tensor contracts for the shared perception camera transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldfoundry.core.geometry.transforms import (
    closed_form_inverse_se3,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_rotation_matrix,
    quaternion_wxyz_to_rotation_matrix_normalized,
    rotation_matrix_to_quaternion_wxyz_tensor,
    rotation_matrix_to_quaternion_wxyz_trace,
    rotation_matrix_to_unit_quaternion_wxyz,
    slerp_quaternions,
    standardize_quaternion_wxyz,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_wxyz_conversions_keep_sign_and_normalization_contracts(dtype: torch.dtype) -> None:
    q = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, -3, 0], [-2, 0, 0, 0]], dtype=dtype)
    r = quaternion_wxyz_to_rotation_matrix(q)
    expected = torch.stack(
        [torch.eye(3), torch.diag(torch.tensor([1, -1, -1])), torch.diag(torch.tensor([-1, 1, -1])), torch.eye(3)]
    ).to(dtype)
    torch.testing.assert_close(r, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        rotation_matrix_to_quaternion_wxyz_tensor(r), normalize_quaternion_wxyz(q.abs()), rtol=0, atol=0
    )
    torch.testing.assert_close(
        rotation_matrix_to_unit_quaternion_wxyz(r), normalize_quaternion_wxyz(q.abs()), rtol=0, atol=0
    )
    assert standardize_quaternion_wxyz(q)[-1, 0] == 2
    assert normalize_quaternion_wxyz(q)[-1, 0] == 1


def test_mirror_trace_branch_preserves_nonstandardized_quaternion_sign() -> None:
    q = torch.tensor([[-0.2, 0.9, 0.1, 0.1]])
    r = quaternion_wxyz_to_rotation_matrix_normalized(q)
    trace = rotation_matrix_to_quaternion_wxyz_trace(r)
    conditioned = rotation_matrix_to_quaternion_wxyz_tensor(r)
    assert trace[0, 0] < 0
    assert conditioned[0, 0] > 0
    torch.testing.assert_close(quaternion_wxyz_to_rotation_matrix_normalized(trace), r, rtol=1e-6, atol=1e-6)


def test_mirror_normalize_first_products_are_not_replaced_by_two_over_norm() -> None:
    torch.manual_seed(19)
    q = torch.randn(16, 4)
    assert not torch.equal(quaternion_wxyz_to_rotation_matrix_normalized(q), quaternion_wxyz_to_rotation_matrix(q))


@pytest.mark.parametrize("rows", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_inverse_se3_retains_torch_and_numpy_allocation(rows: int, dtype: torch.dtype) -> None:
    pose = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    pose[:, :3, 3] = torch.tensor([[1, 2, 3], [-1, -2, -3]], dtype=dtype)
    expected = pose.clone()
    expected[:, :3, 3] *= -1
    actual = closed_form_inverse_se3(pose[:, :rows])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        closed_form_inverse_se3(pose[:, :rows], pose[:, :3, :3], pose[:, :3, 3:]), expected, rtol=0, atol=0
    )
    numpy_result = closed_form_inverse_se3(pose[:, :rows].numpy())
    assert numpy_result.dtype == np.float64
    np.testing.assert_array_equal(numpy_result, expected.numpy())


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_slerp_handles_antipodal_nearby_and_distant_unit_pairs(fraction: float) -> None:
    q = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    other = torch.tensor([[-1.0, 0, 0, 0], [1.0, 0, 0, 0], [0.0, 1, 0, 0]])
    result = slerp_quaternions(q, other, fraction)
    torch.testing.assert_close(result[:2], q[:2], rtol=0, atol=0)
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(3), rtol=1e-6, atol=1e-6)
    assert torch.isfinite(result).all()
