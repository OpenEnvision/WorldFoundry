# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare scalar and multidimensional streaming statistics with batch reductions."""

from __future__ import annotations

from collections.abc import Callable
from math import prod

import numpy as np
import pytest
import torch

from worldfoundry.core.utils.tensors.torch import RunningMeanStd

pytestmark = pytest.mark.unit

Batch = np.ndarray | torch.Tensor
FEATURE_SHAPES = [(), (2,), (2, 3)]


def _check_statistics(
    factory: Callable[[np.ndarray], Batch], *, from_moments: bool, feature_shape: tuple[int, ...]
) -> None:
    stats = RunningMeanStd()
    assert stats.mean is None
    assert stats.var is None
    assert stats.std is None
    assert stats.count == 0
    data = np.arange(1, 3 * prod(feature_shape) + 1, dtype=np.float64).reshape((3, *feature_shape))
    start = 0

    for end in (2, 3):
        batch = factory(data[start:end])
        if from_moments:
            if isinstance(batch, torch.Tensor):
                stats.update_from_moments(batch.mean(dim=0), batch.var(dim=0, unbiased=False), len(batch))
            else:
                stats.update_from_moments(batch.mean(axis=0), batch.var(axis=0), len(batch))
        else:
            stats.update(batch)

        expected = data[:end]
        if isinstance(batch, torch.Tensor):
            assert isinstance(stats.mean, torch.Tensor)
            assert isinstance(stats.var, torch.Tensor)
            assert isinstance(stats.std, torch.Tensor)
            assert stats.mean.dtype == stats.var.dtype == stats.std.dtype == batch.dtype
            assert stats.mean.device == stats.var.device == stats.std.device == batch.device
            torch.testing.assert_close(stats.mean, batch.new_tensor(expected.mean(axis=0)))
            torch.testing.assert_close(stats.var, batch.new_tensor(expected.var(axis=0)))
            torch.testing.assert_close(stats.std, batch.new_tensor(expected.std(axis=0)))
        else:
            assert isinstance(stats.mean, (np.ndarray, np.number))
            assert isinstance(stats.var, (np.ndarray, np.number))
            assert isinstance(stats.std, (np.ndarray, np.number))
            np.testing.assert_allclose(stats.mean, expected.mean(axis=0))
            np.testing.assert_allclose(stats.var, expected.var(axis=0))
            np.testing.assert_allclose(stats.std, expected.std(axis=0))
        assert stats.count == end
        start = end


def test_running_mean_std_starts_without_statistics() -> None:
    stats = RunningMeanStd()
    assert stats.mean is stats.var is stats.std is None
    assert stats.count == 0


def test_running_mean_std_handles_numpy_scalar_features() -> None:
    stats = RunningMeanStd()
    stats.update(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(stats.mean, 2.0)
    np.testing.assert_allclose(stats.var, 2.0 / 3.0)
    std = stats.std
    assert isinstance(std, (np.ndarray, np.number))
    np.testing.assert_allclose(std, np.sqrt(2.0 / 3.0))
    assert stats.count == 3


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("from_moments", [False, True], ids=["batch", "moments"])
@pytest.mark.parametrize("feature_shape", FEATURE_SHAPES, ids=["scalar", "vector", "matrix"])
def test_running_mean_std_matches_batch_statistics(
    backend: str, from_moments: bool, feature_shape: tuple[int, ...]
) -> None:
    def factory(values: np.ndarray) -> Batch:
        return values.copy() if backend == "numpy" else torch.tensor(values, dtype=torch.float64)

    _check_statistics(factory, from_moments=from_moments, feature_shape=feature_shape)


@pytest.mark.gpu
@pytest.mark.parametrize("from_moments", [False, True], ids=["batch", "moments"])
@pytest.mark.parametrize("feature_shape", FEATURE_SHAPES, ids=["scalar", "vector", "matrix"])
def test_running_mean_std_preserves_cuda_device_and_dtype(from_moments: bool, feature_shape: tuple[int, ...]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")

    def factory(values: np.ndarray) -> torch.Tensor:
        return torch.tensor(values, device="cuda", dtype=torch.float64)

    _check_statistics(factory, from_moments=from_moments, feature_shape=feature_shape)
