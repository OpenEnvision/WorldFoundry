# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Check checkpoint ownership, seeds, indexing, and finite dataset splitting."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from worldfoundry.core.utils.tensors import torch as torch_utils
from worldfoundry.core.utils.tensors.torch import (
    DDPMethodWrapper,
    get_seed,
    load_torch,
    sequential_split_dataset,
    set_seed_everywhere,
    torch_flatten_indices,
    torch_multi_index_select,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("prefix", ["", "scene."])
@pytest.mark.parametrize("keep_vars", [False, True])
def test_method_wrapper_preserves_parent_checkpoint_destination(prefix: str, keep_vars: bool) -> None:
    net = nn.Linear(2, 2)
    wrapper = DDPMethodWrapper(net, "forward")
    parent = nn.ModuleDict({"model": net, "method": wrapper})
    destination: dict[str, torch.Tensor] = {"existing": torch.tensor(7.0)}
    state = parent.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    assert state is destination
    assert set(state) == {"existing", f"{prefix}model.weight", f"{prefix}model.bias"}
    assert state[f"{prefix}model.weight"].requires_grad is keep_vars
    assert wrapper.state_dict() == {}
    assert wrapper.state_dict(destination=destination, prefix="method.", keep_vars=keep_vars) is destination
    value = torch.ones(1, 2)
    torch.testing.assert_close(wrapper(value), net(value))


def test_method_wrapper_accepts_module_positional_state_dict_arguments() -> None:
    wrapper = DDPMethodWrapper(nn.Linear(2, 2), "forward")
    destination: dict[str, torch.Tensor] = {"existing": torch.tensor(7.0)}
    with pytest.warns(FutureWarning, match="Positional args"):
        assert wrapper.state_dict(destination, "method.", True) is destination


def _checkpoint_roundtrip(tmp_path: Path, device: str) -> None:
    net = nn.Linear(2, 2, device=device)
    parent = nn.ModuleDict({"model": net, "method": DDPMethodWrapper(net, "forward")})
    path = tmp_path / "weights.pt"
    torch.save(parent.state_dict(), path)
    state = load_torch(str(tmp_path), path.name, map_location=device)
    assert set(state) == {"model.weight", "model.bias"}
    restored = nn.Linear(2, 2, device=device)
    restored.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    value = torch.tensor([[1.0, 2.0]], device=device)
    torch.testing.assert_close(restored(value), net(value))


def test_checkpoint_roundtrip_restores_the_weight_owner(tmp_path: Path) -> None:
    _checkpoint_roundtrip(tmp_path, "cpu")


@pytest.mark.gpu
def test_cuda_checkpoint_roundtrip_restores_the_weight_owner(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    _checkpoint_roundtrip(tmp_path, "cuda")


def test_load_torch_reports_the_joined_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import worldfoundry.core.model_loading as model_loading

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("checkpoint unreadable")

    monkeypatch.setattr(model_loading, "load_torch_checkpoint", fail)
    with pytest.raises(RuntimeError, match="checkpoint unreadable") as error:
        load_torch(str(tmp_path), "weights.pt")
    assert f"Error loading {tmp_path / 'weights.pt'}" in str(error.value)


@pytest.mark.parametrize("seed", [0, 7, None, -1, "system"])
@pytest.mark.parametrize("strategy", ["none", "system", "raise"])
def test_get_seed_preserves_resolution_policy(
    monkeypatch: pytest.MonkeyPatch, seed: int | str | None, strategy: Literal["none", "system", "raise"]
) -> None:
    monkeypatch.setattr(torch_utils.time, "time", lambda: 1234.5678)
    ticks = int(1234.5678 * 100000)
    generated = (
        ((ticks & 0xFF000000) >> 24)
        + ((ticks & 0x00FF0000) >> 8)
        + ((ticks & 0x0000FF00) << 8)
        + ((ticks & 0x000000FF) << 24)
    )
    if isinstance(seed, int) and seed >= 0:
        assert get_seed(seed, handle_invalid_seed=strategy) == seed
    elif seed == "system" or strategy == "system":
        assert get_seed(seed, handle_invalid_seed=strategy) == generated
    elif strategy == "none":
        assert get_seed(seed, handle_invalid_seed=strategy) is None
    else:
        with pytest.raises(ValueError, match="Invalid random seed"):
            get_seed(seed, handle_invalid_seed=strategy)


@pytest.fixture
def restore_random_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    monkeypatch.setattr(torch_utils, "set_deterministic", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args, **kwargs: None)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def test_seed_everywhere_reproduces_python_numpy_and_torch(restore_random_state: None) -> None:
    def sample() -> tuple[float, np.ndarray, torch.Tensor]:
        return random.random(), np.random.rand(3), torch.rand(3)

    assert set_seed_everywhere(17) == 17
    first = sample()
    assert set_seed_everywhere(17) == 17
    second = sample()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


@pytest.mark.parametrize("available", [False, True])
def test_tensorflow_seeding_remains_optional(
    monkeypatch: pytest.MonkeyPatch, restore_random_state: None, available: bool
) -> None:
    calls: list[int] = []
    if available:
        module = ModuleType("tensorflow")
        random_module = ModuleType("tensorflow.random")
        setattr(random_module, "set_seed", calls.append)
        setattr(module, "random", random_module)
        monkeypatch.setitem(sys.modules, "tensorflow", module)
    else:
        monkeypatch.setitem(sys.modules, "tensorflow", None)
    assert set_seed_everywhere(17, set_tensorflow=True) == 17
    assert calls == ([17] if available else [])


@pytest.mark.parametrize("shape", [(2, 3, 4), [2, 3, 4], torch.Size([2, 3, 4])])
def test_flatten_indices_accepts_multidimensional_shapes(shape: Sequence[int]) -> None:
    indices = torch.tensor([[0, 1, 2], [1, 2, 3]])
    torch.testing.assert_close(torch_flatten_indices(indices, shape), torch.tensor([6, 23]))


def _check_multi_index_select(device: str) -> None:
    values = torch.arange(24, device=device).reshape(2, 3, 4)
    for coordinates in ([[0], [1]], [[0, 1], [1, 2]], [[0, 1, 2], [1, 2, 3]]):
        indices = torch.tensor(coordinates, device=device)
        expected = values[tuple(indices[:, column] for column in range(indices.shape[1]))]
        actual = torch_multi_index_select(values, indices)
        torch.testing.assert_close(actual, expected)
        assert actual.device == values.device


def test_multi_index_select_matches_torch_indexing() -> None:
    _check_multi_index_select("cpu")


@pytest.mark.gpu
def test_multi_index_select_matches_cuda_indexing() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    _check_multi_index_select("cuda")


class _FiniteDataset(Dataset[int]):
    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int) -> int:
        return index


@pytest.mark.parametrize("factory", [_FiniteDataset, lambda: TensorDataset(torch.arange(7))])
@pytest.mark.parametrize("portions", [[0.5, 0.5], [0.2, 0.3, 0.5], [0.0, 1.0]])
def test_sequential_split_preserves_all_samples(factory: Callable[[], Dataset], portions: list[float]) -> None:
    dataset = factory()
    parts = sequential_split_dataset(dataset, portions)
    assert [index for part in parts for index in part.indices] == list(range(7))
    assert sum(len(part) for part in parts) == 7
    assert all(part.dataset is dataset for part in parts)
