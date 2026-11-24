from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.core.model_loading.checkpoints.sharded_safetensors import (
    load_safetensors_into_model_streaming,
    safetensor_checkpoint_files,
)


def test_canonical_model_excludes_processor_statistics(tmp_path) -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    save_file({"weight": model.weight.detach().clone()}, tmp_path / "model.safetensors")
    save_file({"action.mean": torch.zeros(2)}, tmp_path / "policy_normalizer.safetensors")

    assert safetensor_checkpoint_files(tmp_path) == [tmp_path / "model.safetensors"]
    report = load_safetensors_into_model_streaming(model, tmp_path, strict=True)
    assert report == {
        "files": 1,
        "loaded_keys": 1,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
    }


def test_unindexed_canonical_model_shards_exclude_other_safetensors(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    save_file({"weight": model.weight.detach().clone()}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({"bias": model.bias.detach().clone()}, tmp_path / "model-00002-of-00002.safetensors")
    save_file({"action.std": torch.ones(2)}, tmp_path / "policy_normalizer.safetensors")

    files = safetensor_checkpoint_files(tmp_path)
    assert [path.name for path in files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    report = load_safetensors_into_model_streaming(model, tmp_path, strict=True)
    assert report["files"] == 2
    assert report["missing_keys"] == 0
    assert report["unexpected_keys"] == 0


def test_multiple_noncanonical_safetensors_fail_closed(tmp_path) -> None:
    save_file({"a": torch.zeros(1)}, tmp_path / "weights-a.safetensors")
    save_file({"b": torch.zeros(1)}, tmp_path / "weights-b.safetensors")

    with pytest.raises(ValueError, match="ambiguous noncanonical"):
        safetensor_checkpoint_files(tmp_path)


def test_canonical_model_index_wins_over_sibling_index(tmp_path) -> None:
    save_file({"weight": torch.zeros(2, 3)}, tmp_path / "model-00001-of-00001.safetensors")
    save_file({"adapter": torch.zeros(1)}, tmp_path / "adapter.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "adapter.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"adapter": "adapter.safetensors"}}),
        encoding="utf-8",
    )

    assert safetensor_checkpoint_files(tmp_path) == [tmp_path / "model-00001-of-00001.safetensors"]


def test_multiple_noncanonical_indexes_fail_closed(tmp_path) -> None:
    for name in ("first", "second"):
        save_file({name: torch.zeros(1)}, tmp_path / f"{name}.safetensors")
        (tmp_path / f"{name}.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: f"{name}.safetensors"}}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="ambiguous safetensors indexes"):
        safetensor_checkpoint_files(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rename", [False, True])
def test_streaming_key_mapping_retains_assignment_and_dtype(
    tmp_path: Path, assign: bool, dtype: torch.dtype, rename: bool
) -> None:
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([0.25, -0.5])
    counts = torch.tensor([2, 3], dtype=torch.int64)
    prefix = "released." if rename else ""
    shards = {
        "model-00001-of-00002.safetensors": {prefix + "weight": weight},
        "model-00002-of-00002.safetensors": {prefix + "bias": bias, prefix + "counts": counts},
    }
    for filename, tensors in shards.items():
        save_file(tensors, tmp_path / filename)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for filename, tensors in shards.items() for key in tensors}}),
        encoding="utf-8",
    )
    device = "meta" if assign else "cpu"
    model = torch.nn.Linear(3, 2, device=device, dtype=dtype).requires_grad_(False)
    model.register_buffer("counts", torch.empty(2, dtype=torch.int64, device=device))
    previous_weight = model.weight
    report = load_safetensors_into_model_streaming(
        model,
        tmp_path,
        dtype=dtype,
        key_mapping={r"^released\.": ""} if rename else None,
    )
    assert report == {"files": 2, "loaded_keys": 3, "missing_keys": 0, "unexpected_keys": 0, "shape_mismatches": 0}
    assert (model.weight is previous_weight) == (not assign)
    assert not model.weight.requires_grad
    assert model.weight.dtype == dtype
    torch.testing.assert_close(model.weight, weight.to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(model.counts, counts, atol=0, rtol=0)
    inputs = torch.ones(1, 3, dtype=dtype)
    torch.testing.assert_close(
        model(inputs), torch.nn.functional.linear(inputs, weight.to(dtype), bias.to(dtype)), atol=0, rtol=0
    )


@pytest.mark.unit
@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("invalid", ["shape", "collision"])
def test_streaming_key_mapping_validates_canonical_destinations(tmp_path: Path, assign: bool, invalid: str) -> None:
    state = {"released.weight": torch.zeros(2, 3), "released.bias": torch.zeros(2)}
    if invalid == "shape":
        state["released.weight"] = torch.zeros(4, 3)
        message = "shape_mismatches|size mismatch"
    else:
        state["weight"] = torch.ones(2, 3)
        message = "collision|duplicate"
    save_file(state, tmp_path / "model.safetensors")
    model = torch.nn.Linear(3, 2, device="meta" if assign else "cpu")
    with pytest.raises((RuntimeError, ValueError), match=message):
        load_safetensors_into_model_streaming(model, tmp_path, key_mapping={r"^released\.": ""})
