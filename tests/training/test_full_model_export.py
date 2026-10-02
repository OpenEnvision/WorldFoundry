from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from worldfoundry.training.tuning import (
    FULL_MODEL_INDEX_NAME,
    FULL_MODEL_MANIFEST_NAME,
    inspect_full_model,
    load_full_model,
    save_full_model,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(4, 6)
        self.output = nn.Linear(6, 3, bias=False)
        self.register_buffer("scale", torch.tensor([0.25], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.input(value))) * self.scale


def test_full_model_export_is_sharded_and_strictly_loadable(tmp_path) -> None:
    torch.manual_seed(7)
    source = _TinyModel()
    destination = tmp_path / "full-model"

    artifact = save_full_model(
        source,
        destination,
        max_shard_size_bytes=80,
    )

    assert artifact == inspect_full_model(destination)
    assert artifact.tensor_count == len(source.state_dict())
    assert artifact.parameter_count == sum(value.numel() for value in source.parameters())
    assert artifact.trainable_parameter_count == artifact.parameter_count
    assert FULL_MODEL_INDEX_NAME in artifact.file_size_bytes
    manifest = json.loads((destination / FULL_MODEL_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == "worldfoundry-full-model"
    assert manifest["format"] == "safetensors"
    assert len(set(manifest["weight_map"].values())) >= 2
    assert "version" not in manifest

    restored = _TinyModel()
    for parameter in restored.parameters():
        parameter.data.zero_()
    load_full_model(restored, destination)
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)


def test_full_model_export_detects_payload_mutation(tmp_path) -> None:
    destination = tmp_path / "full-model"
    artifact = save_full_model(_TinyModel(), destination, max_shard_size_bytes=80)
    shard = next(destination / name for name in artifact.file_size_bytes if name.endswith(".safetensors"))

    with shard.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="payload verification failed"):
        inspect_full_model(destination)


def test_full_model_export_is_exclusive(tmp_path) -> None:
    destination = tmp_path / "full-model"
    save_full_model(_TinyModel(), destination)

    with pytest.raises(FileExistsError, match="already exists"):
        save_full_model(_TinyModel(), destination)
