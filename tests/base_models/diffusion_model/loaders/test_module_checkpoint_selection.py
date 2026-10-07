"""Joint checkpoints materialize only the weights owned by each component."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.loaders import (
    CheckpointSpec,
    ModuleLoadSpec,
    NativeModuleLoader,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
    convert_ltx_video_encoder_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import (
    convert_ltx_transformer_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import (
    convert_ltx_embedding_processor_state_dict,
    convert_ltx_gemma_state_dict,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.core.vram.disk_map import DiskMap


class _SelectedWeights(Mapping):
    def __init__(self, key):
        self.key = key
        self.weight = object()

    def __iter__(self):
        return iter(("unrelated.large_weight", self.key))

    def __len__(self):
        return 2

    def __getitem__(self, key):
        assert key == self.key, f"Materialized another component's weight: {key}"
        return self.weight


@pytest.mark.parametrize(
    ("converter", "key", "destination"),
    [
        (convert_ltx_transformer_state_dict, "model.diffusion_model.proj.weight", "velocity_model.proj.weight"),
        (
            convert_ltx_embedding_processor_state_dict,
            "text_embedding_projection.aggregate_embed.weight",
            "processor.feature_extractor.aggregate_embed.weight",
        ),
        (convert_ltx_video_encoder_state_dict, "vae.encoder.weight", "encoder.weight"),
        (
            convert_ltx_gemma_state_dict,
            "language_model.model.embed_tokens.weight",
            "model.model.language_model.embed_tokens.weight",
        ),
    ],
)
def test_ltx_conversion_does_not_materialize_unrelated_components(converter, key, destination):
    source = _SelectedWeights(key)
    converted = converter(source)
    assert converted[destination] is source.weight


def test_native_loader_selects_component_before_reading_checkpoint_tensors(tmp_path, monkeypatch):
    class VideoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)

    expected = VideoEncoder().eval()
    checkpoint = tmp_path / "joint.safetensors"
    tensors = {f"vae.{name}": value for name, value in expected.state_dict().items()}
    tensors["model.diffusion_model.other.weight"] = torch.ones(16, 16)
    save_file(tensors, str(checkpoint))
    reads = []
    get_tensor = DiskMap.__getitem__

    def record_read(mapping, key):
        reads.append(key)
        return get_tensor(mapping, key)

    monkeypatch.setattr(DiskMap, "__getitem__", record_read)
    restored = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=VideoEncoder,
            state_dict_converter=convert_ltx_video_encoder_state_dict,
        ),
        CheckpointSpec(source=str(checkpoint)),
        RuntimePolicy(dtype=torch.float32),
    )
    assert set(reads) == {"vae.encoder.weight", "vae.encoder.bias"}
    sample = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(restored.encoder(sample), expected.encoder(sample), atol=0, rtol=0)
