from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders.checkpoints import (
    CheckpointSpec,
)
from worldfoundry.base_models.diffusion_model.loaders.module import (
    ModuleLoadSpec,
    NativeModuleLoader,
)
from worldfoundry.core.model_loading.file import (
    load_keys_dict,
    load_state_dict,
)
from worldfoundry.core.model_loading.policy import (
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
)

gguf = pytest.importorskip("gguf")


class _TinyGGUFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(64, 64)


def _write_gguf(path, state_dict: dict[str, torch.Tensor]) -> None:
    writer = gguf.GGUFWriter(path, "worldfoundry-test")
    for name, tensor in state_dict.items():
        writer.add_tensor(name, tensor.detach().cpu().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_loader_reads_values_dtypes_and_logical_shapes(tmp_path) -> None:
    path = tmp_path / "tiny.gguf"
    expected = {
        "projection.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "projection.bias": torch.arange(3, dtype=torch.float16),
    }
    _write_gguf(path, expected)

    loaded = load_state_dict(path, torch_dtype=torch.bfloat16)

    assert load_keys_dict(str(path)) == {
        "projection.weight": [3, 4],
        "projection.bias": [3],
    }
    assert loaded["projection.weight"].dtype == torch.bfloat16
    assert loaded["projection.bias"].dtype == torch.bfloat16
    torch.testing.assert_close(
        loaded["projection.weight"].float(),
        expected["projection.weight"],
    )


def test_native_loader_consumes_gguf_and_installs_runtime_compression(tmp_path) -> None:
    path = tmp_path / "model.gguf"
    source = _TinyGGUFModel()
    _write_gguf(path, source.state_dict())

    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_TinyGGUFModel),
        CheckpointSpec(source=str(path)),
        RuntimePolicy(
            quantization=QuantizationPolicy(
                mode=QuantizationMode.GGUF,
                options={"min_features": 1, "runtime_bits": 4, "group_size": 32},
            )
        ),
    )

    assert type(loaded.projection).__name__ == "WeightOnlyLinear"
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["quantization"] == "gguf"
    assert snapshot.effective["quantization_storage"] == (
        "gguf-loaded+groupwise-int4"
    )
    assert loaded.projection(torch.randn(2, 64)).shape == (2, 64)
