from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.loaders.materialize import (
    MaterializedCheckpoint,
)
from worldfoundry.base_models.diffusion_model.loaders.wan_components import (
    load_wan_transformer_checkpoint,
)
from worldfoundry.base_models.diffusion_model.loaders.wan_vsa import (
    convert_wan_vsa_gate_state_dict,
    resolve_wan_vsa_gate_config,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    WanModel,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    approximate_attention as approximate_module,
)
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    approximate_attention_report,
    install_approximate_attention,
)

_TINY_CONFIG = {
    "dim": 128,
    "in_dim": 4,
    "ffn_dim": 192,
    "out_dim": 4,
    "text_dim": 64,
    "freq_dim": 32,
    "eps": 1e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 1,
    "num_layers": 2,
    "has_image_input": False,
    "require_vae_embedding": False,
    "require_clip_embedding": False,
}


def _write_tiny_checkpoint(
    path: Path,
    *,
    with_gates: bool,
    gate_bias: float = 0.0,
) -> dict[str, torch.Tensor]:
    model = WanModel(
        **_TINY_CONFIG,
        vsa_gate_compress=with_gates,
    ).eval()
    if with_gates:
        for block in model.blocks:
            torch.nn.init.zeros_(block.self_attn.gate_compress.weight)
            torch.nn.init.constant_(
                block.self_attn.gate_compress.bias,
                gate_bias,
            )
    native = {
        name: value.detach().contiguous()
        for name, value in model.state_dict().items()
    }
    released: dict[str, torch.Tensor] = {}
    for name, value in native.items():
        if ".self_attn.gate_compress." in name:
            name = name.replace(
                ".self_attn.gate_compress.",
                ".to_gate_compress.",
            )
        released[name] = value
    save_file(released, path)
    return native


def _materialized(path: Path) -> MaterializedCheckpoint:
    return MaterializedCheckpoint(root=path.parent, paths=(path,))


def test_dense_checkpoint_does_not_construct_random_vsa_gates(tmp_path) -> None:
    path = tmp_path / "dense.safetensors"
    _write_tiny_checkpoint(path, with_gates=False)

    loaded = load_wan_transformer_checkpoint(
        path,
        torch_dtype=torch.float32,
        transformer_config=dict(_TINY_CONFIG),
    )

    assert loaded.vsa_gate_compress is False
    assert all(
        not hasattr(block.self_attn, "gate_compress")
        for block in loaded.blocks
    )


def test_complete_qat_checkpoint_constructs_and_loads_every_gate(tmp_path) -> None:
    path = tmp_path / "vsa.safetensors"
    expected = _write_tiny_checkpoint(path, with_gates=True, gate_bias=0.375)

    loaded = load_wan_transformer_checkpoint(
        path,
        torch_dtype=torch.float32,
        transformer_config=dict(_TINY_CONFIG),
    )

    assert loaded.vsa_gate_compress is True
    for index, block in enumerate(loaded.blocks):
        prefix = f"blocks.{index}.self_attn.gate_compress"
        torch.testing.assert_close(
            block.self_attn.gate_compress.weight,
            expected[f"{prefix}.weight"],
        )
        torch.testing.assert_close(
            block.self_attn.gate_compress.bias,
            expected[f"{prefix}.bias"],
        )


@pytest.mark.parametrize(
    "gate_tensors,match",
    (
        (
            {
                "blocks.0.to_gate_compress.weight": torch.zeros(128, 128),
                "blocks.0.to_gate_compress.bias": torch.zeros(128),
            },
            "weight and bias for every layer",
        ),
        (
            {
                "blocks.0.to_gate_compress.weight": torch.zeros(127, 128),
                "blocks.0.to_gate_compress.bias": torch.zeros(128),
                "blocks.1.to_gate_compress.weight": torch.zeros(128, 128),
                "blocks.1.to_gate_compress.bias": torch.zeros(128),
            },
            "shape mismatch",
        ),
    ),
)
def test_partial_or_wrong_shape_qat_headers_fail_before_construction(
    tmp_path,
    gate_tensors,
    match,
) -> None:
    path = tmp_path / "malformed.safetensors"
    save_file(gate_tensors, path)

    with pytest.raises(ValueError, match=match):
        resolve_wan_vsa_gate_config(
            _materialized(path),
            dim=128,
            num_layers=2,
        )


def test_duplicate_gate_aliases_are_rejected() -> None:
    with pytest.raises(KeyError, match="multiple parameters|duplicate parameter"):
        convert_wan_vsa_gate_state_dict(
            {
                "blocks.0.to_gate_compress.weight": torch.zeros(128, 128),
                "blocks.0.self_attn.gate_compress.weight": torch.zeros(128, 128),
            }
        )


def test_malformed_gate_key_is_not_silently_ignored() -> None:
    with pytest.raises(KeyError, match="unsupported Wan VSA gate parameter"):
        convert_wan_vsa_gate_state_dict(
            {"blocks.0.to_gate_compress.kernel": torch.zeros(128, 128)}
        )


class _GateCaptureOps:
    def __init__(self) -> None:
        self.gate: torch.Tensor | None = None

    def video_sparse_attn(
        self,
        q,
        k,
        v,
        *,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        del k, v, variable_block_sizes, q_variable_block_sizes, topk, block_size
        self.gate = compress_attn_weight.detach().clone()
        return q


def test_loaded_gate_reaches_vsa_provider_receipt(monkeypatch, tmp_path) -> None:
    path = tmp_path / "vsa-runtime.safetensors"
    _write_tiny_checkpoint(path, with_gates=True, gate_bias=0.375)
    loaded = load_wan_transformer_checkpoint(
        path,
        torch_dtype=torch.float32,
        transformer_config=dict(_TINY_CONFIG),
    )
    ops = _GateCaptureOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    state = install_approximate_attention(
        loaded,
        ApproximateAttentionConfig(
            kind="vsa",
            sparsity=0.5,
            block_tile=(4, 4, 4),
        ),
    )
    attention = loaded.blocks[0].self_attn
    values = torch.randn(1, 32, 128)

    output = attention.get_processor()._sparse_attention(
        attention,
        values,
        values,
        values,
        values,
        _worldfoundry_sparse_grid=(2, 4, 4),
    )

    assert output is not None
    assert ops.gate is not None
    torch.testing.assert_close(
        ops.gate[:, :, :32],
        torch.full_like(ops.gate[:, :, :32], 0.375),
    )
    assert torch.count_nonzero(ops.gate[:, :, 32:]) == 0
    report = approximate_attention_report(state)
    assert report["runtime_effective"] is True
    assert report["kernel_attempts"] == 1
    assert report["sparse_calls"] == 1
    assert report["provider_path"] == "fastvideo_kernel.video_sparse_attn"
