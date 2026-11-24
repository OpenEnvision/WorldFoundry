# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the canonical H3 HyperFlow variant and released keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from worldfoundry.base_models.diffusion_model.models.denoisers.hyperflow import (
    hyperflow_adapter_plan,
    hyperflow_arch_config,
    hyperflow_foundation_key,
    load_hyperflow_adapter,
    load_hyperflow_transformer,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
    MiniMaxH3DiTBlock,
    MiniMaxH3TimeEmbedder,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.attention import (
    HyperFlowAttention,
    HyperFlowAttentionState,
    resolve_attention_backend,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.conditioning import (
    HyperFlowLinear,
    HyperFlowQKVProjection,
    HyperFlowSwiGLUProjection,
    HyperFlowTwoTimeEmbedder,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.model import (
    HyperFlowDiTModel,
)
from worldfoundry.base_models.diffusion_model.schedulers.hyperflow import (
    DEFAULT_SIGMAS_8STEP,
    HyperFlowSchedule,
    build_row_time_pairs,
)

pytestmark = pytest.mark.unit


def _config() -> dict[str, object]:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 32,
        "num_layers": 3,
        "num_refiner_layers": 1,
        "ffn_dim": 64,
        "in_channels": 4,
        "audio_in_channels": 8,
        "patch_size": [1, 2, 2],
        "text_dim": 12,
        "freq_dim": 8,
        "time_embed_hidden_dim": 32,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
    }


def _write_adapter(path: Path) -> None:
    targets = {
        "attn.to_q": (32, 32),
        "attn.to_k": (32, 32),
        "attn.to_v": (32, 32),
        "attn.to_out.0": (32, 32),
        "ff.net.0.proj": (32, 128),
        "ff.net.2": (64, 32),
    }
    modules = {
        f"{prefix}.{name}": shape
        for prefix in [*(f"transformer_blocks.{index}" for index in range(3)), "token_refiner.refiner_blocks.0"]
        for name, shape in targets.items()
    }
    for prefix in ("time_embedder", "endpoint_time_embedder"):
        modules[f"{prefix}.linear_1"] = (8, 32)
        modules[f"{prefix}.linear_2"] = (32, 16)
    generator = torch.Generator().manual_seed(21)
    tensors = {}
    for name, (in_features, out_features) in modules.items():
        for side, shape in (("A", (4, in_features)), ("B", (out_features, 4))):
            tensors[f"transformer.{name}.lora_{side}.weight"] = torch.randn(shape, generator=generator) * 0.02
    save_file(
        tensors,
        path,
        metadata={
            "hyperflow": "true",
            "hyperflow_gate": "0.25",
            "lora_alpha": "4",
            "lora_rank": "4",
            "hyperflow_sigmas": json.dumps(DEFAULT_SIGMAS_8STEP),
        },
    )


def _model(tmp_path: Path) -> HyperFlowDiTModel:
    torch.manual_seed(20)
    model = HyperFlowDiTModel(hyperflow_arch_config(_config())).float().eval()
    path = tmp_path / "adapter.safetensors"
    _write_adapter(path)
    load_hyperflow_adapter(model, path)
    return model


def _inputs() -> dict[str, object]:
    generator = torch.Generator().manual_seed(22)
    return {
        "hidden_states": torch.randn(1, 5, 16, generator=generator),
        "audio_hidden_states": torch.randn(1, 4, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, 3, 12, generator=generator),
        "timestep": torch.tensor([0.2]),
        "endpoint": torch.tensor([0.4]),
        "timestep_indices": torch.zeros(12, dtype=torch.long),
        "token_tags": torch.tensor([1] * 3 + [2] * 4 + [0] * 5),
        "position_ids": torch.randn(12, 3, generator=generator),
        "video_indices": torch.arange(7, 12),
        "audio_indices": torch.arange(3, 7),
        "text_indices": torch.arange(3),
        "step": 1,
    }


def test_two_time_blend_and_failure_clear_endpoint() -> None:
    base = MiniMaxH3TimeEmbedder(hyperflow_arch_config(_config()))
    embedder = HyperFlowTwoTimeEmbedder(base, gate=0.25)
    timestep, endpoint = torch.tensor([0.2, 0.8]), torch.tensor([0.3, 0.9])
    expected = base(timestep) + 0.25 * (embedder.endpoint(endpoint) - base(timestep))
    with embedder.endpoint_context(endpoint):
        assert torch.equal(embedder(timestep), expected)
    with pytest.raises(RuntimeError, match="endpoint"):
        embedder(timestep)
    with pytest.raises(ValueError, match="Every"), embedder.endpoint_context(endpoint[:1]):
        embedder(timestep)
    assert embedder._endpoint_timesteps is None


def test_lora_keeps_unmerged_peft_arithmetic() -> None:
    torch.manual_seed(23)
    base = nn.Linear(8, 16)
    before = base.weight.detach().clone()
    adapter = HyperFlowLinear(base, rank=4, alpha=8.0)
    value = torch.randn(5, 8)
    expected = base(value) + adapter.lora_B(adapter.lora_A(value)) * 2.0
    assert torch.equal(adapter(value), expected)
    assert torch.equal(base.weight, before)


@pytest.mark.parametrize("head_dim", [None, 4])
def test_qkv_projection_retains_separate_matmuls_and_bias(head_dim: int | None) -> None:
    torch.manual_seed(24)
    projection = HyperFlowQKVProjection(nn.Linear(8, 24), head_dim=head_dim)
    value = torch.randn(1, 3, 8)
    separate = [projection.q(value), projection.k(value), projection.v(value)]
    expected = (
        torch.cat(separate, dim=-1)
        if head_dim is None
        else torch.cat([x.view(1, 3, 2, 4) for x in separate], dim=-1).flatten(-2)
    )
    assert torch.equal(projection(value), expected)


def test_swiglu_keeps_released_parameter_rows_and_swaps_activations() -> None:
    base = nn.Linear(8, 16)
    projection = HyperFlowSwiGLUProjection(base)
    value = torch.randn(3, 8)
    up, gate = base(value).chunk(2, dim=-1)
    assert projection.proj is base
    assert torch.equal(projection(value), torch.cat((gate, up), dim=-1))


def test_canonical_blocks_are_reused_and_requests_are_isolated(tmp_path: Path) -> None:
    model = _model(tmp_path)
    assert all(type(block) is MiniMaxH3DiTBlock for block in model.blocks)
    inputs = _inputs()
    with torch.no_grad():
        first = model(**inputs)
        changed = model(**(inputs | {"endpoint": torch.tensor([0.9])}))
        with pytest.raises(ValueError, match="Every"):
            model(**(inputs | {"endpoint": torch.tensor([0.4, 0.7])}))
        repeated = model(**inputs)
    assert all(torch.equal(left, right) for left, right in zip(first, repeated, strict=True))
    assert any(not torch.equal(left, right) for left, right in zip(first, changed, strict=True))
    assert model.attention_state.step is None
    assert model.time_embedder._endpoint_timesteps is None


@pytest.mark.parametrize(
    "key,expected",
    [
        ("proj_in.weight", "video_patch_proj.weight"),
        ("norm_out.linear.bias", "final_layer.adaln_proj.linear.bias"),
        ("transformer_blocks.49.attn.to_q.weight", "blocks.49.attn.qkv_proj.q.weight"),
        ("token_refiner.refiner_blocks.1.ff.net.0.proj.weight", "token_refiner.blocks.1.mlp.fc1.proj.weight"),
    ],
)
def test_released_foundation_names(key: str, expected: str) -> None:
    assert hyperflow_foundation_key(key) == expected


def test_adapter_keys_are_complete_and_endpoint_is_distinct() -> None:
    keys = [
        f"transformer.{name}.linear_1.lora_{side}.weight"
        for name in ("time_embedder", "endpoint_time_embedder")
        for side in ("A", "B")
    ]
    plan = hyperflow_adapter_plan(keys)
    assert set(plan) == {"time_embedder.base.proj_in", "time_embedder.endpoint.proj_in"}
    with pytest.raises(ValueError, match="complete"):
        hyperflow_adapter_plan(keys[:-1])
    with pytest.raises(ValueError, match="Unexpected"):
        hyperflow_adapter_plan(["transformer.unknown.weight"])


def test_loader_rejects_incomplete_base_and_adapter_shape(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    save_file({"proj_in.weight": torch.zeros(32, 16)}, tmp_path / "model.safetensors")
    adapter = tmp_path / "adapter.safetensors"
    _write_adapter(adapter)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_hyperflow_transformer(tmp_path, adapter)
    model = HyperFlowDiTModel(hyperflow_arch_config(_config())).float()
    save_file(
        {
            "transformer.time_embedder.linear_1.lora_A.weight": torch.zeros(4, 9),
            "transformer.time_embedder.linear_1.lora_B.weight": torch.zeros(32, 4),
        },
        adapter,
        metadata={"hyperflow": "true", "hyperflow_gate": "0.25", "lora_alpha": "4", "lora_rank": "4"},
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_hyperflow_adapter(model, adapter)


def test_row_pairs_pin_conditioning_but_distinguish_equal_start_times() -> None:
    timestep, endpoint, indices = build_row_time_pairs(
        video_indices=torch.tensor([3, 4]),
        audio_indices=torch.tensor([1, 2]),
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
        num_text_tokens=1,
        video_timestep=0.25,
        video_endpoint=0.5,
        audio_timestep=0.25,
        audio_endpoint=0.75,
        condition_video_timestep=0.999,
    )
    assert torch.equal(timestep[indices], torch.tensor([0.25, 1.0, 0.25, 0.999, 0.25]))
    assert torch.equal(endpoint[indices], torch.tensor([0.5, 1.0, 0.75, 0.999, 0.5]))


def test_fixed_schedule_and_euler_round_trip() -> None:
    schedule = HyperFlowSchedule()
    base = torch.tensor(DEFAULT_SIGMAS_8STEP)
    assert torch.equal(schedule.video_sigmas, 12 * base / (1 + 11 * base))
    assert torch.equal(schedule.audio_sigmas, 3 * base / (1 + 2 * base))
    sample, velocity = torch.randn(3, 8), torch.randn(3, 8)
    for audio, sigmas in ((False, schedule.video_sigmas), (True, schedule.audio_sigmas)):
        for index in range(8):
            time = 1 - sigmas[index]
            denoised = sample + (1 - time) * velocity
            ratio = sigmas[index + 1] / sigmas[index]
            assert torch.equal(
                schedule.step(velocity, sample, index, audio=audio), ratio * sample + (1 - ratio) * denoised
            )
    with pytest.raises(ValueError, match="nine sigma"):
        HyperFlowSchedule([1.0, 0.0])


def test_sol_attention_schedule_leaves_refiner_and_first_two_blocks_dense() -> None:
    state = HyperFlowAttentionState()
    policies = [HyperFlowAttention(state, layer=layer, sol_attn=True) for layer in (None, 0, 1, 2, 49)]
    for step in (None, 0, 1, 2, 7):
        state.step = step
        expected = [False, False, False, step is not None and step >= 2, step is not None and step >= 2]
        assert [policy.use_sparse() for policy in policies] == expected
    assert resolve_attention_backend("sdpa", gpus=4) == "native"
    with pytest.raises(ValueError, match="multi-GPU"):
        resolve_attention_backend("_flash_3", gpus=2)


@pytest.mark.parametrize("backend", ["native", "_native_math"])
def test_native_attention_does_not_import_diffusers(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    import builtins

    original = builtins.__import__

    def checked(name: str, *args: Any, **kwargs: Any) -> object:
        if name.startswith("diffusers"):
            raise AssertionError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    assert resolve_attention_backend(backend, gpus=1) == backend
    policy = HyperFlowAttention(HyperFlowAttentionState(step=0), layer=2, sol_attn=True, backend=backend)
    q, k, v = (torch.randn(7, 2, 16) for _ in range(3))
    from worldfoundry.core.attention.backends.native import scaled_dot_product_attention

    expected = scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        backend="math" if backend == "_native_math" else None,
    )[0].transpose(0, 1)
    assert torch.equal(policy(q, k, v, cu_seqlens=torch.tensor([0, 7]), softmax_scale=0.25), expected)
