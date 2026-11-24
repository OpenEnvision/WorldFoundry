# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Alaya's loader and execution policies compose with native LTX layers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.utils.checkpoint
from safetensors.torch import load_file, save_file

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import PerturbationType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import BasicAVTransformerBlock
from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.alayaworld_v11 import AlayaWorldLTXModel
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.core.model_loading.lora import merge_named_lora_
from worldfoundry.synthesis.visual_generation.alayaworld_v11.alayaworld_v11_runtime.ltx2.modules import (
    attention as attention_policy,
)
from worldfoundry.synthesis.visual_generation.alayaworld_v11.alayaworld_v11_runtime.ltx2.modules.model_ltx_2_3 import (
    LTX23Model,
)

pytestmark = pytest.mark.unit


def _config(**updates: Any) -> dict[str, Any]:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "cross_attention_dim": 16,
        "caption_channels": 12,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "caption_proj_before_connector": True,
        **updates,
    }


def _initialize(model: LTX23Model) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)


def _cpu_attention(model: AlayaWorldLTXModel) -> None:
    for block in model.blocks:
        block.attn1.attention_function = attention_policy.AttentionFunction.PYTORCH
        block.attn2.attention_function = attention_policy.AttentionFunction.PYTORCH
        block.attn2.masked_attention_function = attention_policy.AttentionFunction.PYTORCH


def _inputs(dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    return {
        "x": torch.randn(1, 4, 2, 2, 2, dtype=dtype),
        "t": torch.tensor([750.0]),
        "context": torch.randn(1, 3, 16, dtype=dtype),
        "seq_len": 8,
        "cond_latent_frames": 1,
    }


@pytest.mark.parametrize("with_action", [False, True])
def test_alayaworld_v11_native_loader_preserves_block_and_lora_keys(tmp_path: Path, with_action: bool) -> None:
    config = _config(enable_action_control=with_action)
    reference = LTX23Model(**config).to(dtype=torch.bfloat16).eval()
    _initialize(reference)
    checkpoint = tmp_path / "transformer.pt"
    torch.save(reference.state_dict(), checkpoint)
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=LTX23Model, config=config),
        CheckpointSpec(source=str(checkpoint)),
        RuntimePolicy(device="cpu", dtype=torch.bfloat16),
    )
    assert isinstance(loaded, AlayaWorldLTXModel)
    assert type(loaded.blocks[0]) is BasicAVTransformerBlock
    assert isinstance(loaded.blocks[0].attn1, Attention)
    assert loaded.transformer_blocks is loaded.blocks
    assert loaded._no_split_modules == ["BasicAVTransformerBlock"]
    assert loaded.config.num_layers == 1
    assert not any(key.startswith("transformer_blocks.") for key in loaded.state_dict())
    assert not loaded.gradient_checkpointing
    torch.testing.assert_close(loaded.state_dict(), reference.state_dict(), rtol=0, atol=0)

    a, b = torch.randn(2, 16, dtype=torch.bfloat16), torch.randn(16, 2, dtype=torch.bfloat16)
    lora = tmp_path / "lora.safetensors"
    save_file(
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": a,
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": b,
        },
        lora,
    )
    state = {key.replace("transformer_blocks.", "blocks."): value for key, value in load_file(lora).items()}
    expected = reference.blocks[0].attn1.to_q.weight.detach() + (b.float() @ a.float()).to(torch.bfloat16)
    assert merge_named_lora_(loaded, state, expected_modules=1, device="cpu") == 1
    torch.testing.assert_close(loaded.blocks[0].attn1.to_q.weight, expected, rtol=0, atol=0)
    with torch.no_grad():
        reference.blocks[0].attn1.to_q.weight.copy_(expected)
    _cpu_attention(reference)
    _cpu_attention(loaded)
    inputs = _inputs(torch.bfloat16)
    if with_action:
        inputs["action_vectors"] = torch.randn(1, 3, 6, dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(loaded(**inputs), reference(**inputs), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("training", "grad_enabled", "checkpointing", "expected_calls"),
    [(False, True, True, 1), (True, True, True, 1), (False, False, True, 0), (True, True, False, 0)],
)
def test_alayaworld_v11_checkpoint_policy_includes_eval_with_grad(
    monkeypatch: pytest.MonkeyPatch, training: bool, grad_enabled: bool, checkpointing: bool, expected_calls: int
) -> None:
    model = LTX23Model(**_config()).train(training)
    _initialize(model)
    _cpu_attention(model)
    model.gradient_checkpointing = checkpointing
    original = torch.utils.checkpoint.checkpoint
    calls = []

    def checkpoint(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["use_reentrant"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", checkpoint)
    with torch.set_grad_enabled(grad_enabled):
        result = model(**_inputs())
        if grad_enabled:
            result.square().mean().backward()
    assert calls == [False] * expected_calls


def test_alayaworld_v11_forces_family_flash_policy_and_bf16_kernel_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LTX23Model(**_config(attention_type=attention_policy.AttentionFunction.PYTORCH)).eval()
    _initialize(model)
    for attention in (model.blocks[0].attn1, model.blocks[0].attn2):
        assert attention.attention_function is attention_policy.AttentionFunction.FLASH_ATTENTION_3
    assert model.blocks[0].ada_zero_function.backend == "torch"
    seen = []

    def flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, window_size: tuple[int, int]) -> torch.Tensor:
        seen.append((q.dtype, k.dtype, v.dtype, window_size))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(attention_policy, "flash_attn_func", flash)
    monkeypatch.setattr(attention_policy, "_FLASH_ATTN_3_AVAILABLE", False)
    with torch.no_grad():
        output = model(**_inputs())
    assert output.shape == (1, 4, 2, 2, 2)
    assert seen == [(torch.bfloat16, torch.bfloat16, torch.bfloat16, (-1, -1))] * 2


def test_alayaworld_v11_stg_uses_native_block_count() -> None:
    from worldfoundry.synthesis.visual_generation.alayaworld_v11.rollout import AlayaV11Rollout

    rollout = AlayaV11Rollout.__new__(AlayaV11Rollout)
    rollout.cfg = SimpleNamespace(validation=SimpleNamespace(stg_scale=1.0, stg_blocks=[0]))
    rollout.components = SimpleNamespace(transformer=LTX23Model(**_config()))
    perturbations = rollout._validation_stg_perturbations()
    assert perturbations is not None
    assert perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, 0)
    assert not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)
