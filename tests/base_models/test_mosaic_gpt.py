# Copyright 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from transformers.modeling_outputs import CausalLMOutputWithPast

from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt import attention
from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt.configuration_mosaic_gpt import MosaicGPTConfig
from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt.modeling_mosaic_gpt import MosaicGPT

pytestmark = pytest.mark.unit


def _config(**overrides: Any) -> MosaicGPTConfig:
    values: dict[str, Any] = {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "max_seq_len": 16,
        "vocab_size": 64,
        "attn_impl": "torch",
        "alibi": True,
        "attn_qk_ln": True,
        "low_precision_layernorm": True,
        "no_bias": True,
        "param_init_fn": "kaiming_normal_",
        "init_device": "cpu",
        "use_cache": False,
    }
    return MosaicGPTConfig(**{**values, **overrides})


def _assert_outputs_equal(actual: CausalLMOutputWithPast, expected: CausalLMOutputWithPast) -> None:
    assert actual.logits is not None and expected.logits is not None
    assert torch.isfinite(actual.logits).all()
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    assert actual.hidden_states is not None and expected.hidden_states is not None
    for value, reference in zip(actual.hidden_states, expected.hidden_states, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert actual.past_key_values is not None and expected.past_key_values is not None
    for pair, expected_pair in zip(actual.past_key_values, expected.past_key_values, strict=True):
        for value, reference in zip(pair, expected_pair, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("low_precision_layernorm", [False, True])
def test_checkpoint_reload_preserves_state_hidden_outputs_and_cache(
    tmp_path: Path, dtype: torch.dtype, alibi: bool, low_precision_layernorm: bool
) -> None:
    torch.manual_seed(17)
    config = _config(alibi=alibi, low_precision_layernorm=low_precision_layernorm)
    model = MosaicGPT(config).eval().to(dtype)
    state = model.state_dict()
    assert "transformer.blocks.0.attn.Wqkv.weight" in state
    assert "transformer.blocks.0.attn.q_ln.weight" in state
    assert not any(name.endswith(".bias") for name in state)
    config.to_json_file(tmp_path / "config.json")
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    restored, loading = MosaicGPT.from_pretrained(tmp_path, local_files_only=True, output_loading_info=True)
    restored = restored.eval().to(dtype)
    assert loading["missing_keys"] == loading["unexpected_keys"] == loading["mismatched_keys"] == []
    assert set(restored.state_dict()) == set(state)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    with torch.inference_mode():
        expected = model(ids, attention_mask=torch.ones_like(ids), use_cache=True, output_hidden_states=True)
        actual = restored(ids, attention_mask=torch.ones_like(ids), use_cache=True, output_hidden_states=True)
        _assert_outputs_equal(actual, expected)
        next_ids = torch.tensor([[8], [9]])
        mask = torch.ones(2, 4, dtype=torch.long)
        expected_next = model(
            next_ids,
            attention_mask=mask,
            past_key_values=expected.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        actual_next = restored(
            next_ids,
            attention_mask=mask,
            past_key_values=actual.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        _assert_outputs_equal(actual_next, expected_next)
        restored(ids[:, :2], attention_mask=torch.ones_like(ids[:, :2]))
        repeated = restored(ids, attention_mask=torch.ones_like(ids), use_cache=True, output_hidden_states=True)
        torch.testing.assert_close(repeated.logits, expected.logits, rtol=0, atol=0)


@pytest.mark.parametrize("implementation", ["torch", "flash", "triton"])
def test_attention_implementation_selection_is_preserved(implementation: str) -> None:
    model = MosaicGPT(_config(attn_impl=implementation, alibi=False)).eval()
    expected = {
        "torch": attention.scaled_multihead_dot_product_attention,
        "flash": attention.flash_attn_fn,
        "triton": attention.triton_flash_attn_fn,
    }
    assert model.transformer.blocks[0].attn.attn_fn is expected[implementation]


def test_generation_and_prefix_cache_contracts() -> None:
    torch.manual_seed(21)
    model = MosaicGPT(_config()).eval()
    ids = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        tokens = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=2, use_cache=False, do_sample=False, pad_token_id=0
        )
    assert tokens.shape == (1, 5)
    torch.testing.assert_close(tokens[:, :3], ids, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="right padding"):
        model.prepare_inputs_for_generation(ids, attention_mask=torch.tensor([[1, 1, 0]]))
    prefix = MosaicGPT(_config(prefix_lm=True)).eval()
    with pytest.raises(NotImplementedError, match="use_cache=False"):
        prefix.prepare_inputs_for_generation(ids, attention_mask=torch.ones_like(ids), use_cache=False)
    prepared = prefix.prepare_inputs_for_generation(ids, attention_mask=torch.ones_like(ids))
    assert prepared["use_cache"] is True
    torch.testing.assert_close(prepared["prefix_mask"], torch.ones_like(ids, dtype=torch.bool), rtol=0, atol=0)


def test_labels_are_not_an_inference_input() -> None:
    model = MosaicGPT(_config()).eval()
    ids = torch.tensor([[2, 3]])
    with pytest.raises(ValueError, match="labels are not supported"):
        model(ids, attention_mask=torch.ones_like(ids), labels=ids)


def test_attention_mask_remains_required() -> None:
    model = MosaicGPT(_config()).eval()
    with pytest.raises(AttributeError, match="bool"):
        model(torch.tensor([[2, 3]]))


def test_generation_beam_callback_reorders_the_legacy_tuple_cache() -> None:
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    values = keys + 1
    reordered = MosaicGPT._reorder_cache([(keys, values)], torch.tensor([1, 0]))
    assert len(reordered) == 1
    torch.testing.assert_close(reordered[0][0], keys.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(reordered[0][1], values.flip(0), rtol=0, atol=0)
