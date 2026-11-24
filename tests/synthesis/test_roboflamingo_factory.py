# Copyright 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt.configuration_mosaic_gpt import MosaicGPTConfig
from worldfoundry.base_models.llm_mllm_core.mllm.mosaic_gpt.modeling_mosaic_gpt import MosaicGPT
from worldfoundry.synthesis.action_generation.roboflamingo.modeling import factory

pytestmark = pytest.mark.unit


def _mosaic_checkpoint(path: Path) -> MosaicGPT:
    path.mkdir(parents=True, exist_ok=True)
    config = MosaicGPTConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=64,
        max_seq_len=16,
        attn_impl="torch",
        alibi=True,
        attn_qk_ln=True,
        low_precision_layernorm=True,
        no_bias=True,
        init_device="cpu",
        auto_map={
            "AutoConfig": "configuration_mosaic_gpt.MosaicGPTConfig",
            "AutoModelForCausalLM": "mosaic_gpt.MosaicGPT",
        },
    )
    torch.manual_seed(31)
    model = MosaicGPT(config).eval()
    config.to_json_file(path / "config.json")
    save_file(model.state_dict(), path / "model.safetensors", metadata={"format": "pt"})
    return model


@pytest.mark.parametrize("checkpoint_python", [False, True])
def test_mosaic_gpt_loads_explicitly_without_checkpoint_code(tmp_path: Path, checkpoint_python: bool) -> None:
    expected = _mosaic_checkpoint(tmp_path)
    if checkpoint_python:
        for name in ("mosaic_gpt.py", "configuration_mosaic_gpt.py"):
            (tmp_path / name).write_text(
                'raise AssertionError("checkpoint Python must not execute")\n', encoding="utf-8"
            )
    imported = set(sys.modules)
    loaded = factory._load_language_encoder(str(tmp_path))
    assert type(loaded) is MosaicGPT
    assert not any(name.startswith("transformers_modules.") for name in set(sys.modules) - imported)
    assert set(loaded.state_dict()) == set(expected.state_dict())
    ids = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        output = loaded(ids, attention_mask=torch.ones_like(ids))
        reference = expected(ids, attention_mask=torch.ones_like(ids))
    torch.testing.assert_close(output.logits, reference.logits, rtol=0, atol=0)


def test_native_transformers_language_models_keep_their_loader(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = GPT2LMHeadModel(GPT2Config(n_embd=32, n_head=4, n_layer=1, n_positions=16, vocab_size=64)).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = factory._load_language_encoder(str(tmp_path))
    assert type(loaded) is GPT2LMHeadModel
    assert set(loaded.state_dict()) == set(model.state_dict())
    ids = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)


def test_factory_composes_the_canonical_mosaic_model_into_action_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_clip.transformer import VisionTransformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path = tmp_path / "mpt-1b-redpajama-200b"
    _mosaic_checkpoint(path)
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{f"t{i}": i for i in range(1, 64)}}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    text = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    text.save_pretrained(path)
    vision = torch.nn.Module()
    vision.visual = VisionTransformer(
        image_size=16, patch_size=8, width=32, layers=1, heads=4, mlp_ratio=2, output_dim=32, output_tokens=True
    )
    processor = object()

    def load_vision(name: str, *, pretrained: str) -> tuple[torch.nn.Module, None, Any]:
        assert name == "tiny-clip" and pretrained == "local"
        return vision, None, processor

    monkeypatch.setattr(factory.open_clip, "create_model_and_transforms", load_vision)
    monkeypatch.setattr(factory.open_clip, "get_model_config", lambda name: {"vision_cfg": {"width": 32}})
    model, returned_processor, loaded_tokenizer = factory.create_model_and_transforms(
        "tiny-clip",
        "local",
        str(path),
        str(path),
        llm_name="mpt_3b",
        window_size=1,
    )
    assert isinstance(model.lang_encoder, MosaicGPT)
    assert model.lang_encoder.get_input_embeddings().num_embeddings == len(loaded_tokenizer)
    assert returned_processor is processor
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    tokens = loaded_tokenizer("<image> t1 <|endofchunk|>", return_tensors="pt")
    with torch.inference_mode():
        output = model(
            vision_x=torch.randn(1, 1, 1, 3, 16, 16),
            lang_x=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
    action, gripper = output.logits
    assert action.shape == (1, 1, 6)
    assert gripper.shape == (1, 1, 1)
    assert torch.isfinite(action).all() and torch.isfinite(gripper).all()
