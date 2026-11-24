# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Go1's native load site accepts released InternLM2 MLP names and canonical saves."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP
from worldfoundry.synthesis.action_generation.go1.configuration_go1 import GO1ModelConfig
from worldfoundry.synthesis.action_generation.go1.modeling_go1 import GO1Model
from worldfoundry.synthesis.action_generation.go1.runtime import GO1Runtime, GO1RuntimeConfig
from worldfoundry.synthesis.action_generation.go1.tokenization_internlm2 import InternLM2Tokenizer

pytestmark = pytest.mark.unit


def _small_go1_config(latent_planning: bool) -> GO1ModelConfig:
    return GO1ModelConfig(
        vision_config={
            "image_size": 8,
            "patch_size": 2,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "use_flash_attn": False,
        },
        llm_config={
            "architectures": ["InternLM2ForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "pad_token_id": 0,
            "attn_implementation": "eager",
        },
        action_config={
            "architectures": ["ActionExpertModel"],
            "input_hidden_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "state_dim": 3,
            "action_dim": 2,
            "action_chunk_size": 3,
            "attn_implementation": "eager",
        },
        latent_planner_config={
            "architectures": ["ActionExpertModel"],
            "input_hidden_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "vocab_size": 4,
            "action_dim": 1,
            "state_token_num": 0,
            "attn_implementation": "eager",
        },
        noise_scheduler_config={
            "num_train_timesteps": 10,
            "num_inference_timesteps": 2,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
            "clip_sample": False,
        },
        latent_planning=latent_planning,
        template="internlm2-chat",
        ps_version="v2",
        img_context_token_id=31,
        pad_token_id=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("latent_planning", [False, True])
@pytest.mark.parametrize("released_names", [False, True])
def test_go1_runtime_streams_checkpoint_into_shared_mlp_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, latent_planning: bool, released_names: bool
) -> None:
    torch.manual_seed(12)
    config = _small_go1_config(latent_planning)
    source = GO1Model(config, torch_dtype=dtype).to(dtype).eval()
    config.save_pretrained(tmp_path)
    state = source.state_dict()
    released = {}
    for key, tensor in state.items():
        destination = key
        if released_names:
            for canonical, original in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
                destination = destination.replace(f".feed_forward.{canonical}.", f".feed_forward.{original}.")
        released[destination] = tensor.contiguous()
    assert len(released) == len(state)
    if released_names:
        assert sum(".feed_forward.w1." in key for key in released) == (6 if latent_planning else 4)
    items = list(released.items())
    split = len(items) // 2
    shards = {
        "model-00001-of-00002.safetensors": dict(items[:split]),
        "model-00002-of-00002.safetensors": dict(items[split:]),
    }
    for filename, tensors in shards.items():
        save_file(tensors, tmp_path / filename, metadata={"format": "pt"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for filename, tensors in shards.items() for key in tensors}}),
        encoding="utf-8",
    )

    def load_tokenizer(path: Path, **kwargs: Any) -> SimpleNamespace:
        assert path == tmp_path
        assert kwargs == {"model_max_length": 32, "local_files_only": True, "trust_remote_code": False}
        return SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(InternLM2Tokenizer, "from_pretrained", load_tokenizer)
    runtime = GO1Runtime(
        GO1RuntimeConfig(
            checkpoint_location=str(tmp_path),
            camera_keys=("front",),
            camera_aliases={},
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
            prompt_template={},
            variant="go1",
            control_frequency=30,
            max_sequence_length=32,
            normalization_epsilon=1e-6,
            device="cpu",
            torch_dtype=str(dtype).removeprefix("torch."),
            attention_backend="eager",
        )
    )
    loaded = runtime._load_model()
    assert runtime._load_model() is loaded
    assert type(loaded.language_model.model.layers[0].feed_forward) is QwenMLP
    assert type(loaded.action_model.layers[0].feed_forward) is QwenMLP
    assert loaded.time_embedder.dtype == loaded.freq_embedder.dtype == dtype
    assert state.keys() == loaded.state_dict().keys()
    for key, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, state[key], atol=0, rtol=0)
    for key, tensor in loaded.named_buffers():
        torch.testing.assert_close(tensor, dict(source.named_buffers())[key], atol=0, rtol=0)
    inputs = {
        "pixel_values": torch.randn(1, 3, 8, 8).to(dtype),
        "input_ids": torch.tensor([[1, 31, 31, 31, 31, 2, 3]]),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "image_flags": torch.ones(1, 1, dtype=torch.long),
        "state": torch.randn(1, 1, 3).to(dtype),
        "ctrl_freqs": torch.tensor([[30.0]]).to(dtype),
    }
    with torch.no_grad():
        torch.manual_seed(42)
        expected = source(**inputs, return_dict=True).action_logits
        torch.manual_seed(42)
        actual = loaded(**inputs, return_dict=True).action_logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
