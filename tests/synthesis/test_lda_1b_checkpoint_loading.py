# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""LDA-1B loads released DINOv3 LayerScale names into the canonical parameter graph."""

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import init_empty_weights
from transformers import AutoImageProcessor, AutoProcessor, Qwen3VLConfig

from worldfoundry.core.nn.blocks.layers import LayerScale
from worldfoundry.synthesis.action_generation.lda_1b.dinov3_configuration import DINOv3ViTConfig
from worldfoundry.synthesis.action_generation.lda_1b.dinov3_modeling import DINOv3ViTModel
from worldfoundry.synthesis.action_generation.lda_1b.modeling import LDAInferenceModel
from worldfoundry.synthesis.action_generation.lda_1b.runtime import LDA1BRuntime, LDA1BRuntimeConfig

pytestmark = pytest.mark.unit


def _small_vision_config(gated_mlp: bool = False) -> DINOv3ViTConfig:
    config = DINOv3ViTConfig(
        image_size=8,
        patch_size=4,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_register_tokens=1,
        layerscale_value=0.25,
        use_gated_mlp=gated_mlp,
    )
    config._attn_implementation = "eager"
    return config


def _released_names(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        re.sub(r"^(action_model\.vision_encoder\.layer\.\d+\.layer_scale[12])\.gamma$", r"\1.lambda1", key): tensor
        for key, tensor in state.items()
    }


def _small_lda_configs() -> tuple[Qwen3VLConfig, dict[str, Any], DINOv3ViTConfig]:
    qwen = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    )
    qwen._attn_implementation = "eager"
    qwen.text_config._attn_implementation = "eager"
    qwen.vision_config._attn_implementation = "eager"
    action = {
        "action_model_type": "DiT-B",
        "hidden_size": 16,
        "action_dim": 2,
        "state_dim": 3,
        "future_action_window_size": 1,
        "num_inference_timesteps": 2,
        "vision_encoder_type": "dinov3",
        "num_views": 1,
        "obs_horizon": 1,
        "max_num_embodiments": 1,
        "num_target_vision_tokens": 1,
        "add_pos_embed": False,
        "num_timestep_buckets": 100,
        "diffusion_model_cfg": {"cross_attention_dim": 16, "output_dim": 16, "num_layers": 1, "dropout": 0.0},
    }
    return qwen, action, DINOv3ViTConfig(**_small_vision_config().to_dict())


def test_dinov3_initializes_canonical_layer_scale() -> None:
    config = _small_vision_config()
    model = DINOv3ViTModel(config)
    for layer in model.layer:
        for scale in (layer.layer_scale1, layer.layer_scale2):
            assert type(scale) is LayerScale
            assert not scale.inplace
            torch.testing.assert_close(scale.gamma, torch.full((16,), 0.25), atol=0, rtol=0)
    config.layerscale_value = 0.125
    model._init_weights(model.layer[0].layer_scale1)
    torch.testing.assert_close(model.layer[0].layer_scale1.gamma, torch.full((16,), 0.125), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gated_mlp", [False, True])
@pytest.mark.parametrize("payload_style", ["plain", "wrapped", "ddp"])
def test_lda_loads_released_dinov3_subtree(
    tmp_path: Path, dtype: torch.dtype, gated_mlp: bool, payload_style: str
) -> None:
    torch.manual_seed(17)
    config = _small_vision_config(gated_mlp)
    source = DINOv3ViTModel(config).to(dtype).eval()
    prefix = "action_model.vision_encoder."
    state = {prefix + key: value for key, value in source.state_dict().items()}
    released = _released_names(state)
    assert len(released) == len(state)
    assert sum(key.endswith(".lambda1") for key in released) == 4
    if payload_style == "ddp":
        released = {"module." + key: value for key, value in released.items()}
    payload = released if payload_style == "plain" else {"state_dict": released}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    loaded_state = LDA1BRuntime._load_state_dict(checkpoint)
    assert loaded_state.keys() == state.keys()
    for key, tensor in loaded_state.items():
        torch.testing.assert_close(tensor, state[key], atol=0, rtol=0)
    with init_empty_weights(include_buffers=False):
        loaded = DINOv3ViTModel(config)
    loaded.load_state_dict(
        {key.removeprefix(prefix): value for key, value in loaded_state.items()}, strict=True, assign=True
    )
    loaded.to(dtype).eval()
    pixels = torch.randn(2, 3, 8, 8).to(dtype)
    with torch.no_grad():
        expected = source(pixels).last_hidden_state
        actual = loaded(pixels).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_lda_layer_scale_remap_is_scoped_and_rejects_collisions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    canonical_key = "action_model.vision_encoder.layer.12.layer_scale2.gamma"
    released_key = canonical_key.removesuffix("gamma") + "lambda1"
    unrelated = "action_model.model.layer.12.layer_scale2.lambda1"
    torch.save({canonical_key: torch.ones(16), unrelated: torch.zeros(2)}, checkpoint)
    state = LDA1BRuntime._load_state_dict(checkpoint)
    assert set(state) == {canonical_key, unrelated}
    torch.save({canonical_key: torch.ones(16), released_key: torch.zeros(16)}, checkpoint)
    with pytest.raises(ValueError, match="remap collision"):
        LDA1BRuntime._load_state_dict(checkpoint)


@pytest.mark.parametrize("released_names", [False, True])
def test_lda_runtime_loads_complete_policy_with_canonical_layer_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released_names: bool
) -> None:
    torch.manual_seed(29)
    qwen_config, action_config, vision_config = _small_lda_configs()
    source = LDAInferenceModel(qwen_config=qwen_config, action_config=action_config, vision_config=vision_config).eval()
    qwen_root, vision_root = tmp_path / "qwen", tmp_path / "vision"
    qwen_config.save_pretrained(qwen_root)
    vision_config.save_pretrained(vision_root)
    (qwen_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (vision_root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    state = source.state_dict()
    torch.save(_released_names(state) if released_names else state, tmp_path / "model.pt")

    def qwen_processor(path: Path, **kwargs: Any) -> SimpleNamespace:
        assert path == qwen_root
        return SimpleNamespace(tokenizer=SimpleNamespace())

    def vision_processor(path: Path, **kwargs: Any) -> SimpleNamespace:
        assert path == vision_root
        return SimpleNamespace()

    monkeypatch.setattr(AutoProcessor, "from_pretrained", qwen_processor)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", vision_processor)
    runtime = LDA1BRuntime(
        LDA1BRuntimeConfig(
            checkpoint_location=str(tmp_path),
            checkpoint_filename="model.pt",
            qwen_location=str(qwen_root),
            vision_location=str(vision_root),
            action_config=action_config,
            variant={},
            prompt_template="{prompt}",
            device="cpu",
            torch_dtype="float32",
            attention_backend="eager",
        )
    )
    loaded = runtime._load_model()
    assert runtime._load_model() is loaded
    assert type(loaded.action_model.vision_encoder.layer[0].layer_scale1) is LayerScale
    assert state.keys() == loaded.state_dict().keys()
    for key, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, state[key], atol=0, rtol=0)
    inputs = {
        "vl_embs": torch.randn(1, 3, 16),
        "state": torch.randn(1, 1, 3),
        "curr_pixels": torch.randn(1, 1, 1, 3, 8, 8),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    expected = source.action_model.predict_action(**inputs, generator=torch.Generator().manual_seed(41))
    actual = loaded.action_model.predict_action(**inputs, generator=torch.Generator().manual_seed(41))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
