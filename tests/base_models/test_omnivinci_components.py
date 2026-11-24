# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Small real-tensor contracts for OmniVinci's released inference components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Qwen2AudioEncoder,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2TokenizerFast,
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig

from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.base_projector import (
    DownSampleBlock,
    MultimodalProjector,
    MultimodalProjectorConfig,
)
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.configuration_vila import VILAConfig
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.constants import MEDIA_TOKENS, MM_BOS_EOS_TOKENS
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.media_encoder import RotaryEmbedding, pool
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.modeling_vila import VILAForCausalLM
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.sound_base_projector import (
    SoundMultimodalProjector,
    SoundMultimodalProjectorConfig,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def omnivinci_checkpoint(tmp_path: Path) -> Path:
    """Write random small weights with the five released component directories."""
    special_tokens = [
        "<unk>",
        "[PAD]",
        "[BOS]",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<vila/sentinel>",
        "\n",
        *MEDIA_TOKENS.values(),
        *(token for pair in MM_BOS_EOS_TOKENS.values() for token in pair),
    ]
    vocabulary = {
        token: index for index, token in enumerate([*special_tokens, "user", "assistant", "describe", "clip"])
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="<|im_end|>",
        model_max_length=4096,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.chat_template = template
    tokenizer.save_pretrained(tmp_path / "llm")
    (tmp_path / "qwen2.jinja").write_text(template)
    config = VILAConfig(
        hidden_size=32,
        mm_hidden_size=24,
        sound_hidden_size=8,
        image_aspect_ratio="dynamic_s2",
        num_video_frames=8,
        fps=0.0,
        mm_vision_select_layer=-2,
        mm_vision_select_feature="cls_patch",
        dynamic_s2=True,
        s2_scales="8,16,24",
        s2_max_split_size=8,
        s2_resize_output_to_scale_idx=-1,
        sound_mm_projector="mlp",
        chat_template="qwen2",
        model_dtype="torch.float32",
        mm_use_bos_eos_tokens=True,
        unified_audio_encoder=True,
        load_audio_in_video=False,
        image_encoder={"_target_": "BasicImageEncoder"},
        video_encoder={
            "_target_": "TSPVideoEncoder",
            "pool_sizes": [[8, 1, 1]],
            "trope_dim": 8,
            "embed_time": "True",
            "time_embed_type": "lang",
            "end_tokens": "None",
        },
        sound_encoder={
            "_target_": "BasicSoundEncoder",
            "trope_dim": 8,
            "embed_time": "True",
            "time_embed_type": "lang",
            "end_tokens": "None",
        },
        # The local loader must ignore remote-code declarations in checkpoint data.
        auto_map={"AutoConfig": "missing_remote.Config", "AutoModel": "missing_remote.Model"},
    )
    config.save_pretrained(tmp_path)
    MultimodalProjector(MultimodalProjectorConfig(), config).save_pretrained(tmp_path / "mm_projector")
    SoundMultimodalProjector(SoundMultimodalProjectorConfig(), config).save_pretrained(tmp_path / "sound_mm_projector")
    SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=2,
            vision_use_head=False,
            _attn_implementation="eager",
        )
    ).save_pretrained(tmp_path / "vision_tower")
    SiglipImageProcessor(size={"height": 8, "width": 8}).save_pretrained(tmp_path / "vision_tower")
    Qwen2AudioEncoder(
        Qwen2AudioEncoderConfig(
            d_model=8,
            num_mel_bins=4,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            max_source_positions=1500,
            _attn_implementation="eager",
        )
    ).save_pretrained(tmp_path / "sound_tower")
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            _attn_implementation="eager",
        )
    ).save_pretrained(tmp_path / "llm")
    return tmp_path


@pytest.fixture
def cpu_omnivinci_loaders(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record production FA2 requests, using eager only for the CPU tensor fixture."""
    calls = []
    for cls in (SiglipVisionModel, Qwen2AudioEncoder):
        load = cls.from_pretrained

        def local_load(*args: Any, _load: Any = load, **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            assert kwargs["attn_implementation"] == "flash_attention_2"
            assert kwargs["local_files_only"] is True
            kwargs["attn_implementation"] = "eager"
            return _load(*args, **kwargs)

        monkeypatch.setattr(cls, "from_pretrained", local_load)
    return calls


@pytest.mark.parametrize("side", [3, 4])
def test_downsample_block_keeps_released_channel_order(side: int) -> None:
    inputs = torch.arange(side * side * 2, dtype=torch.float32).view(1, side * side, 2)
    grid = inputs.view(1, side, side, 2)
    if side % 2:
        grid = torch.nn.functional.pad(grid, (0, 0, 0, 1, 0, 1))
    expected = torch.stack(
        [
            torch.cat([grid[0, row, col], grid[0, row, col + 1], grid[0, row + 1, col], grid[0, row + 1, col + 1]])
            for row in range(0, grid.shape[1], 2)
            for col in range(0, grid.shape[2], 2)
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(DownSampleBlock()(inputs), expected, rtol=0, atol=0)


def test_temporal_pool_uses_scalar_partial_window_mean() -> None:
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    torch.testing.assert_close(pool(inputs, 2, 0), torch.tensor([[2.0, 3.0], [12.5, 17.5]]), rtol=0, atol=0)


def test_rotary_axial_cache_preserves_frequency_values() -> None:
    rotary = RotaryEmbedding(8, theta=50000)
    first = rotary.get_axial_freqs(2, 5)
    second = rotary.get_axial_freqs(2, 5)
    assert first.shape == (2, 5, 16)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert list(rotary.state_dict()) == ["freqs"]


def test_local_loader_keeps_component_keys_dtype_and_generation_defaults(
    omnivinci_checkpoint: Path, cpu_omnivinci_loaders: list[dict[str, Any]]
) -> None:
    from safetensors import safe_open

    model = VILAForCausalLM.from_pretrained(omnivinci_checkpoint, torch_dtype=torch.float32, device_map="cpu")
    components = {
        "mm_projector": model.mm_projector,
        "sound_mm_projector": model.sound_mm_projector,
        "vision_tower": model.vision_tower.vision_tower,
        "sound_tower": model.sound_tower.audio_tower,
        "llm": model.llm,
    }
    for name, component in components.items():
        with safe_open(omnivinci_checkpoint / name / "model.safetensors", framework="pt") as checkpoint:
            assert set(component.state_dict()) == set(checkpoint.keys())
            for key, value in component.state_dict().items():
                torch.testing.assert_close(value, checkpoint.get_tensor(key), rtol=0, atol=0)
    assert model.llm_model_embed_tokens is model.llm.model.embed_tokens
    assert model.dtype == torch.float32
    assert model.training is False
    assert len(cpu_omnivinci_loaders) == 2
    generation = model.default_generation_config
    assert generation.do_sample is False
    assert generation.max_length == model.tokenizer.model_max_length
    assert generation.eos_token_id == model.tokenizer.eos_token_id
    assert model.tokenizer.media_token_ids == {
        name: model.tokenizer.convert_tokens_to_ids(token) for name, token in MEDIA_TOKENS.items()
    }
    first = model.default_generation_config
    first.max_new_tokens = 3
    assert model.default_generation_config.max_new_tokens is None


def test_processor_and_local_model_embed_image_video_in_released_half_precision(
    omnivinci_checkpoint: Path, cpu_omnivinci_loaders: list[dict[str, Any]]
) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from worldfoundry.synthesis.visual_generation.omnivinci.omnivinci_runtime.auto_processor import VILAProcessor

    image_path = omnivinci_checkpoint / "input.png"
    video_path = omnivinci_checkpoint / "input.avi"
    Image.fromarray(np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)).save(image_path)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*"MJPG"), 5.0, (32, 24))
    assert writer.isOpened()
    for value in (0, 60, 120):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    processor = VILAProcessor.from_pretrained(omnivinci_checkpoint)
    model = VILAForCausalLM.from_pretrained(omnivinci_checkpoint, torch_dtype=torch.float16, device_map="cpu")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": "describe clip"},
            ],
        }
    ]
    inputs = processor([processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)])
    assert len(inputs.media["video"][0]) == 8
    assert inputs.media["image"][0].dtype == torch.float16
    assert inputs.media["video"][0].dtype == torch.float16
    assert inputs.media["video_info"][0][0]["has_audio"] is False
    with torch.inference_mode():
        embedded, labels, mask = model._embed(inputs.input_ids, inputs.media, inputs.media_config, None, None)
        assert embedded.dtype == torch.float16
        assert mask.all()
        assert labels.eq(-100).all()
        assert embedded.shape[1] > inputs.input_ids.shape[1]
        output = model.llm(inputs_embeds=embedded, attention_mask=mask, use_cache=False).logits
        assert torch.isfinite(output).all()
    assert len(cpu_omnivinci_loaders) == 2
