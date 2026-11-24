"""Weight-free Pika H3 Flex adapter and schedule contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pika import (
    load_pika_h3_flex_adapter,
)
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_sigma_schedule,
)
from worldfoundry.pipelines.pika_h3_flex import pipeline_pika_h3_flex as pika_module
from worldfoundry.pipelines.pika_h3_flex.pipeline_pika_h3_flex import (
    PIKA_H3_FLEX_KNOTS,
    PikaH3FlexPipeline,
)

pytestmark = pytest.mark.unit


def _tiny_model() -> MiniMaxH3DiTModel:
    return MiniMaxH3DiTModel(
        MiniMaxH3DiTArchConfig(
            num_layers=1,
            token_refiner_num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            attention_head_dim=8,
            ffn_hidden_size=4,
            text_dim=8,
            time_embed_hidden_size=8,
            time_embed_dim=4,
            adaln_out_features=18 * 8,
            final_adaln_out_features=2 * 8,
            rope_inv_freq_len=2,
        )
    )


def test_pika_lora_converts_diffusers_targets_to_fused_h3(tmp_path: Path) -> None:
    model = _tiny_model()
    before_qkv = model.blocks[0].attn.qkv_proj.weight.detach().clone()
    adapter = tmp_path / "adapter.safetensors"
    save_file(
        {
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(2, 8),
            "base_model.model.transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(8, 2),
            "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_A.default.weight": torch.ones(2, 8),
            "base_model.model.transformer_blocks.0.ff.net.0.proj.lora_B.default.weight": torch.ones(8, 2),
        },
        str(adapter),
    )

    report = load_pika_h3_flex_adapter(model, adapter, rank=2, alpha=2)

    assert len(report["merged_targets"]) == 2
    q_delta = (model.blocks[0].attn.qkv_proj.weight[:8] - before_qkv[:8]).float()
    assert torch.allclose(q_delta, torch.full_like(q_delta, 2.0), atol=0.02, rtol=0.0)


def test_pika_schedule_preserves_six_evaluation_knots() -> None:
    video = minimax_h3_sigma_schedule(PIKA_H3_FLEX_KNOTS, shift_scale=12.0)
    audio = minimax_h3_sigma_schedule(PIKA_H3_FLEX_KNOTS, shift_scale=3.0)

    assert len(video) == len(audio) == 7
    assert video[-1] == audio[-1] == 0.0
    assert all(left > right for left, right in zip(video, video[1:]))
    assert all(left > right for left, right in zip(audio, audio[1:]))


def test_pika_pipeline_defaults_to_fl2va_adapter() -> None:
    pipeline = PikaH3FlexPipeline(transformer=object(), device="cpu")

    assert pipeline.MODEL_ID == "pika-h3-flex"
    assert pipeline.adapter_task == "fl2va"
    assert pipeline.DEFAULT_NUM_INFERENCE_STEPS == 6


def test_ref2va_loading_selects_the_canonical_h3_partition(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_load(cls, *, model_path, required_components, device, model_id, component_devices=None):
        del cls, model_path, device, model_id, component_devices
        captured.update(required_components)
        return SimpleNamespace(
            transformer=object(),
            video_vae=None,
            audio_vae=None,
            text_encoder=None,
            tokenizer=None,
            device="cpu",
            component_devices={"transformer": "cpu"},
        )

    monkeypatch.setattr(
        pika_module.NativeMiniMaxH3Pipeline,
        "from_pretrained",
        classmethod(fake_load),
    )
    monkeypatch.setattr(pika_module, "load_pika_h3_flex_adapter", lambda *args, **kwargs: {})
    monkeypatch.setattr(PikaH3FlexPipeline, "_load_processor", staticmethod(lambda _root: object()))

    PikaH3FlexPipeline.from_pretrained(
        base_model_path=tmp_path,
        adapter_path=tmp_path / "adapter.safetensors",
        adapter_task="ref2va",
    )

    assert captured["partition"] == "Ref2VA"
    assert captured["transformer_subfolder"] == "transformer"
