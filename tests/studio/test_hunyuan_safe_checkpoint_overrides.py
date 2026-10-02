from pathlib import Path
from types import SimpleNamespace

import torch
from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
from worldfoundry.base_models.diffusion_model.models.encoders.hunyuan_video.component import (
    HunyuanVideoPromptConditioner,
)
from worldfoundry.base_models.diffusion_model.models.encoders.hunyuan_video.original import (
    _llm_final_layer_norm,
)
from worldfoundry.base_models.diffusion_model.recipes.hunyuan_video import hunyuan_video_t2v_recipe
from worldfoundry.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoT2VPipeline
from worldfoundry.studio.inference.catalog import find_entry


def test_original_hunyuan_accepts_converted_safetensors(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "tencent--HunyuanVideo"
    transformer = root / "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.safetensors"
    vae = root / "hunyuan-video-t2v-720p/vae/pytorch_model.safetensors"
    transformer.parent.mkdir(parents=True)
    vae.parent.mkdir(parents=True)
    transformer.touch()
    vae.touch()
    captured = {}

    def fake_from_pretrained(model_id, **kwargs):
        captured.update(model_id=model_id, **kwargs)
        return object()

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", fake_from_pretrained)
    HunyuanVideoT2VPipeline.from_pretrained(
        root,
        device="cpu",
        checkpoint_overrides={"transformer": transformer, "vae": vae},
        release_text_encoders_after_encode=True,
    )

    assert captured["model_id"] == "hunyuanvideo-t2v"
    assert captured["checkpoint_overrides"]["transformer"] == transformer
    assert captured["checkpoint_overrides"]["vae"] == vae
    assert captured["component_options"]["conditioner:main"]["release_after_encode"] is True
    resolved_options = NativeDiffusionAssembler._component_options(
        hunyuan_video_t2v_recipe(), captured["component_options"]
    )
    assert any(values.get("release_after_encode") is True for values in resolved_options.values())
    assert "checkpoint_overrides" in find_entry("hunyuanvideo-t2v").load_params
    assert "release_text_encoders_after_encode" in find_entry("hunyuanvideo-t2v").load_params


def test_original_hunyuan_can_release_one_shot_text_models() -> None:
    primary = SimpleNamespace(model=torch.nn.Linear(2, 2))
    clip = SimpleNamespace(model=torch.nn.Linear(2, 2))
    conditioner = HunyuanVideoPromptConditioner(primary, clip, release_after_encode=True)

    conditioner._release_models()

    assert primary.model is None
    assert clip.model is None


def test_original_hunyuan_resolves_llava_language_model_norm() -> None:
    norm = torch.nn.LayerNorm(2)
    llava = SimpleNamespace(language_model=SimpleNamespace(norm=norm))

    assert _llm_final_layer_norm(llava) is norm
