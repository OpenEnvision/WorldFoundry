from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.wonderjourney.wonderjourney_runtime.util.checkpoints import (
    diffusers_fp16_load_kwargs,
    local_model_ref,
    oneformer_local_load_kwargs,
    require_existing_checkpoint,
)
from worldfoundry.synthesis.visual_generation.wonderjourney.wonderjourney_runtime.util import chatGPT4


def test_checkpoint_lookup_falls_back_to_sibling_ckpt_root(tmp_path: Path, monkeypatch) -> None:
    primary = tmp_path / "ckpts"
    fallback = tmp_path / "ckpt"
    primary.mkdir()
    fallback.mkdir()
    model = fallback / "WonderJourney" / "depth.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(primary))

    assert local_model_ref("remote/model", "WonderJourney/depth.pt") == str(model)
    assert require_existing_checkpoint("WonderJourney/depth.pt") == str(model)


def test_missing_checkpoint_reports_both_standard_roots(tmp_path: Path, monkeypatch) -> None:
    primary = tmp_path / "ckpts"
    primary.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(primary))

    with pytest.raises(FileNotFoundError) as exc_info:
        require_existing_checkpoint("WonderJourney/missing.pth")

    message = str(exc_info.value)
    assert str(primary / "WonderJourney" / "missing.pth") in message
    assert str(tmp_path / "ckpt" / "WonderJourney" / "missing.pth") in message


def test_local_diffusers_fp16_variant_requires_all_components(tmp_path: Path) -> None:
    expected = {
        "text_encoder": "model.fp16.safetensors",
        "unet": "diffusion_pytorch_model.fp16.safetensors",
        "vae": "diffusion_pytorch_model.fp16.safetensors",
    }
    for subfolder, filename in expected.items():
        component = tmp_path / subfolder
        component.mkdir()
        (component / filename).write_bytes(b"safetensors")

    assert diffusers_fp16_load_kwargs(str(tmp_path)) == {
        "variant": "fp16",
        "use_safetensors": True,
    }
    assert diffusers_fp16_load_kwargs(str(tmp_path), subfolder="vae") == {
        "variant": "fp16",
        "use_safetensors": True,
    }
    assert diffusers_fp16_load_kwargs("stabilityai/stable-diffusion-2-inpainting") == {
        "revision": "fp16"
    }


def test_keyword_generation_falls_back_when_spacy_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_spacy(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("spacy is intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_spacy)
    monkeypatch.setattr(chatGPT4, "_nlp", None)
    monkeypatch.setattr(chatGPT4, "_nlp_load_attempted", False)
    generator = chatGPT4.TextpromptGen(tmp_path)

    assert generator.generate_keywords("a quiet village beyond the trees") == "a quiet village beyond the trees"


def test_local_oneformer_processor_uses_bundled_metadata(tmp_path: Path) -> None:
    (tmp_path / "coco_panoptic.json").write_text("{}", encoding="utf-8")

    assert oneformer_local_load_kwargs(str(tmp_path)) == {"local_files_only": True}
    assert oneformer_local_load_kwargs(str(tmp_path), processor=True) == {
        "local_files_only": True,
        "repo_path": str(tmp_path),
    }
    assert oneformer_local_load_kwargs("shi-labs/oneformer_coco_swin_large", processor=True) == {}
