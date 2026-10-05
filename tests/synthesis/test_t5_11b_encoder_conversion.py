from __future__ import annotations

import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration

from scripts.inference import convert_t5_11b_encoder_to_safetensors as converter


def _tiny_config() -> T5Config:
    return T5Config(
        vocab_size=16,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        tie_word_embeddings=True,
    )


def test_conversion_extracts_strict_encoder_and_links_assets(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "encoder"
    source.mkdir()
    config = _tiny_config()
    config.save_pretrained(source)
    (source / "spiece.model").write_bytes(b"tokenizer fixture")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")

    checkpoint = T5ForConditionalGeneration(config).state_dict()
    checkpoint.pop("encoder.embed_tokens.weight", None)
    torch.save(checkpoint, source / "pytorch_model.bin")
    monkeypatch.setattr(converter, "_torch_version", lambda: (2, 6))

    converted = converter.convert(source, output)

    expected = T5EncoderModel(config).state_dict()
    actual = load_file(converted)
    assert set(actual) == set(expected)
    assert torch.equal(actual["shared.weight"], actual["encoder.embed_tokens.weight"])
    with safe_open(converted, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {
            "architecture": "T5EncoderModel",
            "format": "pt",
            "source": "pytorch_model.bin",
        }
    for name in ("config.json", "spiece.model", "tokenizer.json"):
        link = output / name
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link))
        assert link.samefile(source / name)
    assert not (output / ".model.safetensors.incomplete").exists()


def test_conversion_requires_patched_torch_for_pickle_loading(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "encoder"
    source.mkdir()
    _tiny_config().save_pretrained(source)
    torch.save({"shared.weight": torch.zeros(1)}, source / "pytorch_model.bin")
    monkeypatch.setattr(converter, "_torch_version", lambda: (2, 5))

    try:
        converter.convert(source, output, link_assets=False)
    except RuntimeError as exc:
        assert "Torch >=2.6" in str(exc)
        assert "CVE-2025-32434" in str(exc)
    else:
        raise AssertionError("unsafe Torch version was accepted")


def test_conversion_refuses_to_overwrite_existing_output(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "encoder"
    source.mkdir()
    output.mkdir()
    (source / "pytorch_model.bin").write_bytes(b"fixture")
    (output / "model.safetensors").write_bytes(b"existing")

    try:
        converter.convert(source, output, link_assets=False)
    except FileExistsError as exc:
        assert "refusing to replace" in str(exc)
    else:
        raise AssertionError("existing output was overwritten")
