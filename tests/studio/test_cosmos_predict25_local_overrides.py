from __future__ import annotations

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.pipelines.cosmos.pipeline_cosmos_predict2p5 import CosmosPredict2p5Pipeline
from worldfoundry.studio.inference.catalog import find_entry


def test_cosmos_predict25_catalog_uses_cosmos_local_components() -> None:
    entry = find_entry("cosmos-predict2.5")

    assert entry.default_model_ref
    components = entry.default_load_kwargs["required_components"]
    assert components["vae_model_path"] == entry.default_model_ref
    assert "Cosmos-Reason1-7B" in components["text_encoder_model_path"]


def test_cosmos_transfer25_catalog_uses_local_transfer_and_predict_components() -> None:
    entry = find_entry("cosmos-transfer-2.5")

    assert "Cosmos-Transfer2.5-2B" in entry.default_model_ref
    components = entry.default_load_kwargs["required_components"]
    assert "Cosmos-Predict2.5-2B" in components["vae_model_path"]
    assert "Cosmos-Reason1-7B" in components["text_encoder_model_path"]
    assert entry.default_call_kwargs["controlnet_variant"] == "edge"


def test_cosmos_predict25_accepts_safe_local_checkpoint_files(tmp_path, monkeypatch) -> None:
    transformer = tmp_path / "cosmos-2b.safetensors"
    vae = tmp_path / "tokenizer.safetensors"
    text_encoder = tmp_path / "reason1"
    transformer.write_bytes(b"transformer")
    vae.write_bytes(b"vae")
    text_encoder.mkdir()
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", fake_from_pretrained)

    CosmosPredict2p5Pipeline.from_pretrained(
        model_path=transformer,
        vae_model_path=vae,
        text_encoder_model_path=text_encoder,
        device="cpu",
    )

    assert captured["model_id"] == "cosmos-predict2.5-2b"
    assert captured["checkpoint_overrides"] == {
        "transformer": str(transformer.resolve()),
        "vae": str(vae.resolve()),
        "text-encoder": str(text_encoder.resolve()),
        "tokenizer": str(text_encoder.resolve()),
    }


def test_cosmos_predict25_reads_model_path_from_runner_options(tmp_path, monkeypatch) -> None:
    transformer = tmp_path / "cosmos-2b.safetensors"
    transformer.write_bytes(b"transformer")
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", fake_from_pretrained)

    runner_options = {"model_path": str(transformer), "model_id": "cosmos-predict-2.5"}
    CosmosPredict2p5Pipeline.from_pretrained(model_path=runner_options, device="cpu")

    assert captured["model_id"] == "cosmos-predict2.5-2b"
    assert captured["checkpoint_overrides"] == {"transformer": str(transformer.resolve())}
    assert CosmosPredict2p5Pipeline.plan(model_path=runner_options)["checkpoint"] == str(transformer)
