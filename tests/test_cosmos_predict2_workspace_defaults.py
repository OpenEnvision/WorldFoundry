from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.pipelines.cosmos.pipeline_cosmos_predict2 import CosmosPredict2Pipeline
from worldfoundry.studio.inference import catalog as studio_catalog


def test_cosmos_predict2_accepts_sharded_t5_safetensors_export(tmp_path: Path) -> None:
    model_root = tmp_path / "nvidia--Cosmos-Predict2-2B-Video2World"
    text_root = model_root / "text_encoder"
    tokenizer_root = model_root / "tokenizer"
    for relative in (
        "model-720p-16fps.pt",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "tokenizer/config.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.pth",
    ):
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (text_root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"encoder.block.0":"model-00001-of-00002.safetensors",'
        '"encoder.block.1":"model-00002-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )

    overrides = CosmosPredict2Pipeline._checkpoint_overrides(
        model_root,
        {
            "checkpoint_path": str(model_root),
            "text_encoder_model_path": str(text_root),
            "text_tokenizer_path": str(tokenizer_root),
        },
        model_id="cosmos-predict2-2b-video2world",
    )

    assert overrides is not None
    text_encoder = overrides["text-encoder"]
    assert isinstance(text_encoder, CheckpointSpec)
    assert text_encoder.sources == (str(text_root.resolve()),)
    assert text_encoder.files == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    assert overrides["text-tokenizer"] == str(tokenizer_root.resolve())


def test_cosmos_predict2_workspace_defaults_use_complete_video2world_export(monkeypatch, tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    model_root = checkpoint_root / "nvidia--Cosmos-Predict2-2B-Video2World"
    for relative in (
        "model-720p-16fps.pt",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.pth",
    ):
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))

    entry = studio_catalog.find_entry("cosmos-predict2")

    assert entry.default_model_ref == str(model_root)
    assert entry.default_load_kwargs == {
        "checkpoint_path": str(model_root),
        "text_encoder_model_path": str(model_root / "text_encoder"),
        "text_tokenizer_path": str(model_root / "tokenizer"),
        "offload_mode": "block",
    }
    assert entry.default_input_path.endswith("worldfoundry/data/test_cases/studio_demo/00/image.jpg")
    assert entry.default_call_kwargs["num_frames"] == 93
    assert entry.default_call_kwargs["num_inference_steps"] == 35
    assert entry.default_call_kwargs["height"] == 704
    assert entry.default_call_kwargs["width"] == 1280
