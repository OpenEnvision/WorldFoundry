from __future__ import annotations

from unittest.mock import Mock

from worldfoundry.cli.model_run import load_model_run_schema


def test_framepack_in_tree_runner_is_exposed_as_runnable() -> None:
    load_model_run_schema.cache_clear()

    schema = load_model_run_schema("framepack")

    assert schema.integration_status == "integrated"
    assert schema.runner_entry_kind == "runnable_runner"
    assert schema.runnable is True
    assert schema.blocked_reason == ""
    assert schema.runner_target == "worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner"


def test_framepack_local_hunyuan_tokenizer_bypasses_invalid_root_config(tmp_path, monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    from worldfoundry.synthesis.visual_generation.framepack.framepack_runtime import inference

    for name in (
        "f1_mode",
        "high_vram",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "feature_extractor",
        "image_encoder",
        "transformer",
        "stream",
        "outputs_folder",
    ):
        monkeypatch.setattr(inference, name, None, raising=False)
    loaders = {}
    for name in (
        "LlamaModel",
        "CLIPTextModel",
        "LlamaTokenizerFast",
        "CLIPTokenizer",
        "AutoencoderKLHunyuanVideo",
        "SiglipImageProcessor",
        "SiglipVisionModel",
        "HunyuanVideoTransformer3DModelPacked",
    ):
        loader = Mock()
        loader.from_pretrained.return_value.cpu.return_value = Mock()
        monkeypatch.setattr(inference, name, loader)
        loaders[name] = loader.from_pretrained
    monkeypatch.setattr(inference, "enable_framepack_global_patches", lambda: None)
    monkeypatch.setattr(inference, "get_cuda_free_memory_gb", lambda _: 24)
    monkeypatch.setattr(inference.DynamicSwapInstaller, "install_model", Mock())
    monkeypatch.setattr(inference, "AsyncStream", Mock())

    hunyuan_root = tmp_path / "hunyuan"
    checkpoint_path = tmp_path / "checkpoint"
    redux_root = tmp_path / "redux"
    output_dir = tmp_path / "output"
    inference.initialize(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        hunyuan_root=str(hunyuan_root),
        flux_redux_root=str(redux_root),
    )

    loaders["LlamaTokenizerFast"].assert_called_once_with(str(hunyuan_root / "tokenizer"))
    loaders["HunyuanVideoTransformer3DModelPacked"].assert_called_once_with(
        str(checkpoint_path), torch_dtype=torch.bfloat16
    )
    loaders["SiglipImageProcessor"].assert_called_once_with(str(redux_root), subfolder="feature_extractor")
    assert inference.outputs_folder == str(output_dir / ".framepack_outputs")
