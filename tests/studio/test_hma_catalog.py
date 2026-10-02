from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_hma_catalog_executes_official_runtime_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "sources"))

    entry = find_entry("hma")

    assert entry.display_name == "HMA"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(tmp_path / "ckpts" / "liruiw--hma-base-cont")
    assert entry.default_input_path == str(
        tmp_path / "sources" / "HMA" / "assets" / "langtable_prompt" / "frame_00.png"
    )
    assert entry.default_load_kwargs == {
        "model_id": "hma",
        "checkpoint_dir": str(tmp_path / "ckpts" / "liruiw--hma-base-cont"),
        "base_model_dir": str(tmp_path / "ckpts" / "stabilityai--stable-video-diffusion-img2vid"),
    }
    assert entry.default_call_kwargs["plan_only"] is False
    assert entry.default_call_kwargs["generated_frames"] == 6
    assert entry.default_call_kwargs["maskgit_steps"] == 2
    assert entry.default_call_kwargs["direction"] == "right"


def test_hma_inference_contract_uses_portable_paths_and_executes_by_default() -> None:
    spec = get_model_inference_spec("hma")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert variant.load_kwargs["checkpoint_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert variant.load_kwargs["base_model_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert task.inputs[0].default.startswith("${WORLDFOUNDRY_MODEL_SOURCE_DIR}/")
    assert task.default_call_kwargs["plan_only"] is False
    assert task.default_call_kwargs["generated_frames"] == 6
    assert task.default_call_kwargs["prompt_horizon"] == 3
