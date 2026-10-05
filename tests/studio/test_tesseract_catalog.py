from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_tesseract_catalog_executes_local_rgbdn_runtime_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    entry = find_entry("tesseract")

    checkpoint = tmp_path / "ckpts" / "anyeZHY--tesseract" / "tesseract_v01e_rgbdn_sft"
    assert entry.display_name == "TesserAct"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(checkpoint)
    assert entry.default_input_path.endswith("test_vla_case1/droid/exterior_image_1_left.png")
    assert entry.default_load_kwargs == {
        "model_id": "tesseract",
        "checkpoint_dir": str(checkpoint),
        "base_model_dir": str(tmp_path / "ckpts" / "THUDM--CogVideoX-5b-I2V"),
    }
    assert entry.default_call_kwargs["geometry_mode"] == "synthetic-gradient"
    assert entry.default_call_kwargs["num_frames"] == 49
    assert entry.default_call_kwargs["num_inference_steps"] == 4
    assert entry.default_call_kwargs["plan_only"] is False


def test_tesseract_inference_contract_uses_only_portable_paths() -> None:
    spec = get_model_inference_spec("tesseract")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert variant.status == "requires_local_checkpoints"
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert variant.load_kwargs["checkpoint_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert variant.load_kwargs["base_model_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert task.inputs[0].default.startswith("${WORLDFOUNDRY_REPO_ROOT}/")
    assert task.default_call_kwargs["geometry_mode"] == "synthetic-gradient"
    assert task.default_call_kwargs["plan_only"] is False
