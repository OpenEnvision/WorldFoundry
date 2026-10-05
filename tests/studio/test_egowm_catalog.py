from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_egowm_catalog_executes_official_runtime_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setenv("WORLDFOUNDRY_MODEL_SOURCE_DIR", str(tmp_path / "sources"))

    entry = find_entry("egowm")

    assert entry.display_name == "EgoWM"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(tmp_path / "ckpts" / "anuragba--egowm" / "svd_25dof_nav.pth")
    assert entry.default_input_path == ""
    assert entry.default_load_kwargs == {
        "model_id": "egowm",
        "checkpoint_path": str(tmp_path / "ckpts" / "anuragba--egowm" / "svd_25dof_nav.pth"),
        "base_model_dir": str(tmp_path / "ckpts" / "stabilityai--stable-video-diffusion-img2vid"),
    }
    assert entry.default_call_kwargs["plan_only"] is False
    assert entry.default_call_kwargs["num_frames"] == 8
    assert entry.default_call_kwargs["num_inference_steps"] == 25


def test_egowm_inference_contract_uses_portable_paths_and_executes_by_default() -> None:
    spec = get_model_inference_spec("egowm")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert variant.load_kwargs["checkpoint_path"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert variant.load_kwargs["base_model_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert task.default_call_kwargs["plan_only"] is False
    assert task.default_call_kwargs["num_frames"] == 8
    assert task.default_call_kwargs["num_inference_steps"] == 25
    assert any(field.field_id == "conditions_path" for field in task.inputs)
    assert next(field for field in task.inputs if field.field_id == "input_path").default is None
