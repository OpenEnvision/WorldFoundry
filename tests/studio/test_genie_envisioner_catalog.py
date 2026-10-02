from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_genie_envisioner_catalog_executes_local_checkpoint_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    entry = find_entry("genie-envisioner")

    checkpoint = (
        tmp_path
        / "ckpts"
        / "agibot-world--Genie-Envisioner"
        / "GE_base_fast_v0.1.safetensors"
    )
    assert entry.display_name == "Genie Envisioner"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(checkpoint)
    assert entry.default_input_path.endswith("test_vla_case1/aloha/observation_images_cam_high.png")
    assert entry.default_load_kwargs == {
        "model_id": "genie-envisioner",
        "checkpoint_path": str(checkpoint),
        "base_model_dir": str(tmp_path / "ckpts" / "Lightricks--LTX-Video"),
    }
    assert entry.default_call_kwargs["input_mode"] == "explicit-three-view"
    assert len(entry.default_call_kwargs["input_views"]) == 3
    assert all("test_vla_case1/aloha/" in path for path in entry.default_call_kwargs["input_views"])
    assert entry.default_call_kwargs["n_previous"] == 4
    assert entry.default_call_kwargs["plan_only"] is False


def test_genie_envisioner_inference_contract_uses_portable_paths() -> None:
    spec = get_model_inference_spec("genie-envisioner")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert variant.status == "requires_local_checkpoints"
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert all(
        str(variant.load_kwargs[key]).startswith("${WORLDFOUNDRY_CKPT_DIR}/")
        for key in ("checkpoint_path", "base_model_dir")
    )
    assert task.inputs[0].default.startswith("${WORLDFOUNDRY_REPO_ROOT}/")
    assert task.default_call_kwargs["input_mode"] == "explicit-three-view"
    assert len(task.default_call_kwargs["input_views"]) == 3
    assert task.default_call_kwargs["plan_only"] is False
