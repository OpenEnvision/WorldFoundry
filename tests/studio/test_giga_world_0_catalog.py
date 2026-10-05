from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_giga_world_catalog_executes_local_checkpoint_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    entry = find_entry("giga-world-0")

    transformer = (
        tmp_path / "ckpts" / "open-gigaai--GigaWorld-0-Video-GR1-2b" / "transformer"
    )
    assert entry.display_name == "GigaWorld-0"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(transformer)
    assert entry.default_input_path.endswith("test_vla_case1/droid/exterior_image_1_left.png")
    assert entry.default_load_kwargs == {
        "model_id": "giga-world-0",
        "transformer_model_dir": str(transformer),
        "text_encoder_model_dir": str(tmp_path / "ckpts" / "google-t5--t5-11b-encoder"),
        "vae_model_dir": str(
            tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-1.3B-Diffusers" / "vae"
        ),
    }
    assert entry.default_call_kwargs["attention_backend"] == "natten"
    assert entry.default_call_kwargs["num_frames"] == 61
    assert entry.default_call_kwargs["num_inference_steps"] == 30
    assert entry.default_call_kwargs["plan_only"] is False


def test_giga_world_inference_contract_uses_portable_paths() -> None:
    spec = get_model_inference_spec("giga-world-0")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert variant.status == "requires_local_checkpoints"
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert all(
        str(variant.load_kwargs[key]).startswith("${WORLDFOUNDRY_CKPT_DIR}/")
        for key in ("transformer_model_dir", "text_encoder_model_dir", "vae_model_dir")
    )
    assert task.inputs[0].default.startswith("${WORLDFOUNDRY_REPO_ROOT}/")
    assert task.default_call_kwargs["attention_backend"] == "natten"
    assert task.default_call_kwargs["plan_only"] is False
