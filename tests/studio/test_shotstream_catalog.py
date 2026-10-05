from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_shotstream_catalog_executes_official_runtime_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    entry = find_entry("shotstream")

    assert entry.display_name == "ShotStream"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(tmp_path / "ckpts" / "KlingTeam--ShotStream")
    assert entry.default_load_kwargs == {
        "model_id": "shotstream",
        "checkpoint_dir": str(tmp_path / "ckpts" / "KlingTeam--ShotStream"),
        "wan_model_dir": str(tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-1.3B"),
    }
    assert entry.default_call_kwargs["plan_only"] is False
    assert entry.default_call_kwargs["seed"] == 42
    assert "input_csv" in entry.call_params


def test_shotstream_inference_contract_uses_portable_paths_and_executes_by_default() -> None:
    spec = get_model_inference_spec("shotstream")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert variant.load_kwargs["checkpoint_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert variant.load_kwargs["wan_model_dir"].startswith("${WORLDFOUNDRY_CKPT_DIR}/")
    assert task.default_call_kwargs["plan_only"] is False
    assert task.default_call_kwargs["fps"] == 16
