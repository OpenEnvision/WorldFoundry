from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.studio.inference.catalog import find_entry


def test_evoke_catalog_uses_local_checkpoint_and_unified_python() -> None:
    entry = find_entry("evoke")

    assert Path(entry.default_model_ref).name == "Evoke"
    components = entry.default_load_kwargs["required_components"]
    assert Path(components["vigeo_path"]).name == "ViGeo1.1"
    assert Path(components["python_executable"]).name.startswith("python")
    assert entry.default_call_kwargs["num_inference_steps"] == 3
    assert entry.default_call_kwargs["guidance_scale"] == 1.0


def test_magi2_catalog_uses_local_preview_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "magi2-preview").mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    entry = find_entry("magi2-preview")

    assert Path(entry.default_model_ref).name == "magi2-preview"
    components = entry.default_load_kwargs["required_components"]
    assert Path(components["python_executable"]).name.startswith("python")
    assert Path(components["python_executable"]).parent.parent.name == "magi2-preview"
    assert set(components) == {"python_executable"}
    assert entry.default_call_kwargs["use_refiner"] is False
    assert entry.default_call_kwargs["num_inference_steps"] == 100
    assert find_entry("magi2").model_id == "magi2-preview"


def test_magi2_text_environment_requires_transformers_55() -> None:
    from worldfoundry.runtime.conda import load_runtime_conda_env_specs, unified_env_blocker

    spec = load_runtime_conda_env_specs()["magi2-preview"]
    assert spec.env_name == "magi2-preview"
    assert "transformers==5.5.0" in spec.pip_packages
    assert unified_env_blocker(spec) is not None
