from __future__ import annotations

from worldfoundry.synthesis.action_generation.vlanext.runtime import _local_hfd_ref


def test_local_hfd_ref_supports_legacy_hfd_models_root(tmp_path, monkeypatch):
    hfd_root = tmp_path / "hfd"
    model_dir = tmp_path / "hfd_models" / "Qwen--Qwen3-VL-2B-Instruct"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))

    assert _local_hfd_ref("Qwen/Qwen3-VL-2B-Instruct") == str(model_dir)
