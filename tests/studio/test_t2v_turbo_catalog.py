from __future__ import annotations

import worldfoundry.studio.inference.catalog as catalog
from worldfoundry.studio.inference.catalog import (
    _t2v_turbo_base_ref,
    _t2v_turbo_default_load_kwargs,
    _t2v_turbo_lora_ref,
)


def test_t2v_turbo_catalog_resolves_both_owner_prefixed_checkpoints(tmp_path, monkeypatch) -> None:
    base = tmp_path / "ckpts" / "VideoCrafter--VideoCrafter2" / "model.ckpt"
    lora = tmp_path / "ckpts" / "jiachenli-ucsb--T2V-Turbo-VC2" / "unet_lora.pt"
    base.parent.mkdir(parents=True)
    lora.parent.mkdir(parents=True)
    base.touch()
    lora.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _t2v_turbo_base_ref() == str(base)
    assert _t2v_turbo_lora_ref() == str(lora)
    assert _t2v_turbo_default_load_kwargs() == {
        "base_checkpoint": str(base),
        "lora_checkpoint": str(lora),
    }


def test_t2v_turbo_catalog_uses_public_hub_fallbacks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *names: [])

    assert _t2v_turbo_base_ref() == "VideoCrafter/VideoCrafter2"
    assert _t2v_turbo_lora_ref() == "jiachenli-ucsb/T2V-Turbo-VC2"


def test_t2v_turbo_exact_catalog_entry_binds_base_and_lora(tmp_path, monkeypatch) -> None:
    base = tmp_path / "ckpts" / "VideoCrafter--VideoCrafter2" / "model.ckpt"
    lora = tmp_path / "ckpts" / "jiachenli-ucsb--T2V-Turbo-VC2" / "unet_lora.pt"
    base.parent.mkdir(parents=True)
    lora.parent.mkdir(parents=True)
    base.touch()
    lora.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    entry = catalog.find_entry("t2v_turbo_t2v")

    assert entry.default_model_ref == str(base)
    assert entry.default_load_kwargs == {
        "base_checkpoint": str(base),
        "lora_checkpoint": str(lora),
    }
    assert {"base_checkpoint", "lora_checkpoint", "offload_mode"} <= set(entry.load_params)
