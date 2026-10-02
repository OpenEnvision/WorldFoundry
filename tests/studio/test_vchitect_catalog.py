from __future__ import annotations

import worldfoundry.studio.inference.catalog as catalog
from worldfoundry.studio.inference.catalog import _vchitect2_default_ref


def test_vchitect_catalog_resolves_owner_prefixed_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "ckpts" / "Vchitect--Vchitect-2.0-2B"
    checkpoint.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _vchitect2_default_ref() == str(checkpoint)


def test_vchitect_catalog_uses_public_hub_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *names: [])

    assert _vchitect2_default_ref() == "Vchitect/Vchitect-2.0-2B"
