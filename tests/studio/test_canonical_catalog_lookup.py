# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from worldfoundry.studio.inference import catalog

pytestmark = pytest.mark.unit


def _unexpected_catalog_scan() -> NoReturn:
    raise AssertionError("single-model lookup must not resolve the full catalog")


@pytest.fixture(scope="module")
def canonical_entries(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, catalog.CatalogEntry]]:
    checkpoint_root = tmp_path_factory.mktemp("canonical-checkpoints")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
        patch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))
        patch.delenv("WORLDFOUNDRY_STUDIO_MODEL_REF", raising=False)
        patch.delenv("WORLDFOUNDRY_STUDIO_FIXED_MODEL_REF", raising=False)
        yield {entry.model_id: entry for entry in catalog._canonical_runtime_entries()}


@pytest.mark.parametrize(
    ("model_id", "canonical_id"),
    (
        ("easyanimate_i2v", "easyanimate_i2v"),
        ("easyanimate-i2v", "easyanimate_i2v"),
        ("easyanimate", "easyanimate_i2v"),
        ("dynamicrafter_1024_i2v", "dynamicrafter_1024_i2v"),
        ("dynamicrafter-512-i2v", "dynamicrafter_512_i2v"),
        ("allegro-ti2v", "allegro_ti2v"),
        ("cogvideox", "cogvideox_5b_t2v"),
        ("cogvideox_2b_t2v", "cogvideox_5b_t2v"),
        ("cogvideox-5b-i2v", "cogvideox_5b_t2v"),
        ("gen_3_i2v", "gen_3_i2v"),
        ("ltx-video-i2v", "ltx_video_i2v"),
        ("ltx2_i2v", "ltx2_i2v"),
        ("minimax-i2v", "minimax_i2v"),
        ("t2v-turbo", "t2v_turbo_t2v"),
        ("vchitect-2-t2v", "vchitect_2_t2v"),
        ("videocrafter1_i2v", "videocrafter1_i2v"),
        ("videocrafter1-t2v", "videocrafter1_t2v"),
        ("videocrafter2_t2v", "videocrafter2_t2v"),
        ("wan2.1_i2v", "wan2.1_i2v"),
        ("wan2.1-t2v", "wan2.1_t2v"),
    ),
)
def test_single_model_lookup_preserves_canonical_contract(
    monkeypatch: pytest.MonkeyPatch,
    canonical_entries: dict[str, catalog.CatalogEntry],
    model_id: str,
    canonical_id: str,
) -> None:
    monkeypatch.setattr(catalog, "discover_catalog", _unexpected_catalog_scan)

    entry = catalog.find_entry(model_id)

    assert entry == canonical_entries[canonical_id]


def test_single_model_lookup_resolves_only_its_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[tuple[str, ...]] = []

    def checkpoint_ref(*names: str, fallback: str = "") -> str:
        requested.append(names)
        return fallback

    monkeypatch.setattr(catalog, "discover_catalog", _unexpected_catalog_scan)
    monkeypatch.setattr(catalog, "_discover_ast_pipelines", lambda: ())
    monkeypatch.setattr(catalog, "_checkpoint_model_ref", checkpoint_ref)
    monkeypatch.setattr(catalog, "_t2v_turbo_base_ref", _unexpected_catalog_scan)

    entry = catalog.find_entry("easyanimate")

    assert requested == [
        (
            "hfd/alibaba-pai--EasyAnimateV5.1-7b-zh-InP",
            "alibaba-pai--EasyAnimateV5.1-7b-zh-InP",
        )
    ]
    assert entry.default_model_ref == "alibaba-pai/EasyAnimateV5.1-7b-zh-InP"
    assert "sparkler" in entry.default_prompt.lower()
    assert entry.default_input_path.endswith("studio_demo/00/image.jpg")


def test_unregistered_canonical_lookup_does_not_discover_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "_discover_ast_pipelines", _unexpected_catalog_scan)

    assert list(catalog._canonical_runtime_entries("not-a-registered-model")) == []


@pytest.mark.parametrize(
    "override",
    ("WORLDFOUNDRY_STUDIO_MODEL_REF_EASYANIMATE_I2V", "WORLDFOUNDRY_STUDIO_MODEL_REF"),
)
def test_canonical_lookup_retains_explicit_checkpoint_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: str
) -> None:
    checkpoint = str(tmp_path / "custom-checkpoint")
    monkeypatch.setenv(override, checkpoint)
    monkeypatch.setattr(catalog, "discover_catalog", _unexpected_catalog_scan)

    assert catalog.find_entry("easyanimate-i2v").default_model_ref == checkpoint
    assert next(iter(catalog._canonical_runtime_entries("easyanimate_i2v"))).default_model_ref == checkpoint
