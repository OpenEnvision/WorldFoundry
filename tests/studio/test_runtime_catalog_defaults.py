# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NoReturn

import pytest

from worldfoundry.studio.inference import catalog

pytestmark = pytest.mark.unit


def _unexpected_catalog_scan() -> NoReturn:
    raise AssertionError("concrete runtime lookup must preserve canonical defaults without a full catalog scan")


@pytest.mark.parametrize(
    ("model_id", "canonical_id", "checkpoint", "seed", "requires_image"),
    (
        ("cogvideox_2b_t2v", "cogvideox_2b_t2v", "CogVideoX-2b", 42, False),
        ("cogvideox-2b", "cogvideox_2b_t2v", "CogVideoX-2b", 42, False),
        ("cogvideox_5b_i2v", "cogvideox_5b_i2v", "CogVideoX-5b-I2V", 45, True),
        ("cogvideox-5b-i2v", "cogvideox_5b_i2v", "CogVideoX-5b-I2V", 45, True),
    ),
)
def test_concrete_runtime_lookup_retains_its_own_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    canonical_id: str,
    checkpoint: str,
    seed: int,
    requires_image: bool,
) -> None:
    requested: list[tuple[str, ...]] = []

    def checkpoint_ref(*names: str, fallback: str = "") -> str:
        requested.append(names)
        return fallback

    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_MODEL_REF", raising=False)
    monkeypatch.delenv(f"WORLDFOUNDRY_STUDIO_MODEL_REF_{canonical_id.upper()}", raising=False)
    monkeypatch.setattr(catalog, "discover_catalog", _unexpected_catalog_scan)
    monkeypatch.setattr(catalog, "_discover_catalog_infos", _unexpected_catalog_scan)
    monkeypatch.setattr(catalog, "_checkpoint_model_ref", checkpoint_ref)

    entry = catalog.find_runtime_entry(model_id)

    assert requested == [(checkpoint, f"THUDM--{checkpoint}")]
    assert entry.model_id == canonical_id
    assert entry.module_path.endswith(f"pipeline_{canonical_id}")
    assert entry.default_model_ref == f"THUDM/{checkpoint}"
    assert entry.default_call_kwargs == {
        "fps": 8,
        "height": 480,
        "width": 720,
        "num_frames": 49,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "seed": seed,
    }
    assert ("i2v" in entry.tags) is requires_image
    if requires_image:
        assert entry.default_input_path.endswith("studio_demo/00/image.jpg")
        assert "sparkler" in entry.default_prompt.lower()
    else:
        assert entry.default_input_path == ""
        assert "city street" in entry.default_prompt.lower()
