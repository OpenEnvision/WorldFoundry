# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from worldfoundry.studio.inference.catalog import discovery

pytestmark = pytest.mark.unit

# Studio entries that serve one variant of a catalog model and name that variant.
VARIANT_DISPLAY_NAMES = {
    "dreamx-world-5b": "DreamX-World 5B AR",
    "echo-memory-context-k1": "Echo-Memory Context K=1",
    "gr00t": "GR00T N1.7",
    "joyai-echo-wm": "JoyAI Echo-WM",
    "matrix-game-3.5-first-person": "Matrix-Game 3.5 First-Person Base",
    "smolvla": "SmolVLA LIBERO",
}


def test_studio_display_names_come_from_the_model_catalog() -> None:
    catalog_names = discovery._catalog_model_names()
    mismatched = {
        entry.model_id: entry.display_name
        for entry in discovery.discover_catalog()
        if entry.model_id in catalog_names and entry.display_name != catalog_names[entry.model_id]
    }

    assert mismatched == VARIANT_DISPLAY_NAMES
