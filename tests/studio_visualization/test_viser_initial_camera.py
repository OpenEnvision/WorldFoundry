# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Studio's Viser viewer points new clients at the geometry on every supported Viser API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from worldfoundry.studio.visualization.backends.viser import _set_initial_camera

pytestmark = pytest.mark.unit

POSITION = (3.0, 3.0, 3.0)
LOOK_AT = (0.0, 0.0, 0.0)
UP = (0.0, 0.0, 1.0)


def test_initial_camera_configures_server_level_camera() -> None:
    initial_camera = SimpleNamespace()
    _set_initial_camera(SimpleNamespace(initial_camera=initial_camera), POSITION, LOOK_AT, UP, fov=1.0)

    assert vars(initial_camera) == {"position": POSITION, "look_at": LOOK_AT, "up": UP, "fov": 1.0}


def test_initial_camera_sets_each_connecting_client_without_server_level_camera() -> None:
    assigned: list[tuple[str, object]] = []

    class RecordingCamera:
        def __setattr__(self, name: str, value: object) -> None:
            assigned.append((name, value))

    callbacks: list[Any] = []
    _set_initial_camera(SimpleNamespace(on_client_connect=callbacks.append), POSITION, LOOK_AT, UP, fov=1.0)

    assert len(callbacks) == 1
    callbacks[0](SimpleNamespace(camera=RecordingCamera()))
    # The client position setter also moves look_at, so look_at must be assigned after it.
    assert assigned == [("up_direction", UP), ("position", POSITION), ("look_at", LOOK_AT), ("fov", 1.0)]
