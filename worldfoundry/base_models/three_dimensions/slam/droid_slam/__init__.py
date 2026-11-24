"""Canonical DROID-SLAM runtime used by geometry metrics."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES


def checkpoint_path() -> Path:
    asset = BASE_MODEL_CAPABILITIES["droid_slam"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


__all__ = ["Droid", "checkpoint_path"]


def __getattr__(name: str):
    if name == "Droid":
        from .droid import Droid

        return Droid
    raise AttributeError(name)
