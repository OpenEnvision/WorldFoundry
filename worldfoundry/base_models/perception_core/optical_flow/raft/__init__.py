"""Canonical RAFT optical-flow runtime."""

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

from .raft import RAFT
from .utils.utils import InputPadder


def checkpoint_path() -> Path:
    """Resolve the RAFT Things checkpoint using the shared capability registry."""
    asset = BASE_MODEL_CAPABILITIES["raft"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


__all__ = ["RAFT", "InputPadder", "checkpoint_path"]
