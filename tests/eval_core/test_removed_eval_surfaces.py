"""Tombstones for eval surfaces that were deleted without a successor."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_visual_generation_training_package_was_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldfoundry.training.visual_generation")
    assert not (REPO_ROOT / "worldfoundry" / "training" / "visual_generation").exists()


def test_solaris_multiplayer_eval_task_was_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldfoundry.evaluation.tasks.official.solaris_multiplayer")
    assert not (REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "official").exists()
