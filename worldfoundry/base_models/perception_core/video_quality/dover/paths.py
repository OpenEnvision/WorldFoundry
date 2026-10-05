"""Path helpers for the DOVER video quality model."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.io.paths import package_data_path

_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return _ROOT


def config_path() -> Path:
    return package_data_path('models', 'runtime', 'configs', 'dover', 'dover.yml')


def checkpoint_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_DOVER_CKPT")
    if explicit:
        return Path(explicit).expanduser()

    evalcrafter_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if evalcrafter_dir:
        candidate = Path(evalcrafter_dir).expanduser() / "DOVER" / "pretrained_weights" / "DOVER.pth"
        if candidate.is_file():
            return candidate

    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        roots = [
            Path(ckpt_dir).expanduser() / "DOVER" / "pretrained_weights",
            Path(ckpt_dir).expanduser() / "evalcrafter" / "DOVER" / "pretrained_weights",
        ]
        for root in roots:
            candidate = root / "DOVER.pth"
            if candidate.is_file():
                return candidate

    return Path("checkpoints") / "DOVER" / "pretrained_weights" / "DOVER.pth"
