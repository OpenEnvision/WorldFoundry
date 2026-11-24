"""Path helpers for the in-tree GroundingDINO Swin-T runtime."""

from __future__ import annotations

import os
from pathlib import Path

from worldfoundry.core.io.paths import REPO_ROOT

_ROOT = Path(__file__).resolve().parent


def config_path() -> Path:
    return _ROOT / "config" / "GroundingDINO_SwinT_OGC.py"


def checkpoint_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_GROUNDING_DINO_CKPT")
    if explicit:
        return Path(explicit).expanduser()

    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        roots = [
            Path(ckpt_dir).expanduser() / "hfd" / "ShilongLiu--GroundingDINO",
            Path(ckpt_dir).expanduser() / "GroundingDINO",
            Path(ckpt_dir).expanduser() / "evalcrafter" / "GroundingDINO",
        ]
        for root in roots:
            candidate = root / "groundingdino_swint_ogc.pth"
            if candidate.is_file():
                return candidate

    evalcrafter_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if evalcrafter_dir:
        candidate = Path(evalcrafter_dir).expanduser() / "GroundingDINO" / "groundingdino_swint_ogc.pth"
        if candidate.is_file():
            return candidate

    for candidate in (
        REPO_ROOT.parent / "ckpt" / "GroundingDINO" / "groundingdino_swint_ogc.pth",
        REPO_ROOT.parent / "ckpt" / "WorldScore" / "groundingdino_swint_ogc.pth",
    ):
        if candidate.is_file():
            return candidate

    return Path("checkpoints") / "ckpt" / "groundingdino_swint_ogc.pth"
