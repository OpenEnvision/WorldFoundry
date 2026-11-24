"""Inference checkpoint loading for PerceptionLM consolidated weights."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("WORLDFOUNDRY_PERCEPTIONLM_CHECKPOINT")


def get_consolidated_ckpt_path(ckpt_dir: Path, mp_rank: int = 0, mp_size: int = 1) -> Path:
    if mp_size == 1:
        if mp_rank != 0:
            raise ValueError("Single-rank consolidated PerceptionLM checkpoint expects mp_rank=0.")
        no_rank_path = ckpt_dir / "consolidated.pth"
        if no_rank_path.exists():
            return no_rank_path
    return ckpt_dir / f"consolidated.{mp_rank:02d}.pth"


def load_consolidated_checkpoint(
    model: nn.Module,
    consolidated_path: str,
    vision_model_path: str | None = None,
) -> None:
    ckpt_path = Path(consolidated_path)
    cp_file = get_consolidated_ckpt_path(ckpt_path, mp_rank=0, mp_size=1)
    if cp_file.exists():
        state_dict = torch.load(cp_file, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
    else:
        checkpoint_files = sorted(ckpt_path.glob("consolidated.*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No consolidated PerceptionLM checkpoint found in {ckpt_path}.")
        state_dict = {}
        for ckpt_file in checkpoint_files:
            part = torch.load(ckpt_file, weights_only=True)
            if "model" in part:
                part = part["model"]
            state_dict.update(part)

    model.vision_projector.init_tensors()
    model.vision_model.init_tensors()
    model.rope_embeddings.reset_parameters()

    if vision_model_path is not None:
        model.vision_model.load_ckpt(vision_model_path)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = [k for k in missing_keys if "tied_module.weight" not in k]
    if vision_model_path is not None:
        missing_keys = [k for k in missing_keys if "vision_model." not in k]
    if missing_keys:
        logger.warning("Missing keys when loading PerceptionLM: %s", missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys when loading PerceptionLM: %s", unexpected_keys)
