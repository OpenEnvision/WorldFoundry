"""Resolve inference checkpoints from local paths or W&B run URLs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_WANDB_URL_RE = re.compile(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/]+)")

logger = logging.getLogger(__name__)






def _find_latest_checkpoint(output_dir: Path) -> Path:
    """Find the latest ``checkpoint-{step}/checkpoint.pth`` in an output directory."""
    checkpoint_dirs = sorted(
        output_dir.glob("checkpoint-*/checkpoint.pth"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoint_path = checkpoint_dirs[-1]
    logger.info(f"Resolved to latest checkpoint: {checkpoint_path}")
    return checkpoint_path


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint path from a local path or W&B run URL.

    Accepts:
    - A direct path to a ``checkpoint.pth`` file.
    - A checkpoint directory (``checkpoint-{step}/``) containing ``checkpoint.pth``.
    - An output directory containing ``checkpoint-{step}/`` subdirectories (picks latest).
    - A W&B run URL like ``https://wandb.ai/entity/project/runs/run_id/overview``.
    """
    if not isinstance(checkpoint, Path):
        # First check if it's a W&B URL; if so, resolve to the run's output_dir.
        match = _WANDB_URL_RE.match(checkpoint)
        if match:
            entity, project, run_id = match.groups()

            import wandb  # noqa: PLC0415 -- optional dep, used only for W&B-URL checkpoints

            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")
            output_dir = Path(run.config["run"]["output_dir"])
            logger.info(f"W&B run output_dir for {checkpoint}: {output_dir}")
            return _find_latest_checkpoint(output_dir)

        path = Path(checkpoint)
    else:
        # A Path is always local.
        path = checkpoint

    # Direct path to a .pth file.
    if path.suffix == ".pth":
        return path

    # A checkpoint-{step} dir containing checkpoint.pth.
    if (path / "checkpoint.pth").is_file():
        return path / "checkpoint.pth"

    # An output dir containing checkpoint-{step}/ subdirectories.
    return _find_latest_checkpoint(path)
