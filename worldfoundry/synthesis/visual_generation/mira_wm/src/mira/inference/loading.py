"""Load the right world-model class from a checkpoint directory (used by the offline tools)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch

    from worldfoundry.synthesis.visual_generation.mira_wm.src.mira.world_model.latent_world_model import (
        LatentWorldModel,
    )


def load_world_model(checkpoint_path: Path, device: str | torch.device) -> tuple[LatentWorldModel, Any]:
    """Load the right world-model class from a checkpoint dir.

    Dispatches on ``model.architecture._target_`` in the saved ``world_model_config.yaml`` and calls
    that class's ``load_from_checkpoint`` (single-player or 4-player), so the saved config goes
    through the same ``_target_``/removed-field cleaning the dedicated loaders apply. Falls back to
    :class:`LatentWorldModel` for older checkpoints without a ``_target_``. Returns ``(model, run_config)``.
    """
    from omegaconf import OmegaConf  # optional dep, used only on the real path

    from worldfoundry.synthesis.visual_generation.mira_wm.src.mira.world_model.latent_world_model import (
        LatentWorldModel,
    )
    from worldfoundry.synthesis.visual_generation.mira_wm.src.mira.world_model.multi_wrapper_world_model import (
        MultiWrapperWorldModel,
    )

    config_path = LatentWorldModel._find_config(checkpoint_path)
    if config_path is None:
        raise FileNotFoundError(
            f"Could not find '{LatentWorldModel.CONFIG_FILENAME}' in parent directories of {checkpoint_path}."
        )

    cfg = OmegaConf.load(config_path)
    arch_target = OmegaConf.select(cfg, "model.architecture._target_")
    model_cls = (
        MultiWrapperWorldModel
        if arch_target is not None and str(arch_target).endswith("MultiWrapperWorldModel")
        else LatentWorldModel
    )
    # Both variants share the LatentWorldModel surface (duck-typed); annotate as such.
    return cast("LatentWorldModel", model_cls.load_from_checkpoint(checkpoint_path, device=device)), cfg
