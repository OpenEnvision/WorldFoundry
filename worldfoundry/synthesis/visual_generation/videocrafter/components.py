"""Lazy access to VideoCrafter inference components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoCrafterComponents:
    """Functions required by the VideoCrafter runtime."""

    batch_ddim_sampling: object
    load_image_batch: object
    load_model_checkpoint: object
    instantiate_from_config: object


def load_videocrafter_components() -> VideoCrafterComponents:
    """Load the packaged inference helpers and shared LVDM factory."""

    from .videocrafter_inference import (
        batch_ddim_sampling,
        load_image_batch,
        load_model_checkpoint,
    )
    from worldfoundry.core.model_loading.factory import instantiate_from_config

    return VideoCrafterComponents(
        batch_ddim_sampling=batch_ddim_sampling,
        load_image_batch=load_image_batch,
        load_model_checkpoint=load_model_checkpoint,
        instantiate_from_config=instantiate_from_config,
    )


__all__ = ["VideoCrafterComponents", "load_videocrafter_components"]
