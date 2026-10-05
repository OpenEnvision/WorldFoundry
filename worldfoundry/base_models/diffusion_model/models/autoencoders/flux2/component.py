"""FLUX.2 :class:`~...contracts.LatentEncoder` construction and batch encode.

Loads ``ae.safetensors`` from ``black-forest-labs/FLUX.2-dev`` through
:class:`~...loaders.module.NativeModuleLoader`.  :func:`encode_video_batch_refs`
folds ``[B,T,H,W,C]`` frames through the 2-D AE.  The module itself
exposes ``encode`` / ``decode`` on :class:`~.model.Flux2Autoencoder`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ....components import ComponentBuildContext, ComponentKey, ComponentKind
from ....loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from ....optimizations import RuntimePolicy
from .model import Flux2Autoencoder, Flux2AutoencoderConfig

FLUX2_REPO_ID = "black-forest-labs/FLUX.2-dev"
FLUX2_AUTOENCODER_FILENAME = "ae.safetensors"


def default_flux2_autoencoder_checkpoint() -> CheckpointSpec:
    """Describe the filtered Hub asset without downloading it eagerly."""

    return CheckpointSpec(
        repo_id=FLUX2_REPO_ID,
        files=(FLUX2_AUTOENCODER_FILENAME,),
        allow_patterns=(FLUX2_AUTOENCODER_FILENAME,),
    )


def build_flux2_autoencoder(context: ComponentBuildContext) -> Flux2Autoencoder:
    """Build one FLUX.2 autoencoder using the shared native module loader."""

    checkpoint = context.checkpoint() or default_flux2_autoencoder_checkpoint()
    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Flux2Autoencoder,
            config={"params": Flux2AutoencoderConfig()},
        ),
        checkpoint,
        context.policy,
    )
    if not isinstance(module, Flux2Autoencoder):
        raise TypeError(f"expected Flux2Autoencoder, got {type(module).__name__}")
    return module


def load_flux2_autoencoder(
    checkpoint_path: str | os.PathLike[str] | None = None,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    policy: RuntimePolicy | None = None,
) -> Flux2Autoencoder:
    """Convenience entrypoint backed by the same component factory as recipes."""

    configured = checkpoint_path or os.environ.get("WORLDFOUNDRY_FLUX2_AE_PATH") or os.environ.get("AE_MODEL_PATH")
    checkpoint = (
        CheckpointSpec(source=str(Path(configured).expanduser()))
        if configured
        else default_flux2_autoencoder_checkpoint()
    )
    context = ComponentBuildContext(
        model_id="flux2",
        key=_AUTOENCODER_COMPONENT_KEY,
        checkpoints={"weights": checkpoint},
        policy=policy or RuntimePolicy(device=device, dtype=dtype),
    )
    return build_flux2_autoencoder(context)


def encode_video_batch_refs(
    autoencoder: Flux2Autoencoder,
    video_batch: torch.Tensor,
) -> torch.Tensor:
    """Encode ``[B, T, H, W, C]`` normalized frames to ``[B, T, C, h, w]``."""

    if video_batch.ndim != 5:
        raise ValueError(f"Expected video_batch with shape [B, T, H, W, C], got {tuple(video_batch.shape)}")
    batch, frames, height, width, channels = video_batch.shape
    if channels != 3:
        raise ValueError(f"Expected three RGB channels, got {channels}")

    images = video_batch.permute(0, 1, 4, 2, 3).reshape(
        batch * frames,
        channels,
        height,
        width,
    )
    parameter = next(autoencoder.parameters())
    encoded = autoencoder.encode(images.to(device=parameter.device, dtype=parameter.dtype))
    return encoded.reshape(batch, frames, *encoded.shape[1:])


# A latent encoder is reusable even when a benchmark consumes it outside the
# standard denoising execution strategy.
_AUTOENCODER_COMPONENT_KEY = ComponentKey(ComponentKind.LATENT_ENCODER, "flux2")


__all__ = [
    "FLUX2_AUTOENCODER_FILENAME",
    "FLUX2_REPO_ID",
    "build_flux2_autoencoder",
    "default_flux2_autoencoder_checkpoint",
    "encode_video_batch_refs",
    "load_flux2_autoencoder",
]
