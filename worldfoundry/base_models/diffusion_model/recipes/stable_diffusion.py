# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Declarative SD2 inpainting and WonderWorld Marigold component graphs."""

from __future__ import annotations

from ..components import STANDARD_COMPONENT_BINDINGS, ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.stable_diffusion import build_sd_image_codec
from ..models.denoisers.stable_diffusion import build_sd_unet_denoiser
from ..models.encoders.stable_diffusion import build_sd_clip_conditioner
from ..models.initializers.stable_diffusion import build_sd_image_initializer
from ..schedulers.stable_diffusion import build_sd_discrete_scheduler
from .spec import NativeDiffusionRecipe

SD2_INPAINT_REPO = "sd2-community/stable-diffusion-2-inpainting"
SD2_INPAINT_REVISION = "5f74973cbb64c8568780732c17f43eb269d63a0d"
MARIGOLD_DEPTH_REVISION = "f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c"
MARIGOLD_NORMALS_REVISION = "b105f7cfa1972b57d8e9ed72a896b2768a52f8e9"


def _image_recipe(model_id: str, repo: str, revision: str, mode: str) -> NativeDiffusionRecipe:
    inpaint = mode == "rgb"
    tokenizer_files = (
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
    )
    checkpoints = {
        "unet": CheckpointSpec(
            repo_id=repo,
            revision=revision,
            files=("unet/diffusion_pytorch_model.fp16.safetensors",),
            resources=("unet/config.json",),
        ),
        "text_encoder": CheckpointSpec(
            repo_id=repo,
            revision=revision,
            files=("text_encoder/model.fp16.safetensors",),
            resources=("text_encoder/config.json",),
        ),
        "tokenizer": CheckpointSpec(repo_id=repo, revision=revision, files=tokenizer_files),
        "vae": CheckpointSpec(
            repo_id=repo,
            revision=revision,
            files=("vae/diffusion_pytorch_model.fp16.safetensors",),
            resources=("vae/config.json",),
        ),
        "scheduler": CheckpointSpec(repo_id=repo, revision=revision, files=("scheduler/scheduler_config.json",)),
    }
    bindings = {**STANDARD_COMPONENT_BINDINGS, "latent_encoder": ComponentKey(ComponentKind.DECODER)}
    return NativeDiffusionRecipe(
        model_id=model_id,
        components=(
            ComponentSpec(
                ComponentKey(ComponentKind.DENOISER),
                build_sd_unet_denoiser,
                {"weights": "unet"},
                {"mode": "inpaint" if inpaint else "prediction"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.CONDITIONER),
                build_sd_clip_conditioner,
                {"weights": "text_encoder", "tokenizer": "tokenizer"},
                {"empty_text": not inpaint},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.DECODER), build_sd_image_codec, {"weights": "vae"}, {"mode": mode}
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.LATENT_INITIALIZER),
                build_sd_image_initializer,
                options={"mode": "inpaint" if inpaint else "prediction"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.SCHEDULER),
                build_sd_discrete_scheduler,
                {"weights": "scheduler"},
                {"algorithm": "euler-unscaled" if mode == "depth" else "ddim"},
            ),
        ),
        checkpoints=checkpoints,
        execution=ExecutionSpec(bindings=bindings),
        capabilities=frozenset({"image-inpainting" if inpaint else "monocular-" + mode}),
        metadata={"native_inference": True, "architecture": "sd2-unet", "output_layout": "BCHW", "output_mode": mode},
    )


def sd2_inpaint_recipe() -> NativeDiffusionRecipe:
    """Nine-channel SD2, sampled masked-image VAE, text CFG and DDIM."""
    return _image_recipe("sd2-inpaint", SD2_INPAINT_REPO, SD2_INPAINT_REVISION, "rgb")


def wonderworld_depth_recipe() -> NativeDiffusionRecipe:
    """Marigold v1-0 with WonderWorld's unit-noise, unscaled Euler semantics."""
    return _image_recipe("wonderworld-marigold-depth", "prs-eth/marigold-depth-v1-0", MARIGOLD_DEPTH_REVISION, "depth")


def marigold_normals_recipe() -> NativeDiffusionRecipe:
    """Marigold v0-1 normals, empty CLIP text and deterministic DDIM."""
    return _image_recipe("marigold-normals-v0-1", "prs-eth/marigold-normals-v0-1", MARIGOLD_NORMALS_REVISION, "normals")
