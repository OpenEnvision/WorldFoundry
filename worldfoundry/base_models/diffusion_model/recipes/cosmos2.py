"""Native Cosmos Predict2 Video2World 2B and 14B recipes.

Each public function returns a :class:`NativeDiffusionRecipe`.  Recipes
do not construct runners.

Both scales bind a Predict2 DiT, T5-11B conditioner, Cosmos 2.5 video
codec (encoder + decoder), Video2World initializer (``sigma_max=80``),
and a Karras x0 Adams-Bashforth-2 scheduler.  Strategy is ``standard``
with ``guidance_mode="positive"`` (CFG, not dual-expert).  Same latent
geometry: 16 channels, 8x spatial / 4x temporal compression.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.cosmos2p5 import build_cosmos25_video_codec
from ..models.denoisers.cosmos2 import build_cosmos2_2b_denoiser, build_cosmos2_14b_denoiser
from ..models.encoders.t5 import build_t5_encoder_conditioner
from ..models.initializers.cosmos2 import build_cosmos2_video2world_initializer
from ..schedulers import build_karras_x0_ab2_scheduler
from .spec import NativeDiffusionRecipe

COSMOS2_2B_MODEL_ID = "cosmos-predict2-2b-video2world"
COSMOS2_14B_MODEL_ID = "cosmos-predict2-14b-video2world"
COSMOS2_2B_REPO_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS2_14B_REPO_ID = "nvidia/Cosmos-Predict2-14B-Video2World"
COSMOS2_2B_REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
COSMOS2_14B_REVISION = "03b03a377fede782647afac998f674d9f358e319"
T5_REPO_ID = "google-t5/t5-11b"
T5_REVISION = "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3"


def _recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    denoiser_factory,
    aliases: tuple[str, ...],
) -> NativeDiffusionRecipe:
    """Bind Predict2 DiT, T5, 2.5 VAE, Video2World init, and Karras x0 AB2."""

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    codec = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    text_tokenizer_files = ("config.json", "spiece.model", "tokenizer.json")
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, denoiser_factory, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                build_t5_encoder_conditioner,
                {"weights": "text-encoder", "tokenizer": "text-tokenizer"},
                {"max_length": 512, "negative_fallback": "prompt", "zero_padding": True},
            ),
            ComponentSpec(codec, build_cosmos25_video_codec, {"weights": "vae"}),
            ComponentSpec(
                initializer,
                build_cosmos2_video2world_initializer,
                options={"sigma_max": 80.0, "spatial_compression": 8, "temporal_compression": 4},
            ),
            ComponentSpec(
                scheduler,
                build_karras_x0_ab2_scheduler,
                options={"sigma_min": 0.002, "sigma_max": 80.0, "rho": 7.0},
            ),
        ),
        execution=ExecutionSpec(
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "latent_encoder": codec,
                "scheduler": scheduler,
                "decoder": codec,
            },
            options={"guidance_mode": "positive"},
        ),
        checkpoints={
            "transformer": CheckpointSpec(
                repo_id=repo_id,
                revision=revision,
                files=("model-720p-16fps.pt",),
                allow_patterns=("model-720p-16fps.pt",),
            ),
            "vae": CheckpointSpec(
                repo_id=repo_id,
                revision=revision,
                files=("tokenizer/tokenizer.pth",),
                allow_patterns=("tokenizer/tokenizer.pth",),
            ),
            "text-encoder": CheckpointSpec(
                repo_id=T5_REPO_ID,
                revision=T5_REVISION,
                files=("pytorch_model.bin",),
                allow_patterns=("pytorch_model.bin", "config.json"),
            ),
            "text-tokenizer": CheckpointSpec(
                repo_id=T5_REPO_ID,
                revision=T5_REVISION,
                files=text_tokenizer_files,
                allow_patterns=text_tokenizer_files,
            ),
        },
        capabilities=frozenset(
            {"image-to-world", "video-to-world", "classifier-free-guidance"}
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": "cosmos-predict2-minimal-v1-lvg-dit",
            "native_inference": True,
            "output_layout": "BCTHW",
            "upstream_revision": "931c53d77e2604a02dbd588e01cfa311d25fd503",
        },
    )


def cosmos2_2b_video2world_recipe() -> NativeDiffusionRecipe:
    """Predict2 2B Video2World: T5 + Karras x0 AB2, ``standard`` CFG."""

    return _recipe(
        model_id=COSMOS2_2B_MODEL_ID,
        repo_id=COSMOS2_2B_REPO_ID,
        revision=COSMOS2_2B_REVISION,
        denoiser_factory=build_cosmos2_2b_denoiser,
        aliases=(
            "cosmos-predict2",
            "cosmos-predict-2",
            "cosmos2",
            COSMOS2_2B_REPO_ID,
        ),
    )


def cosmos2_14b_video2world_recipe() -> NativeDiffusionRecipe:
    """Predict2 14B Video2World; same roles and CFG as the 2B recipe."""

    return _recipe(
        model_id=COSMOS2_14B_MODEL_ID,
        repo_id=COSMOS2_14B_REPO_ID,
        revision=COSMOS2_14B_REVISION,
        denoiser_factory=build_cosmos2_14b_denoiser,
        aliases=("cosmos-predict2-14b", COSMOS2_14B_REPO_ID),
    )


__all__ = ["cosmos2_2b_video2world_recipe", "cosmos2_14b_video2world_recipe"]
