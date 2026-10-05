"""Native Cosmos Predict1 GEN3C (7B) camera-controlled world recipe.

Returns a :class:`NativeDiffusionRecipe` only.  Does not construct a
runner.

Binds the GEN3C FA-DiT denoiser, T5-11B conditioner (mask-ones, CFG
positive branch), Cosmos Tokenizer1 video codec as encoder and decoder,
GEN3C initializer with a MoGe depth model, and a Karras x0 Euler
scheduler (``sigma_min=0.0002``, ``sigma_max=80``, ``rho=7``).

Strategy is the default ``standard`` path with
``guidance_mode="positive"`` (CFG on the positive prediction, not a
dual-expert split).  Capabilities cover image-to-world, camera-controlled
video, and rendered 3D-cache conditioning.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.cosmos1 import build_cosmos1_video_codec
from ..models.denoisers.cosmos1 import build_cosmos1_gen3c_denoiser
from ..models.encoders.t5 import build_t5_encoder_conditioner
from ..models.initializers.cosmos1 import build_cosmos1_gen3c_initializer
from ..schedulers import build_karras_x0_euler_scheduler
from .spec import NativeDiffusionRecipe

GEN3C_MODEL_ID = "gen3c-cosmos1-7b"
GEN3C_REPO_ID = "nvidia/GEN3C-Cosmos-7B"
GEN3C_REVISION = "9bcfdb4f3924f41376daeadf6200826c12a3bf8e"
TOKENIZER_REPO_ID = "nvidia/Cosmos-Tokenize1-CV8x8x8-720p"
TOKENIZER_REVISION = "b6af495317c76f287a4131e9299936b1533f5f9f"
T5_REPO_ID = "google-t5/t5-11b"
T5_REVISION = "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3"
MOGE_REPO_ID = "Ruicheng/moge-vitl"
MOGE_REVISION = "979e84da9415762c30e6c0cf8dc0962896c793df"


def gen3c_recipe() -> NativeDiffusionRecipe:
    """GEN3C Cosmos-1 7B: T5 + Karras x0 Euler + depth-conditioned init.

    Strategy ``standard``; ``guidance_mode="positive"`` for CFG.
    """

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    codec = ComponentKey(ComponentKind.LATENT_ENCODER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    tokenizer_files = ("encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt")
    text_tokenizer_files = ("config.json", "spiece.model", "tokenizer.json")
    return NativeDiffusionRecipe(
        model_id=GEN3C_MODEL_ID,
        aliases=("gen3c", "cosmos1-gen3c", "cosmos-predict1-gen3c", GEN3C_REPO_ID),
        components=(
            ComponentSpec(denoiser, build_cosmos1_gen3c_denoiser, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                build_t5_encoder_conditioner,
                {"weights": "text-encoder", "tokenizer": "text-tokenizer"},
                {
                    "max_length": 512,
                    "mask_mode": "ones",
                    "negative_fallback": "prompt",
                    "zero_padding": True,
                },
            ),
            ComponentSpec(codec, build_cosmos1_video_codec, {"weights": "tokenizer"}),
            ComponentSpec(initializer, build_cosmos1_gen3c_initializer, {"depth": "depth-model"}),
            ComponentSpec(
                scheduler,
                build_karras_x0_euler_scheduler,
                options={"sigma_min": 0.0002, "sigma_max": 80.0, "rho": 7.0},
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
                repo_id=GEN3C_REPO_ID,
                revision=GEN3C_REVISION,
                files=("model.pt",),
                allow_patterns=("model.pt",),
            ),
            "tokenizer": CheckpointSpec(
                repo_id=TOKENIZER_REPO_ID,
                revision=TOKENIZER_REVISION,
                files=tokenizer_files,
                allow_patterns=tokenizer_files,
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
            "depth-model": CheckpointSpec(
                repo_id=MOGE_REPO_ID,
                revision=MOGE_REVISION,
                files=("model.pt",),
                allow_patterns=("model.pt",),
            ),
        },
        capabilities=frozenset(
            {
                "image-to-world",
                "camera-controlled-video",
                "rendered-3d-cache-conditioning",
                "classifier-free-guidance",
            }
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 8},
        metadata={
            "architecture": "cosmos-predict1-gen3c-faditv2-7b",
            "native_inference": True,
            "output_layout": "BCTHW",
            "upstream_revision": "bc6d6848381df59755d30c8e6c6f4bdcf47109fb",
        },
    )


__all__ = ["gen3c_recipe"]
