"""Native declarative recipes for SkyReels V2 (Wan-DF) and V3 (R2V).

Each public function returns a :class:`NativeDiffusionRecipe`.  Recipes
do not construct runners.

Family split
------------
- V2 DF 1.3B 540P: Wan-style T2V DiT, UMT5 conditioner, T2V initializer,
  UniPC (shift 8.0), Wan 2.1 VAE decoder.  Default ``standard``
  bindings.  CFG plus FPS conditioning.
- V3 R2V 14B: SkyReels V3 DiT, Diffusers Wan text conditioner,
  reference-image latent initializer (up to 4 refs), UniPC (shift 5.0),
  Diffusers Wan codec as encoder + decoder.  Strategy
  ``dual-condition-guidance`` with a secondary image guidance scale
  (default 1.0) in addition to text CFG.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.wan import build_diffusers_wan_video_codec, build_wan_video_decoder
from ..models.denoisers.wan import build_skyreels_v2_denoiser, build_skyreels_v3_denoiser
from ..models.encoders.wan import build_diffusers_wan_text_conditioner, build_wan_text_conditioner
from ..models.initializers.wan import (
    build_wan_reference_latent_initializer,
    build_wan_t2v_latent_initializer,
)
from ..schedulers import build_wan_flow_unipc_scheduler
from .spec import NativeDiffusionRecipe
from .wan import WAN_TOKENIZER_FILES


SKYREELS_V2_MODEL_ID = "skyreels-v2"
SKYREELS_V2_REPO_ID = "Skywork/SkyReels-V2-DF-1.3B-540P"
SKYREELS_V2_REVISION = "1100111771ba2d921f10e76991f54db9f73edb3d"
SKYREELS_V3_MODEL_ID = "skyreels-v3"
SKYREELS_V3_REPO_ID = "Skywork/SkyReels-V3-R2V-14B"
SKYREELS_V3_REVISION = "8df04fa97e062099633b366d19a6b0b2dabd5a69"


def skyreels_v2_recipe() -> NativeDiffusionRecipe:
    """SkyReels-V2 DF 1.3B T2V: Wan stack + UniPC, ``standard`` CFG."""

    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=SKYREELS_V2_REPO_ID,
            revision=SKYREELS_V2_REVISION,
            files=("model.safetensors",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=SKYREELS_V2_REPO_ID,
            revision=SKYREELS_V2_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=SKYREELS_V2_REPO_ID,
            revision=SKYREELS_V2_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=SKYREELS_V2_REPO_ID,
            revision=SKYREELS_V2_REVISION,
            files=("Wan2.1_VAE.pth",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=SKYREELS_V2_MODEL_ID,
        aliases=("skyreels2", "skyreels-v2-t2v", SKYREELS_V2_REPO_ID),
        components=(
            ComponentSpec(
                ComponentKey(ComponentKind.DENOISER),
                build_skyreels_v2_denoiser,
                {"weights": "dit"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.CONDITIONER),
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"passthrough_inputs": True},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.LATENT_INITIALIZER),
                build_wan_t2v_latent_initializer,
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.SCHEDULER),
                build_wan_flow_unipc_scheduler,
                options={"shift": 8.0},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.DECODER),
                build_wan_video_decoder,
                {"weights": "vae"},
            ),
        ),
        checkpoints=checkpoints,
        capabilities=frozenset({"text-to-video", "classifier-free-guidance", "fps-conditioning"}),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": "skyreels-v2-wan-df-1.3b",
            "native_inference": True,
            "output_layout": "BCTHW",
            "checkpoint_graph_match": "830/830",
        },
    )


def skyreels_v3_recipe() -> NativeDiffusionRecipe:
    """SkyReels-V3 14B reference-to-video with dual-condition CFG.

    Binds V3 DiT, Diffusers Wan text conditioner, reference initializer,
    UniPC, and Wan codec (encoder + decoder).  Strategy
    ``dual-condition-guidance``: text CFG plus
    ``secondary_guidance_scale`` from ``image_guidance_scale``.
    """

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    codec = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=SKYREELS_V3_REPO_ID,
            revision=SKYREELS_V3_REVISION,
            files=("transformer/diffusion_pytorch_model.safetensors",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=SKYREELS_V3_REPO_ID,
            revision=SKYREELS_V3_REVISION,
            files=("text_encoder/model.safetensors.index.json",),
            allow_patterns=("text_encoder/*",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=SKYREELS_V3_REPO_ID,
            revision=SKYREELS_V3_REVISION,
            files=(
                "tokenizer/special_tokens_map.json",
                "tokenizer/spiece.model",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            ),
            allow_patterns=("tokenizer/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=SKYREELS_V3_REPO_ID,
            revision=SKYREELS_V3_REVISION,
            files=("vae/diffusion_pytorch_model.safetensors",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=SKYREELS_V3_MODEL_ID,
        aliases=("skyreels-v3-r2v", "skyreels-v3-reference-to-video", SKYREELS_V3_REPO_ID),
        components=(
            ComponentSpec(denoiser, build_skyreels_v3_denoiser, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_diffusers_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"tokenizer_subdir": "tokenizer"},
            ),
            ComponentSpec(
                initializer,
                build_wan_reference_latent_initializer,
                options={"max_reference_images": 4},
            ),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": 5.0},
            ),
            ComponentSpec(codec, build_diffusers_wan_video_codec, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(
            strategy="dual-condition-guidance",
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "scheduler": scheduler,
                "decoder": codec,
                "latent_encoder": codec,
            },
            options={
                "secondary_guidance_scale": 1.0,
                "secondary_guidance_input": "image_guidance_scale",
            },
        ),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {"reference-to-video", "classifier-free-guidance", "dual-condition-guidance"}
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": "skyreels-v3-wan-r2v-14b",
            "native_inference": True,
            "output_layout": "BCTHW",
            "checkpoint_graph_match": "1095/1095",
        },
    )


__all__ = [
    "SKYREELS_V2_MODEL_ID",
    "SKYREELS_V2_REPO_ID",
    "SKYREELS_V2_REVISION",
    "SKYREELS_V3_MODEL_ID",
    "SKYREELS_V3_REPO_ID",
    "SKYREELS_V3_REVISION",
    "skyreels_v2_recipe",
    "skyreels_v3_recipe",
]
