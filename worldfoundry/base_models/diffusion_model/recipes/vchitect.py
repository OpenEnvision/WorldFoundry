"""Native declarative recipe for Vchitect-2.0 2B text-to-video.

Returns a :class:`NativeDiffusionRecipe` only.  Does not construct a
runner.

Binds the Vchitect MMDiT denoiser, triple text-encoder conditioner
(CLIP-L / CLIP-G / T5 from the Hub snapshot), Vchitect latent
initializer, the released FlowMatch Euler timetable (shift 3.0), and an SD3 frame-wise
VAE decoder.  Execution retains the standard five-role binding with its
Vchitect-specific guidance strategy.
CFG follows the released cosine timestep curve; temporal compression is 1 (per-frame latents).  No dual
expert or multistage refinement.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.sd3 import build_sd3_frame_decoder
from ..models.denoisers.vchitect import build_vchitect_denoiser
from ..models.encoders.vchitect import build_vchitect_prompt_conditioner
from ..models.initializers.vchitect import build_vchitect_latent_initializer
from ..schedulers.vchitect import build_vchitect_flow_match_euler_scheduler
from .spec import NativeDiffusionRecipe

VCHITECT_MODEL_ID = "vchitect-2-t2v"
VCHITECT_REPO_ID = "Vchitect/Vchitect-2.0-2B"
VCHITECT_REVISION = "37936818734242d8e75685b9ee43d2d2e447f98c"


def vchitect_2_t2v_recipe() -> NativeDiffusionRecipe:
    """Vchitect-2 T2V: MMDiT + three text encoders + Euler flow-match, CFG."""

    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=VCHITECT_REPO_ID,
            revision=VCHITECT_REVISION,
            files=("transformer/diffusion_pytorch_model.safetensors",),
        ),
        "vae": CheckpointSpec(
            repo_id=VCHITECT_REPO_ID,
            revision=VCHITECT_REVISION,
            files=("vae/diffusion_pytorch_model.safetensors",),
        ),
        "resources": CheckpointSpec(
            repo_id=VCHITECT_REPO_ID,
            revision=VCHITECT_REVISION,
            files=(
                "model_index.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_3/tokenizer_config.json",
                "text_encoder/config.json",
                "text_encoder_2/config.json",
                "text_encoder_3/config.json",
            ),
            allow_patterns=("tokenizer*/**", "text_encoder*/**", "model_index.json"),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=VCHITECT_MODEL_ID,
        aliases=("vchitect", "vchitect-2", VCHITECT_REPO_ID),
        components=(
            ComponentSpec(
                ComponentKey(ComponentKind.DENOISER),
                build_vchitect_denoiser,
                {"weights": "transformer"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.CONDITIONER),
                build_vchitect_prompt_conditioner,
                {"resources": "resources"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.LATENT_INITIALIZER),
                build_vchitect_latent_initializer,
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.SCHEDULER),
                build_vchitect_flow_match_euler_scheduler,
                options={"shift": 3.0},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.DECODER),
                build_sd3_frame_decoder,
                {"weights": "vae"},
            ),
        ),
        execution=ExecutionSpec(strategy="vchitect-guidance"),
        checkpoints=checkpoints,
        capabilities=frozenset({"text-to-video", "classifier-free-guidance"}),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 1},
        metadata={
            "architecture": "vchitect-2-mmdit",
            "parameter_scale": "2B",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


__all__ = ["VCHITECT_MODEL_ID", "VCHITECT_REPO_ID", "VCHITECT_REVISION", "vchitect_2_t2v_recipe"]
