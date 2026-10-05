"""Native declarative recipe for T2V-Turbo on VideoCrafter2 (LVDM).

Returns a :class:`NativeDiffusionRecipe` only.  Does not construct a
runner.

Binds the Turbo UNet (VideoCrafter2 base + LoRA), VC2 text conditioner,
LVDM T2V initializer, LCM scheduler, and LVDM video decoder.  Execution
is the default ``standard`` five-role binding.  Guidance is distilled
(LCM), not classifier-free dual-pass.  No dual expert or multistage
refinement.  Latents are 4-channel, 8x spatial, no temporal compression.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.lvdm import build_lvdm_video_decoder
from ..models.denoisers.t2v_turbo import build_t2v_turbo_denoiser
from ..models.encoders.t2v_turbo import build_t2v_turbo_conditioner
from ..models.initializers.lvdm import build_lvdm_t2v_latent_initializer
from ..schedulers import build_t2v_turbo_lcm_scheduler
from .spec import NativeDiffusionRecipe


T2V_TURBO_MODEL_ID = "t2v_turbo_t2v"
T2V_TURBO_BASE_REPO_ID = "VideoCrafter/VideoCrafter2"
T2V_TURBO_LORA_REPO_ID = "jiachenli-ucsb/T2V-Turbo-VC2"


def t2v_turbo_t2v_recipe() -> NativeDiffusionRecipe:
    """T2V-Turbo VC2: LVDM UNet + LoRA + LCM, ``standard`` distilled guidance."""

    checkpoints = {
        "base": CheckpointSpec(
            repo_id=T2V_TURBO_BASE_REPO_ID,
            files=("model.ckpt",),
        ),
        "lora": CheckpointSpec(
            repo_id=T2V_TURBO_LORA_REPO_ID,
            files=("unet_lora.pt",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=T2V_TURBO_MODEL_ID,
        aliases=(
            "t2v-turbo",
            T2V_TURBO_BASE_REPO_ID,
            T2V_TURBO_LORA_REPO_ID,
        ),
        components=(
            ComponentSpec(
                ComponentKey(ComponentKind.DENOISER),
                build_t2v_turbo_denoiser,
                {"base": "base", "lora": "lora"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.CONDITIONER),
                build_t2v_turbo_conditioner,
                {"base": "base"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.LATENT_INITIALIZER),
                build_lvdm_t2v_latent_initializer,
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.SCHEDULER),
                build_t2v_turbo_lcm_scheduler,
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.DECODER),
                build_lvdm_video_decoder,
                {"base": "base"},
            ),
        ),
        checkpoints=checkpoints,
        capabilities=frozenset({"text-to-video", "distilled-guidance", "lora"}),
        options={
            "latent_channels": 4,
            "spatial_compression": 8,
            "temporal_compression": 1,
        },
        metadata={
            "architecture": "lvdm-videocrafter2",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


__all__ = [
    "T2V_TURBO_BASE_REPO_ID",
    "T2V_TURBO_LORA_REPO_ID",
    "T2V_TURBO_MODEL_ID",
    "t2v_turbo_t2v_recipe",
]
