"""Native Cosmos3 Nano and Super omni (audio-video) recipes.

Each public function returns a :class:`NativeDiffusionRecipe`.  Recipes
do not construct runners.

Both scales bind a joint audio-video DiT, tokenizer-backed prompt
conditioner, VAE latent initializer, Flow UniPC loaded from the Hub
scheduler config, and a media decoder that also consumes the sound
tokenizer.  Strategy is ``joint-multistage`` with a single stage
(``stage_steps=(35,)``) — one scheduler role (``scheduler-1``), no
spatial upsampler pass.

Capabilities include T2I / T2V / I2V / V2V, joint audio-video, and
action / dynamics heads.  Nano and Super differ only in Hub pin and
transformer shard count (7 vs 27).
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.cosmos3 import build_cosmos3_media_decoder
from ..models.denoisers.cosmos3 import build_cosmos3_joint_denoiser
from ..models.encoders.cosmos3 import build_cosmos3_prompt_conditioner
from ..models.initializers.cosmos3 import build_cosmos3_latent_initializer
from ..schedulers import build_cosmos3_flow_unipc_scheduler
from .spec import NativeDiffusionRecipe

COSMOS3_NANO_MODEL_ID = "cosmos3-nano"
COSMOS3_SUPER_MODEL_ID = "cosmos3-super"
COSMOS3_NANO_REPO_ID = "nvidia/Cosmos3-Nano"
COSMOS3_SUPER_REPO_ID = "nvidia/Cosmos3-Super"
COSMOS3_NANO_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
COSMOS3_SUPER_REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"

COSMOS3_TOKENIZER_FILES = (
    "text_tokenizer/added_tokens.json",
    "text_tokenizer/chat_template.jinja",
    "text_tokenizer/merges.txt",
    "text_tokenizer/special_tokens_map.json",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "text_tokenizer/vocab.json",
)


def _cosmos3_recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    transformer_shards: int,
    aliases: tuple[str, ...],
) -> NativeDiffusionRecipe:
    """Bind joint DiT, prompt conditioner, VAE init, UniPC, and AV decoder."""

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    decoder = ComponentKey(ComponentKind.DECODER)
    transformer_index = "transformer/diffusion_pytorch_model.safetensors.index.json"
    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(transformer_index,),
            allow_patterns=("transformer/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("vae/diffusion_pytorch_model.safetensors",),
            allow_patterns=("vae/*",),
        ),
        "sound": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("sound_tokenizer/diffusion_pytorch_model.safetensors",),
            allow_patterns=("sound_tokenizer/*",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=COSMOS3_TOKENIZER_FILES,
            allow_patterns=("text_tokenizer/*",),
        ),
        "scheduler": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("scheduler/scheduler_config.json",),
            allow_patterns=("scheduler/*",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, build_cosmos3_joint_denoiser, {"weights": "transformer"}),
            ComponentSpec(conditioner, build_cosmos3_prompt_conditioner, {"tokenizer": "tokenizer"}),
            ComponentSpec(initializer, build_cosmos3_latent_initializer, {"weights": "vae"}),
            ComponentSpec(scheduler, build_cosmos3_flow_unipc_scheduler, {"config": "scheduler"}),
            ComponentSpec(
                decoder,
                build_cosmos3_media_decoder,
                {"weights": "vae", "sound": "sound"},
                {"tiled": False},
            ),
        ),
        execution=ExecutionSpec(
            strategy="joint-multistage",
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "scheduler-1": scheduler,
                "decoder": decoder,
            },
            options={"stage_steps": (35,)},
        ),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "text-to-image",
                "text-to-video",
                "image-to-video",
                "video-to-video",
                "joint-audio-video",
                "action-policy",
                "forward-dynamics",
                "inverse-dynamics",
            }
        ),
        options={
            "latent_channels": 48,
            "spatial_compression": 16,
            "temporal_compression": 4,
        },
        metadata={
            "architecture": "cosmos3-omni",
            "native_inference": True,
            "output_layout": "BCTHW",
            "transformer_shards": transformer_shards,
            "upstream_revision": revision,
        },
    )


def cosmos3_nano_recipe() -> NativeDiffusionRecipe:
    """Cosmos3-Nano: joint DiT + UniPC, ``joint-multistage`` one 35-step stage."""

    return _cosmos3_recipe(
        model_id=COSMOS3_NANO_MODEL_ID,
        repo_id=COSMOS3_NANO_REPO_ID,
        revision=COSMOS3_NANO_REVISION,
        transformer_shards=7,
        aliases=("cosmos3",),
    )


def cosmos3_super_recipe() -> NativeDiffusionRecipe:
    """Cosmos3-Super; same joint-multistage roles as Nano, 27 transformer shards."""

    return _cosmos3_recipe(
        model_id=COSMOS3_SUPER_MODEL_ID,
        repo_id=COSMOS3_SUPER_REPO_ID,
        revision=COSMOS3_SUPER_REVISION,
        transformer_shards=27,
        aliases=(),
    )


__all__ = ["cosmos3_nano_recipe", "cosmos3_super_recipe"]
