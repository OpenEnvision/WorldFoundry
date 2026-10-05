"""Independent native recipes for the Echo-Memory (Wan 2.1) family.

Each public function returns one or more :class:`NativeDiffusionRecipe`
values.  Recipes do not construct runners.

Every variant binds the Echo DiT, Wan UMT5 text conditioner (input
passthrough), Echo latent initializer, Wan UniPC (shift from the Echo
recipe spec), and Wan 2.1 VAE decoder.  Strategy is ``frozen-context``:
context frames stay fixed while future latents are sampled.  CFG is on.
Per-id ``context_frames`` and checkpoint availability come from
:data:`ECHO_MEMORY_MODELS`; this module does not invent extra stages or
dual experts.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.wan import build_wan_video_decoder
from ..models.denoisers.echo_memory import build_echo_memory_denoiser
from ..models.denoisers.echo_memory_spec import ECHO_MEMORY_MODELS, EchoMemoryModelSpec
from ..models.encoders.wan import build_wan_text_conditioner
from ..models.initializers.echo_memory import build_echo_memory_latent_initializer
from ..schedulers import build_wan_flow_unipc_scheduler
from .spec import NativeDiffusionRecipe
from .wan import (
    WAN21_T2V_1P3B_REPO_ID,
    WAN21_T2V_1P3B_REVISION,
    WAN_TOKENIZER_FILES,
)


def echo_memory_recipe(spec: EchoMemoryModelSpec) -> NativeDiffusionRecipe:
    """Build one Echo id on Wan text/VAE + ``frozen-context`` UniPC.

    ``spec`` selects DiT weights, aliases, UniPC shift, and context
    frame count.  Bindings are the default five native roles.
    """

    checkpoints = {
        "dit": CheckpointSpec(
            repo_id="Echo-Team/Echo-Memory",
            revision="7645f33efbfd02c58dc471570d0c7bd6a50eec83",
            files=(spec.checkpoint_file,),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=WAN21_T2V_1P3B_REPO_ID,
            revision=WAN21_T2V_1P3B_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=WAN21_T2V_1P3B_REPO_ID,
            revision=WAN21_T2V_1P3B_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=WAN21_T2V_1P3B_REPO_ID,
            revision=WAN21_T2V_1P3B_REVISION,
            files=("Wan2.1_VAE.pth",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=spec.model_id,
        aliases=spec.aliases,
        components=(
            ComponentSpec(
                key=ComponentKey(ComponentKind.DENOISER),
                factory=build_echo_memory_denoiser,
                checkpoints={"weights": "dit"},
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.CONDITIONER),
                factory=build_wan_text_conditioner,
                checkpoints={
                    "weights": "text-encoder",
                    "tokenizer": "tokenizer",
                },
                options={"passthrough_inputs": True},
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.LATENT_INITIALIZER),
                factory=build_echo_memory_latent_initializer,
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.SCHEDULER),
                factory=build_wan_flow_unipc_scheduler,
                options={"shift": spec.recipe.default_sample_shift},
            ),
            ComponentSpec(
                key=ComponentKey(ComponentKind.DECODER),
                factory=build_wan_video_decoder,
                checkpoints={"weights": "vae"},
            ),
        ),
        execution=ExecutionSpec(strategy="frozen-context"),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "image-to-video",
                "action-conditioned-video",
                "memory-research",
                "classifier-free-guidance",
            }
        ),
        options={
            "latent_channels": 16,
            "spatial_compression": 8,
            "temporal_compression": 4,
            "context_frames": spec.recipe.context_frames,
        },
        metadata={
            "architecture": "wan2.1-echo-memory",
            "recipe_id": spec.recipe_id,
            "checkpoint_availability": spec.checkpoint_availability.value,
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


def echo_memory_recipes() -> tuple[NativeDiffusionRecipe, ...]:
    """Return every :data:`ECHO_MEMORY_MODELS` entry as its own recipe."""

    return tuple(echo_memory_recipe(spec) for spec in ECHO_MEMORY_MODELS.values())


__all__ = ["echo_memory_recipe", "echo_memory_recipes"]
