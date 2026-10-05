"""Native declarative recipes for released Gamma-World variants.

Each public function returns a :class:`NativeDiffusionRecipe`.  Recipes
do not construct runners.

Shared roles: Cosmos-multiview DiT, Cosmos-Reason1 conditioner
(full-concat embeddings, length 512), Gamma latent initializer, Wan
UniPC (shift 5.0), and the Gamma video codec as encoder + decoder.
CFG is on for all variants; default guidance/steps are recipe options.

Family split
------------
- Causal few-step: ``autoregressive-window``, distilled-x0, 4 fixed
  timesteps, guidance 1.0, context timestep 128.
- Causal: ``autoregressive-window``, flow prediction, 35 steps,
  guidance 5.0.
- Bidirectional: ``standard`` (full-sequence), 35 steps, guidance 5.0.
"""

from __future__ import annotations

from collections.abc import Callable

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.gamma_world import build_gamma_world_video_codec
from ..models.denoisers.gamma_world import (
    build_gamma_world_bidirectional_denoiser,
    build_gamma_world_causal_denoiser,
    build_gamma_world_causal_few_step_denoiser,
)
from ..models.encoders.gamma_world import build_gamma_world_conditioner
from ..models.initializers.gamma_world import build_gamma_world_latent_initializer
from ..schedulers import build_wan_flow_unipc_scheduler
from .spec import NativeDiffusionRecipe

GAMMA_REPO_ID = "chijw/Gamma-World"
GAMMA_REVISION = "8e15162f3e49d4952c0cd2ec3c99fd19855ec5d2"
REASON_REPO_ID = "nvidia/Cosmos-Reason1-7B"
REASON_REVISION = "375e24000b24baed78f4618d3dd779e47cd96323"

_REASON_TOKENIZER_FILES = (
    "chat_template.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _recipe(
    *,
    model_id: str,
    weight_file: str,
    denoiser_factory: Callable,
    aliases: tuple[str, ...],
    strategy: str,
    steps: int,
    guidance: float,
    execution_options: dict[str, object] | None = None,
    required_num_frames: int | None = None,
) -> NativeDiffusionRecipe:
    """Bind shared Gamma roles; ``strategy`` chooses window vs full-sequence."""

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    codec = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    bindings = {
        "denoiser": denoiser,
        "conditioner": conditioner,
        "latent_initializer": initializer,
        "latent_encoder": codec,
        "scheduler": scheduler,
        "decoder": codec,
    }
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, denoiser_factory, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                build_gamma_world_conditioner,
                {"weights": "text-encoder", "tokenizer": "text-tokenizer"},
                {"sequence_length": 512, "embedding_concat_strategy": "full_concat"},
            ),
            ComponentSpec(initializer, build_gamma_world_latent_initializer),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": 5.0},
            ),
            ComponentSpec(codec, build_gamma_world_video_codec, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(
            strategy=strategy,
            bindings=bindings,
            options=execution_options or {},
        ),
        checkpoints={
            "transformer": CheckpointSpec(
                repo_id=GAMMA_REPO_ID,
                revision=GAMMA_REVISION,
                files=(weight_file,),
                allow_patterns=(weight_file,),
            ),
            "vae": CheckpointSpec(
                repo_id=GAMMA_REPO_ID,
                revision=GAMMA_REVISION,
                files=("tokenizer.pth",),
                allow_patterns=("tokenizer.pth",),
            ),
            "text-encoder": CheckpointSpec(
                repo_id=REASON_REPO_ID,
                revision=REASON_REVISION,
                files=("model.safetensors.index.json",),
                allow_patterns=("model.safetensors.index.json", "model*.safetensors"),
            ),
            "text-tokenizer": CheckpointSpec(
                repo_id=REASON_REPO_ID,
                revision=REASON_REVISION,
                files=_REASON_TOKENIZER_FILES,
                allow_patterns=_REASON_TOKENIZER_FILES,
            ),
        },
        capabilities=frozenset(
            {
                "image-to-world",
                "multi-agent-world-model",
                "action-conditioned-video",
                "classifier-free-guidance",
            }
        ),
        options={
            "latent_channels": 16,
            "spatial_compression": 8,
            "temporal_compression": 4,
            "default_num_inference_steps": steps,
            "default_guidance_scale": guidance,
            **({"required_num_frames": required_num_frames} if required_num_frames is not None else {}),
        },
        metadata={
            "architecture": "gamma-world-cosmos-multiview-dit",
            "native_inference": True,
            "output_layout": "BCTH(VW)",
            "upstream_revision": "6a95de85c439d8ea73eae34c88fbfd4e89ea02e2",
        },
    )


def gamma_world_causal_few_step_recipe() -> NativeDiffusionRecipe:
    """Causal distilled 4-step window sampler (x0, fixed timesteps 1000–250)."""

    return _recipe(
        model_id="gamma-world-causal-few-step",
        weight_file="causal-few-step/model.safetensors",
        denoiser_factory=build_gamma_world_causal_few_step_denoiser,
        aliases=("gamma-world", "gammaworld"),
        strategy="autoregressive-window",
        steps=4,
        guidance=1.0,
        execution_options={
            "prediction_mode": "distilled-x0",
            "fixed_timesteps": (1000, 750, 500, 250),
            "context_timestep": 128,
        },
        required_num_frames=189,
    )


def gamma_world_causal_recipe() -> NativeDiffusionRecipe:
    """Causal flow 35-step ``autoregressive-window`` sampler, CFG 5.0."""

    return _recipe(
        model_id="gamma-world-causal",
        weight_file="causal/model.safetensors",
        denoiser_factory=build_gamma_world_causal_denoiser,
        aliases=(),
        strategy="autoregressive-window",
        steps=35,
        guidance=5.0,
        execution_options={"prediction_mode": "flow"},
        required_num_frames=189,
    )


def gamma_world_bidirectional_recipe() -> NativeDiffusionRecipe:
    """Bidirectional full-sequence ``standard`` sampler, 35 steps, CFG 5.0."""

    return _recipe(
        model_id="gamma-world-bidirectional",
        weight_file="bidirectional/model.safetensors",
        denoiser_factory=build_gamma_world_bidirectional_denoiser,
        aliases=(),
        strategy="standard",
        steps=35,
        guidance=5.0,
        required_num_frames=189,
    )


__all__ = [
    "gamma_world_bidirectional_recipe",
    "gamma_world_causal_few_step_recipe",
    "gamma_world_causal_recipe",
]
