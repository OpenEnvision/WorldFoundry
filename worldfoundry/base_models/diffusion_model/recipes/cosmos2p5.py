"""Native Cosmos Predict 2.5 and Transfer 2.5 recipes.

Each public function returns a :class:`NativeDiffusionRecipe`.  Component
factories are imported lazily so optional Cosmos 2.5 deps stay unloaded
until a recipe is actually built.  Recipes do not construct runners.

Family split
------------
- Predict 2.5 2B / 14B: DiT, Cosmos-Reason1 conditioner, video VAE used
  as both latent initializer and decoder, Wan UniPC (shift 5.0;
  the 2B post-trained checkpoint uses Karras sigmas).  Strategy ``standard`` with
  ``guidance_mode="positive"`` (CFG).  Text / image / video-to-world.
- Transfer 2.5 2B: VACE-style controlled DiT on the same four roles;
  edge-to-world / controlled-video capabilities.

The 2B recipe also pins a pre-trained transformer snapshot as an extra
checkpoint role; it is not a second denoiser expert.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from .spec import NativeDiffusionRecipe

COSMOS25_2B_MODEL_ID = "cosmos-predict2.5-2b"
COSMOS25_14B_MODEL_ID = "cosmos-predict2.5-14b"
COSMOS25_TRANSFER_2B_MODEL_ID = "cosmos-transfer2.5-2b-controlled-video"
COSMOS25_2B_REPO_ID = "nvidia/Cosmos-Predict2.5-2B"
COSMOS25_14B_REPO_ID = "nvidia/Cosmos-Predict2.5-14B"
COSMOS25_TRANSFER_2B_REPO_ID = "nvidia/Cosmos-Transfer2.5-2B"
COSMOS25_REASON_REPO_ID = "nvidia/Cosmos-Reason1-7B"
COSMOS25_2B_REVISION = "f176dc95b4a70f53ce01c4b302851595e7322b00"
COSMOS25_2B_PRETRAINED_REVISION = "15a82a2ec231bc318692aa0456a36537c806e7d4"
COSMOS25_14B_REVISION = "71ebf3e8af30ecfe440bf0481115975fcc052b46"
COSMOS25_REASON_REVISION = "375e24000b24baed78f4618d3dd779e47cd96323"
COSMOS25_TRANSFER_2B_REVISION = "bd963eabcfc2d61dc4ea365cacf41d45ac480aa5"

_REASON_TOKENIZER_FILES = (
    "chat_template.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _build_cosmos25_2b_denoiser(context):
    from ..models.denoisers.cosmos2p5 import build_cosmos25_2b_denoiser

    return build_cosmos25_2b_denoiser(context)


def _build_cosmos25_14b_denoiser(context):
    from ..models.denoisers.cosmos2p5 import build_cosmos25_14b_denoiser

    return build_cosmos25_14b_denoiser(context)


def _build_cosmos25_transfer_2b_denoiser(context):
    from ..models.denoisers.cosmos2p5 import build_cosmos25_transfer_2b_denoiser

    return build_cosmos25_transfer_2b_denoiser(context)


def _build_cosmos25_prompt_conditioner(context):
    from ..models.encoders.cosmos2p5 import build_cosmos25_prompt_conditioner

    return build_cosmos25_prompt_conditioner(context)


def _build_cosmos25_video_codec(context):
    from ..models.autoencoders.cosmos2p5 import build_cosmos25_video_codec

    return build_cosmos25_video_codec(context)


def _build_cosmos25_scheduler(context):
    from ..schedulers import build_wan_flow_unipc_scheduler

    return build_wan_flow_unipc_scheduler(context)


def _recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    weight_file: str,
    denoiser_factory,
    aliases: tuple[str, ...],
    capabilities: frozenset[str] | None = None,
    architecture: str = "cosmos-predict2.5-minimal-v1-lvg-dit",
    pretrained_checkpoint: CheckpointSpec | None = None,
    minimum_frames: int = 1,
    use_karras_sigma: bool = False,
    karras_steps_are_intervals: bool = False,
    zero_pad_image: bool = True,
    conditional_frame_timestep: float = -1.0,
) -> NativeDiffusionRecipe:
    """Bind DiT, Reason1 conditioner, VAE-as-initializer, and Wan UniPC CFG."""

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    codec = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(weight_file,),
            allow_patterns=(weight_file,),
        ),
        "vae": CheckpointSpec(
            repo_id=COSMOS25_2B_REPO_ID,
            revision=COSMOS25_2B_REVISION,
            files=("tokenizer.pth",),
            allow_patterns=("tokenizer.pth",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=COSMOS25_REASON_REPO_ID,
            revision=COSMOS25_REASON_REVISION,
            files=("model.safetensors.index.json",),
            allow_patterns=("model.safetensors.index.json", "model*.safetensors"),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=COSMOS25_REASON_REPO_ID,
            revision=COSMOS25_REASON_REVISION,
            files=_REASON_TOKENIZER_FILES,
            allow_patterns=_REASON_TOKENIZER_FILES,
        ),
    }
    if pretrained_checkpoint is not None:
        checkpoints["transformer-pretrained"] = pretrained_checkpoint
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, denoiser_factory, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                _build_cosmos25_prompt_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(
                codec, _build_cosmos25_video_codec, {"weights": "vae"},
                options={
                    "minimum_frames": minimum_frames,
                    "zero_pad_image": zero_pad_image,
                    "conditional_frame_timestep": conditional_frame_timestep,
                },
            ),
            ComponentSpec(
                scheduler,
                _build_cosmos25_scheduler,
                options={
                    "shift": 5.0,
                    "use_karras_sigma": use_karras_sigma,
                    "karras_steps_are_intervals": karras_steps_are_intervals,
                },
            ),
        ),
        execution=ExecutionSpec(
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": codec,
                "scheduler": scheduler,
                "decoder": codec,
            },
            options={"guidance_mode": "positive"},
        ),
        checkpoints=checkpoints,
        capabilities=capabilities
        or frozenset(
            {
                "text-to-world",
                "image-to-world",
                "video-to-world",
                "classifier-free-guidance",
            }
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": architecture,
            "native_inference": True,
            "output_layout": "BCTHW",
            "upstream_revision": "a2c298b0a3df3778b973fe65e9e58877b292d8a7",
        },
    )


def cosmos25_2b_recipe() -> NativeDiffusionRecipe:
    """Predict 2.5 2B: official post-trained EDM-checkpoint inference setup."""

    return _recipe(
        model_id=COSMOS25_2B_MODEL_ID,
        repo_id=COSMOS25_2B_REPO_ID,
        revision=COSMOS25_2B_REVISION,
        weight_file="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        denoiser_factory=_build_cosmos25_2b_denoiser,
        aliases=("cosmos-predict2.5", "cosmos-predict2p5", COSMOS25_2B_REPO_ID),
        pretrained_checkpoint=CheckpointSpec(
            repo_id=COSMOS25_2B_REPO_ID,
            revision=COSMOS25_2B_PRETRAINED_REVISION,
            files=("base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",),
            allow_patterns=("base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",),
        ),
        use_karras_sigma=True,
        karras_steps_are_intervals=True,
        conditional_frame_timestep=0.1,
    )


def cosmos25_14b_recipe() -> NativeDiffusionRecipe:
    """Predict 2.5 14B; same four-role positive CFG stack as 2B."""

    return _recipe(
        model_id=COSMOS25_14B_MODEL_ID,
        repo_id=COSMOS25_14B_REPO_ID,
        revision=COSMOS25_14B_REVISION,
        weight_file="base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt",
        denoiser_factory=_build_cosmos25_14b_denoiser,
        aliases=(COSMOS25_14B_REPO_ID,),
    )


def cosmos25_transfer_2b_recipe() -> NativeDiffusionRecipe:
    """Transfer 2.5 2B edge-controlled video; same UniPC CFG, VACE DiT."""

    return _recipe(
        model_id=COSMOS25_TRANSFER_2B_MODEL_ID,
        repo_id=COSMOS25_TRANSFER_2B_REPO_ID,
        revision=COSMOS25_TRANSFER_2B_REVISION,
        weight_file="general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
        denoiser_factory=_build_cosmos25_transfer_2b_denoiser,
        aliases=(
            "cosmos-transfer-2.5",
            "cosmos-transfer2.5",
            "cosmos-transfer2p5",
            "cosmos-transfer-2.5-2b",
            COSMOS25_TRANSFER_2B_REPO_ID,
        ),
        capabilities=frozenset(
            {
                "controlled-video",
                "edge-to-world",
                "image-to-world",
                "classifier-free-guidance",
            }
        ),
        architecture="cosmos-transfer2.5-minimal-v4-lvg-vace-dit",
        minimum_frames=93,
        use_karras_sigma=True,
        zero_pad_image=False,
        conditional_frame_timestep=0.0,
    )


__all__ = ["cosmos25_2b_recipe", "cosmos25_14b_recipe", "cosmos25_transfer_2b_recipe"]
