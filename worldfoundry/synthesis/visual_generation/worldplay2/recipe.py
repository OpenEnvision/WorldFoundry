"""WorldPlay2 releases share native Wan2.2 components and compressed-memory execution."""

from __future__ import annotations

from worldfoundry.base_models.diffusion_model.components import (
    ComponentKey,
    ComponentKind,
    ComponentSpec,
    ExecutionSpec,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import build_wan_video_decoder
from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    WAN22_I2V_A14B_REPO_ID,
    WAN22_I2V_A14B_REVISION,
    WAN_TOKENIZER_FILES,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import build_wan_flow_unipc_scheduler

from .conditioning import build_worldplay2_conditioner
from .denoiser import build_worldplay2_denoiser
from .initializer import build_worldplay2_latent_initializer
from .scheduler import FixedPDD4Scheduler

WORLDPLAY2_SOURCE_REVISION = "c5d83e32099116ff3a1437a8a05c764d579f704b"
WORLDPLAY2_RELEASES = {
    "few_step": ("worldplay2", "aejion/WorldPlay2-Fast", "1991d1cd9a3b116c2f6d9513a220c0ee5077ab79"),
    "ar": ("worldplay2-ar", "aejion/WorldPlay2-AR", "09ddd0161a4aaae8eef0eea65068be72dccc222f"),
    "bi": ("worldplay2-bi", "aejion/WorldPlay2-BI", "18f84ce25548383160edd0b5f66dc069ad05cd73"),
}


def build_worldplay2_pdd_scheduler(context):
    del context
    return FixedPDD4Scheduler()


def _recipe(mode):
    model_id, repo_id, revision = WORLDPLAY2_RELEASES[mode]
    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER)
    codec = ComponentKey(ComponentKind.LATENT_ENCODER, "codec")
    checkpoint_metadata = {"license": "CC-BY-NC-4.0", "upstream_source_revision": WORLDPLAY2_SOURCE_REVISION}
    base = {"repo_id": WAN22_I2V_A14B_REPO_ID, "revision": WAN22_I2V_A14B_REVISION}
    return NativeDiffusionRecipe(
        model_id=model_id, aliases=(("worldplay2-fast", repo_id) if mode == "few_step" else (repo_id,)),
        components=(
            ComponentSpec(denoiser, build_worldplay2_denoiser, {"high": "high", "low": "low"}),
            ComponentSpec(conditioner, build_worldplay2_conditioner, {"weights": "t5", "tokenizer": "tokenizer"}),
            ComponentSpec(initializer, build_worldplay2_latent_initializer),
            ComponentSpec(scheduler, build_worldplay2_pdd_scheduler if mode == "few_step" else build_wan_flow_unipc_scheduler,
                          options={} if mode == "few_step" else {"shift": 5.0}),
            ComponentSpec(codec, build_wan_video_decoder, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(strategy="compressed-memory", bindings={
            "denoiser": denoiser, "conditioner": conditioner, "latent_initializer": initializer,
            "scheduler": scheduler, "decoder": codec, "latent_encoder": codec,
        }),
        checkpoints={
            "high": CheckpointSpec(repo_id=repo_id, revision=revision,
                                   files=("high_noise_model/diffusion_pytorch_model.safetensors",), metadata=checkpoint_metadata),
            "low": CheckpointSpec(repo_id=repo_id, revision=revision,
                                  files=("low_noise_model/diffusion_pytorch_model.safetensors",), metadata=checkpoint_metadata),
            "t5": CheckpointSpec(**base, files=("models_t5_umt5-xxl-enc-bf16.pth",)),
            "tokenizer": CheckpointSpec(**base, files=WAN_TOKENIZER_FILES, allow_patterns=("google/umt5-xxl/*",)),
            "vae": CheckpointSpec(**base, files=("Wan2.1_VAE.pth",)),
        },
        options={"mode": mode, "latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        capabilities=frozenset({"image-to-video", "action-control", "prompt-scheduling", "compressed-memory", "chunked-video"}),
        metadata={"architecture": "wan2.2-i2v-a14b", "native_inference": True, "output_layout": "FHWC",
                  "upstream_source_revision": WORLDPLAY2_SOURCE_REVISION},
    )


def worldplay2_recipe():
    return _recipe("few_step")


def worldplay2_ar_recipe():
    return _recipe("ar")


def worldplay2_bi_recipe():
    return _recipe("bi")
