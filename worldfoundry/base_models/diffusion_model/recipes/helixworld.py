"""HelixWorld Preview v1: controlled LTX AV components and clean-context chunks."""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.ltx import build_ltx_media_decoder
from ..models.denoisers.helixworld import build_helixworld_denoiser
from ..models.encoders.helixworld import build_helixworld_conditioner
from ..models.initializers.ltx import build_ltx_multistage_latent_initializer
from ..schedulers.clean_sample import build_clean_sample_noise_scheduler
from .spec import NativeDiffusionRecipe

MODEL_ID = "helixworld"
MODEL_REPO = "NoizAI/HelixWorld-preview"
MODEL_REVISION = "0f6aa9f329a118fcf0e16e0d315da026455ad8da"
SOURCE_REVISION = "3fa5f166834b08ea20ddc925c01f85d936ac03b5"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_REVISION = "68f7ee4fbd59087436ada77ed2d62f373fdd4482"


def helixworld_recipe() -> NativeDiffusionRecipe:
    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler = ComponentKey(ComponentKind.SCHEDULER, "stage-1")
    decoder = ComponentKey(ComponentKind.DECODER)
    return NativeDiffusionRecipe(
        model_id=MODEL_ID, aliases=("helixworld-preview", "helix-world", MODEL_REPO),
        checkpoints={
            "model": CheckpointSpec(repo_id=MODEL_REPO, revision=MODEL_REVISION, files=("weights/model.safetensors",)),
            "gemma": CheckpointSpec(
                repo_id=GEMMA_REPO, revision=GEMMA_REVISION,
                files=tuple(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
                allow_patterns=("*.json",),
            ),
            "tokenizer": CheckpointSpec(
                repo_id=GEMMA_REPO, revision=GEMMA_REVISION,
                files=("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                       "added_tokens.json", "processor_config.json", "preprocessor_config.json", "config.json"),
            ),
        },
        components=(
            ComponentSpec(denoiser, build_helixworld_denoiser, {"weights": "model"}),
            ComponentSpec(conditioner, build_helixworld_conditioner,
                          {"weights": "model", "gemma": "gemma", "tokenizer": "tokenizer"}, {"max_length": 1024}),
            ComponentSpec(initializer, build_ltx_multistage_latent_initializer, {"weights": "model"},
                          {"stage_divisors": (1,), "image_resize_mode": "center_crop"}),
            ComponentSpec(scheduler, build_clean_sample_noise_scheduler, options={"sigmas": (1.0, 0.9, 0.7, 0.4, 0.0)}),
            ComponentSpec(decoder, build_ltx_media_decoder, {"weights": "model"}, {"tiled": True}),
        ),
        execution=ExecutionSpec(
            strategy="joint-chunked",
            bindings={"denoiser": denoiser, "conditioner": conditioner, "latent_initializer": initializer,
                      "scheduler-1": scheduler, "decoder": decoder},
            options={"stage_steps": (4,), "frames_per_chunk": 4, "temporal_compression": 8,
                     "history_chunks": 3, "prefix_chunks": 1, "sampling_seed_offset": 1_000_003,
                     "token_conditions": {"video_control": "video"}},
        ),
        capabilities=frozenset({"image-to-video", "joint-audio-video", "action-conditioned", "camera-controlled"}),
        options={"latent_channels": 128, "spatial_compression": 32, "temporal_compression": 8},
        metadata={"architecture": "ltx-2.3", "native_inference": True, "output_layout": "FHWC",
                  "upstream_source_revision": SOURCE_REVISION, "checkpoint_revision": MODEL_REVISION},
    )
