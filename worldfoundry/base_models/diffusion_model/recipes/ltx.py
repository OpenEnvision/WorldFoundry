"""Native declarative recipes for the LTX-2 / LTX-Video audio-video family.

Each public function returns a :class:`NativeDiffusionRecipe`: component
factories, Hub-pinned :class:`CheckpointSpec` values, and an execution
strategy id.  Recipes do not construct runners.

Family split
------------
- LTX-2 / LTX-2.3 I2V and T2V: joint audio-video DiT, Gemma prompt
  conditioner, multistage latent initializer, two ``LTXFixedEulerScheduler``
  instances, a spatial latent upsampler, and a tiled media decoder.
  Strategy is ``joint-multistage`` with ``stage_steps=(8, 3)``.
- LTX-Video 0.9.8 I2V: video-only DiT, T5 conditioner, two-pass Euler
  (``stage_steps=(7, 3)``), still ``joint-multistage``.

Both paths denoise at a reduced spatial scale, upsample latents, then
refine.  Distilled checkpoints; no dual-expert CFG routing.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.ltx import build_ltx_media_decoder, build_ltx_video_media_decoder
from ..models.denoisers.ltx import build_ltx_joint_denoiser, build_ltx_video_denoiser
from ..models.encoders.ltx import build_ltx_prompt_conditioner
from ..models.encoders.t5 import build_t5_encoder_conditioner
from ..models.initializers.ltx import (
    build_ltx_multistage_latent_initializer,
    build_ltx_video_latent_initializer,
)
from ..models.upsamplers.ltx import build_ltx_spatial_latent_processor
from ..schedulers import build_ltx_fixed_euler_scheduler
from .spec import NativeDiffusionRecipe

LTX2_I2V_MODEL_ID = "ltx-2-i2v"
LTX2_T2V_MODEL_ID = "ltx-2-t2v"
LTX23_I2V_MODEL_ID = "ltx-2.3-i2v"
LTX23_T2V_MODEL_ID = "ltx-2.3-t2v"
LTX_VIDEO_I2V_MODEL_ID = "ltx-video-i2v"
LTX2_REPO_ID = "Lightricks/LTX-2"
LTX23_REPO_ID = "Lightricks/LTX-2.3"
LTX2_REVISION = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
LTX23_REVISION = "76730e634e70a28f4e8d51f5e29c08e40e2d8e74"
LTX_VIDEO_REPO_ID = "Lightricks/LTX-Video"
LTX_VIDEO_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"

LTX_STAGE_1_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX_STAGE_2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
LTX_VIDEO_STAGE_1_SIGMAS = (1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.0)
LTX_VIDEO_STAGE_2_SIGMAS = (0.9094, 0.725, 0.4219, 0.0)
LTX_GEMMA_FILES = tuple(f"text_encoder/model-{index:05d}-of-00011.safetensors" for index in range(1, 12))
LTX_TOKENIZER_FILES = (
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/preprocessor_config.json",
    "tokenizer/processor_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
)
LTX_VIDEO_T5_FILES = tuple(f"text_encoder/model-{index:05d}-of-00004.safetensors" for index in range(1, 5))
LTX_VIDEO_TOKENIZER_FILES = (
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
)


def _ltx_recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    checkpoint_file: str,
    upsampler_file: str,
    aliases: tuple[str, ...],
    generation_type: str = "i2v",
) -> NativeDiffusionRecipe:
    """Assemble one LTX-2 / 2.3 recipe from a generation type and Hub pins.

    Binds joint denoiser, Gemma conditioner, multistage initializer, two
    fixed-sigma Euler schedulers, spatial upsampler, and tiled decoder.
    Selects ``joint-multistage``.  ``generation_type`` is ``i2v`` or ``t2v``.
    """

    try:
        generation_capability = {
            "i2v": "image-to-video",
            "t2v": "text-to-video",
        }[generation_type]
    except KeyError as error:
        raise ValueError(f"unsupported LTX generation type: {generation_type!r}") from error
    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler_one = ComponentKey(ComponentKind.SCHEDULER, "stage-1")
    scheduler_two = ComponentKey(ComponentKind.SCHEDULER, "stage-2")
    processor = ComponentKey(ComponentKind.LATENT_PROCESSOR)
    decoder = ComponentKey(ComponentKind.DECODER)
    checkpoints = {
        "model": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(checkpoint_file,),
        ),
        "upsampler": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(upsampler_file,),
        ),
        "gemma": CheckpointSpec(
            repo_id=LTX2_REPO_ID,
            revision=LTX2_REVISION,
            files=LTX_GEMMA_FILES,
            allow_patterns=("text_encoder/*",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=LTX2_REPO_ID,
            revision=LTX2_REVISION,
            files=LTX_TOKENIZER_FILES,
            allow_patterns=("tokenizer/*",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, build_ltx_joint_denoiser, {"weights": "model"}),
            ComponentSpec(
                conditioner,
                build_ltx_prompt_conditioner,
                {"weights": "model", "gemma": "gemma", "tokenizer": "tokenizer"},
                {"max_length": 1024},
            ),
            ComponentSpec(
                initializer,
                build_ltx_multistage_latent_initializer,
                {"weights": "model"},
            ),
            ComponentSpec(
                scheduler_one,
                build_ltx_fixed_euler_scheduler,
                options={"sigmas": LTX_STAGE_1_SIGMAS},
            ),
            ComponentSpec(
                scheduler_two,
                build_ltx_fixed_euler_scheduler,
                options={"sigmas": LTX_STAGE_2_SIGMAS},
            ),
            ComponentSpec(
                processor,
                build_ltx_spatial_latent_processor,
                {"weights": "upsampler", "statistics": "model"},
            ),
            ComponentSpec(
                decoder,
                build_ltx_media_decoder,
                {"weights": "model"},
                {
                    "tiled": True,
                    "spatial_tile_size": 768,
                    "spatial_overlap": 64,
                    "temporal_tile_size": 80,
                    "temporal_overlap": 24,
                },
            ),
        ),
        execution=ExecutionSpec(
            strategy="joint-multistage",
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "scheduler-1": scheduler_one,
                "scheduler-2": scheduler_two,
                "processor": processor,
                "decoder": decoder,
            },
            options={"stage_steps": (8, 3)},
        ),
        checkpoints=checkpoints,
        capabilities=frozenset({generation_capability, "joint-audio-video", "two-stage-refinement"}),
        options={
            "latent_channels": 128,
            "spatial_compression": 32,
            "temporal_compression": 8,
        },
        metadata={
            "architecture": "ltx-2",
            "native_inference": True,
            "output_layout": "FHWC",
            "upstream_revision": revision,
        },
    )


def ltx2_i2v_recipe() -> NativeDiffusionRecipe:
    """LTX-2 19B distilled I2V on ``joint-multistage`` (8 then 3 Euler steps)."""

    return _ltx_recipe(
        model_id=LTX2_I2V_MODEL_ID,
        repo_id=LTX2_REPO_ID,
        revision=LTX2_REVISION,
        checkpoint_file="ltx-2-19b-distilled.safetensors",
        upsampler_file="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        aliases=("ltx2-i2v",),
    )


def ltx2_t2v_recipe() -> NativeDiffusionRecipe:
    """LTX-2 19B distilled T2V with the released two-stage schedule."""
    return _ltx_recipe(
        model_id=LTX2_T2V_MODEL_ID,
        repo_id=LTX2_REPO_ID,
        revision=LTX2_REVISION,
        checkpoint_file="ltx-2-19b-distilled.safetensors",
        upsampler_file="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        aliases=("ltx2-t2v",),
        generation_type="t2v",
    )


def ltx23_i2v_recipe() -> NativeDiffusionRecipe:
    """LTX-2.3 22B distilled I2V; same roles and two-stage Euler as LTX-2."""

    return _ltx_recipe(
        model_id=LTX23_I2V_MODEL_ID,
        repo_id=LTX23_REPO_ID,
        revision=LTX23_REVISION,
        checkpoint_file="ltx-2.3-22b-distilled-1.1.safetensors",
        upsampler_file="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        aliases=("ltx2.3-i2v", "ltx2_3_i2v"),
    )


def ltx23_t2v_recipe() -> NativeDiffusionRecipe:
    """LTX-2.3 22B distilled T2V; same joint stack as I2V, ``t2v`` capability."""

    return _ltx_recipe(
        model_id=LTX23_T2V_MODEL_ID,
        repo_id=LTX23_REPO_ID,
        revision=LTX23_REVISION,
        checkpoint_file="ltx-2.3-22b-distilled-1.1.safetensors",
        upsampler_file="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        aliases=("ltx2.3-t2v", "ltx2_3_t2v"),
        generation_type="t2v",
    )


def ltx_video_i2v_recipe() -> NativeDiffusionRecipe:
    """LTX-Video 0.9.8 distilled I2V on ``joint-multistage`` (7 then 3 steps).

    Binds the video-only DiT, T5 conditioner, video initializer (first-stage
    scale 2/3), two fixed-sigma Euler schedulers, spatial upsampler with
    AdaIN, and a tiled video decoder with tone-map compression.  No joint
    audio path; refinement is still two-stage.
    """

    denoiser = ComponentKey(ComponentKind.DENOISER)
    conditioner = ComponentKey(ComponentKind.CONDITIONER)
    initializer = ComponentKey(ComponentKind.LATENT_INITIALIZER)
    scheduler_one = ComponentKey(ComponentKind.SCHEDULER, "stage-1")
    scheduler_two = ComponentKey(ComponentKind.SCHEDULER, "stage-2")
    processor = ComponentKey(ComponentKind.LATENT_PROCESSOR)
    decoder = ComponentKey(ComponentKind.DECODER)
    return NativeDiffusionRecipe(
        model_id=LTX_VIDEO_I2V_MODEL_ID,
        aliases=("ltx-video",),
        components=(
            ComponentSpec(denoiser, build_ltx_video_denoiser, {"weights": "model"}),
            ComponentSpec(
                conditioner,
                build_t5_encoder_conditioner,
                {"weights": "text_encoder", "tokenizer": "tokenizer"},
                {
                    "max_length": 256,
                    "context_key": "video_context",
                    "negative_fallback": "omit",
                    "zero_padding": False,
                },
            ),
            ComponentSpec(
                initializer,
                build_ltx_video_latent_initializer,
                {"weights": "model"},
                {"first_stage_scale": 2.0 / 3.0},
            ),
            ComponentSpec(
                scheduler_one,
                build_ltx_fixed_euler_scheduler,
                options={"sigmas": LTX_VIDEO_STAGE_1_SIGMAS},
            ),
            ComponentSpec(
                scheduler_two,
                build_ltx_fixed_euler_scheduler,
                options={"sigmas": LTX_VIDEO_STAGE_2_SIGMAS},
            ),
            ComponentSpec(
                processor,
                build_ltx_spatial_latent_processor,
                {"weights": "upsampler", "statistics": "model"},
                {
                    "first_stage_scale": 2.0 / 3.0,
                    "adain_factor": 1.0,
                },
            ),
            ComponentSpec(
                decoder,
                build_ltx_video_media_decoder,
                {"weights": "model"},
                {
                    "tiled": True,
                    "spatial_tile_size": 768,
                    "spatial_overlap": 64,
                    "temporal_tile_size": 80,
                    "temporal_overlap": 24,
                    "first_stage_scale": 2.0 / 3.0,
                    "latent_upsample_factor": 2,
                    "tone_map_compression": 0.6,
                },
            ),
        ),
        execution=ExecutionSpec(
            strategy="joint-multistage",
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "scheduler-1": scheduler_one,
                "scheduler-2": scheduler_two,
                "processor": processor,
                "decoder": decoder,
            },
            options={"stage_steps": (7, 3)},
        ),
        checkpoints={
            "model": CheckpointSpec(
                repo_id=LTX_VIDEO_REPO_ID,
                revision=LTX_VIDEO_REVISION,
                files=("ltxv-13b-0.9.8-distilled.safetensors",),
            ),
            "upsampler": CheckpointSpec(
                repo_id=LTX_VIDEO_REPO_ID,
                revision=LTX_VIDEO_REVISION,
                files=("ltxv-spatial-upscaler-0.9.8.safetensors",),
            ),
            "text_encoder": CheckpointSpec(
                repo_id=LTX_VIDEO_REPO_ID,
                revision=LTX_VIDEO_REVISION,
                files=LTX_VIDEO_T5_FILES,
                allow_patterns=("text_encoder/*",),
            ),
            "tokenizer": CheckpointSpec(
                repo_id=LTX_VIDEO_REPO_ID,
                revision=LTX_VIDEO_REVISION,
                files=LTX_VIDEO_TOKENIZER_FILES,
                allow_patterns=("tokenizer/*",),
            ),
        },
        capabilities=frozenset({"image-to-video", "two-stage-refinement"}),
        options={
            "latent_channels": 128,
            "spatial_compression": 32,
            "temporal_compression": 8,
        },
        metadata={
            "architecture": "ltx-video-0.9.8",
            "native_inference": True,
            "output_layout": "FHWC",
            "upstream_revision": LTX_VIDEO_REVISION,
        },
    )


__all__ = [
    "LTX23_I2V_MODEL_ID",
    "LTX23_T2V_MODEL_ID",
    "LTX2_I2V_MODEL_ID",
    "LTX2_T2V_MODEL_ID",
    "LTX_VIDEO_I2V_MODEL_ID",
    "ltx23_i2v_recipe",
    "ltx23_t2v_recipe",
    "ltx2_i2v_recipe",
    "ltx2_t2v_recipe",
    "ltx_video_i2v_recipe",
]
