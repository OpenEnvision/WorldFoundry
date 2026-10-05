"""Native declarative recipes for HunyuanVideo and HunyuanVideo 1.5.

Each public function returns a :class:`NativeDiffusionRecipe`.  Recipes
do not construct runners.  Execution is the default ``standard``
strategy; I2V additionally binds the codec as ``latent_encoder``.

Family split
------------
- Original T2V / I2V: Hunyuan DiT, LLaVA-Llama + CLIP-L conditioner,
  flow-match Euler (shift 7.0), original 16-channel VAE.  Embedded
  (non-CFG) guidance.  I2V freezes the first frame.
- HunyuanVideo 1.5 T2V / I2V: 32-channel VAE, LLM + ByT5 + Glyph
  conditioner (FLUX Redux vision on I2V), flow-match Euler (shift 9.0
  T2V / 7.0 I2V), glyph-conditioning.  Default quality pins are the
  official non-distilled 720p weights.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.hunyuan_video import (
    build_hunyuan_video15_codec,
    build_hunyuan_video_original_codec,
)
from ..models.denoisers.hunyuan_video import (
    build_hunyuan_video15_denoiser,
    build_hunyuan_video_denoiser,
    build_hunyuan_video_i2v_denoiser,
)
from ..models.encoders.hunyuan_video import (
    build_hunyuan_video15_prompt_conditioner,
    build_hunyuan_video_prompt_conditioner,
)
from ..models.initializers.hunyuan_video import build_hunyuan_video_latent_initializer
from ..schedulers import build_hunyuan_video_flow_match_scheduler
from .spec import NativeDiffusionRecipe

HUNYUAN_VIDEO_T2V_MODEL_ID = "hunyuanvideo-t2v"
HUNYUAN_VIDEO_I2V_MODEL_ID = "hunyuanvideo-i2v"
HUNYUAN_VIDEO15_T2V_MODEL_ID = "hunyuanvideo-1.5-t2v"
HUNYUAN_VIDEO15_I2V_MODEL_ID = "hunyuanvideo-1.5-i2v"


def _keys():
    return (
        ComponentKey(ComponentKind.DENOISER),
        ComponentKey(ComponentKind.CONDITIONER),
        ComponentKey(ComponentKind.LATENT_INITIALIZER),
        ComponentKey(ComponentKind.SCHEDULER),
        ComponentKey(ComponentKind.LATENT_ENCODER, "codec"),
    )


def _execution(*, image_to_video: bool) -> ExecutionSpec:
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    bindings = {
        "denoiser": denoiser,
        "conditioner": conditioner,
        "latent_initializer": initializer,
        "scheduler": scheduler,
        "decoder": codec,
    }
    if image_to_video:
        bindings["latent_encoder"] = codec
    return ExecutionSpec(bindings=bindings)


def _original_primary_resources(*, image_to_video: bool) -> CheckpointSpec:
    # Tencent's HunyuanVideo model repositories contain the DiT and VAE, but
    # not the LLaVA/Llama text encoder required by the released inference
    # code. Keep that dependency as its own checkpoint role so a local Tencent
    # snapshot is not incorrectly expected to contain files from XTuner.
    return CheckpointSpec(
        repo_id="xtuner/llava-llama-3-8b-v1_1-transformers",
        files=("config.json",),
        allow_patterns=("*",),
        metadata={"resource_role": "hunyuanvideo-primary-text-encoder", "image_to_video": image_to_video},
    )


def _original_clip_resources() -> CheckpointSpec:
    # The pooled CLIP-L conditioning model is likewise a separate upstream
    # repository. Declaring it independently also lets T2V and I2V share one
    # verified local copy instead of duplicating 1.7 GiB of weights.
    return CheckpointSpec(
        repo_id="openai/clip-vit-large-patch14",
        files=("config.json",),
        allow_patterns=("*",),
        metadata={"resource_role": "hunyuanvideo-clip-text-encoder"},
    )


def hunyuan_video_t2v_recipe() -> NativeDiffusionRecipe:
    """Original HunyuanVideo T2V: DiT + dual text encoders + flow-match Euler.

    Binds ``build_hunyuan_video_denoiser``, LLaVA-Llama/CLIP conditioner,
    noise initializer, flow-match scheduler (shift 7.0), and the original
    VAE codec as decoder only.  Strategy ``standard``; embedded guidance.
    """

    repo_id = "tencent/HunyuanVideo"
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=repo_id,
            files=("hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",),
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            files=("hunyuan-video-t2v-720p/vae/pytorch_model.pt",),
            allow_patterns=("hunyuan-video-t2v-720p/vae/*",),
        ),
        "primary": _original_primary_resources(image_to_video=False),
        "clip": _original_clip_resources(),
    }
    return NativeDiffusionRecipe(
        model_id=HUNYUAN_VIDEO_T2V_MODEL_ID,
        aliases=("hunyuanvideo", repo_id),
        components=(
            ComponentSpec(denoiser, build_hunyuan_video_denoiser, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                build_hunyuan_video_prompt_conditioner,
                {"primary": "primary", "clip": "clip"},
            ),
            ComponentSpec(initializer, build_hunyuan_video_latent_initializer),
            ComponentSpec(scheduler, build_hunyuan_video_flow_match_scheduler, options={"shift": 7.0}),
            ComponentSpec(
                codec,
                build_hunyuan_video_original_codec,
                {"weights": "vae"},
                {"config_path": "hunyuan-video-t2v-720p/vae/config.json"},
            ),
        ),
        execution=_execution(image_to_video=False),
        checkpoints=checkpoints,
        capabilities=frozenset({"text-to-video", "embedded-guidance"}),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={"architecture": "hunyuanvideo", "native_inference": True, "output_layout": "BCTHW"},
    )


def hunyuan_video_i2v_recipe() -> NativeDiffusionRecipe:
    """Original HunyuanVideo I2V with frozen first frame and encoded init.

    Same five roles as T2V, but the I2V DiT, ``image_to_video`` conditioner,
    and initializer with ``freeze_first_frame`` and the official stability
    mix for shift 7.  The VAE is also bound as ``latent_encoder``.  Strategy
    ``standard``; embedded guidance.
    """

    repo_id = "tencent/HunyuanVideo-I2V"
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=repo_id,
            files=("hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt",),
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            files=("hunyuan-video-i2v-720p/vae/pytorch_model.pt",),
            allow_patterns=("hunyuan-video-i2v-720p/vae/*",),
        ),
        "primary": _original_primary_resources(image_to_video=True),
        "clip": _original_clip_resources(),
    }
    return NativeDiffusionRecipe(
        model_id=HUNYUAN_VIDEO_I2V_MODEL_ID,
        components=(
            ComponentSpec(denoiser, build_hunyuan_video_i2v_denoiser, {"weights": "transformer"}),
            ComponentSpec(
                conditioner,
                build_hunyuan_video_prompt_conditioner,
                {"primary": "primary", "clip": "clip"},
                {"image_to_video": True},
            ),
            ComponentSpec(
                initializer,
                build_hunyuan_video_latent_initializer,
                options={"image_to_video": True, "freeze_first_frame": True, "stabilize_i2v": True},
            ),
            ComponentSpec(scheduler, build_hunyuan_video_flow_match_scheduler, options={"shift": 7.0}),
            ComponentSpec(
                codec,
                build_hunyuan_video_original_codec,
                {"weights": "vae"},
                {"config_path": "hunyuan-video-i2v-720p/vae/config.json"},
            ),
        ),
        execution=_execution(image_to_video=True),
        checkpoints=checkpoints,
        aliases=(repo_id,),
        capabilities=frozenset({"image-to-video", "embedded-guidance", "frozen-first-frame"}),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={"architecture": "hunyuanvideo-i2v", "native_inference": True, "output_layout": "BCTHW"},
    )


def _h15_recipe(*, image_to_video: bool) -> NativeDiffusionRecipe:
    """Assemble HunyuanVideo 1.5 T2V or I2V on ``standard`` (glyph + optional vision)."""

    repo_id = "tencent/HunyuanVideo-1.5"
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    model_id = HUNYUAN_VIDEO15_I2V_MODEL_ID if image_to_video else HUNYUAN_VIDEO15_T2V_MODEL_ID
    # The default quality demo uses the official non-distilled 720p models.
    # Distilled variants remain opt-in accelerators and must not silently
    # replace the full 50-step checkpoint used for parity validation.
    transformer_dir = "720p_i2v" if image_to_video else "720p_t2v"
    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=repo_id,
            files=(f"transformer/{transformer_dir}/diffusion_pytorch_model.safetensors",),
            allow_patterns=(f"transformer/{transformer_dir}/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            files=("vae/diffusion_pytorch_model.safetensors",),
            allow_patterns=("vae/*",),
        ),
        "resources": CheckpointSpec(
            repo_id=repo_id,
            files=(
                "config.json",
                "text_encoder/llm/config.json",
                "text_encoder/byt5-small/config.json",
                "text_encoder/byt5-small/model.safetensors",
                "text_encoder/Glyph-SDXL-v2/checkpoints/model.safetensors",
                "text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
                "text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
            ),
            allow_patterns=("text_encoder/**",),
        ),
    }
    if image_to_video:
        checkpoints["vision"] = CheckpointSpec(
            repo_id="black-forest-labs/FLUX.1-Redux-dev",
            files=(
                "image_encoder/config.json",
                "image_encoder/model.safetensors",
                "feature_extractor/preprocessor_config.json",
            ),
            allow_patterns=("image_encoder/**", "feature_extractor/**"),
        )
    conditioner_checkpoints = {"resources": "resources"}
    if image_to_video:
        conditioner_checkpoints["vision"] = "vision"
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=(("hunyuanvideo-1.5", repo_id) if not image_to_video else ()),
        components=(
            ComponentSpec(
                denoiser,
                build_hunyuan_video15_denoiser,
                {"weights": "transformer"},
                {
                    "config_path": f"transformer/{transformer_dir}/config.json",
                    "image_to_video": image_to_video,
                },
            ),
            ComponentSpec(
                conditioner,
                build_hunyuan_video15_prompt_conditioner,
                conditioner_checkpoints,
                {"image_to_video": image_to_video},
            ),
            ComponentSpec(
                initializer,
                build_hunyuan_video_latent_initializer,
                options={"image_to_video": image_to_video, "concat_condition": True},
            ),
            ComponentSpec(
                scheduler,
                build_hunyuan_video_flow_match_scheduler,
                options={"shift": 7.0 if image_to_video else 9.0},
            ),
            ComponentSpec(codec, build_hunyuan_video15_codec, {"weights": "vae"}),
        ),
        execution=_execution(image_to_video=image_to_video),
        checkpoints=checkpoints,
        capabilities=frozenset({"image-to-video" if image_to_video else "text-to-video", "glyph-conditioning"}),
        options={"latent_channels": 32, "spatial_compression": 16, "temporal_compression": 4},
        metadata={"architecture": "hunyuanvideo-1.5", "native_inference": True, "output_layout": "BCTHW"},
    )


def hunyuan_video15_t2v_recipe() -> NativeDiffusionRecipe:
    """HunyuanVideo 1.5 T2V (non-distilled 720p) with glyph conditioning.

    Binds the 1.5 DiT, LLM/ByT5/Glyph conditioner, concat-condition
    initializer, flow-match Euler (shift 9.0), and 32-channel VAE.
    Strategy ``standard``.
    """

    return _h15_recipe(image_to_video=False)


def hunyuan_video15_i2v_recipe() -> NativeDiffusionRecipe:
    """HunyuanVideo 1.5 I2V (non-distilled 720p) plus FLUX Redux vision.

    Same stack as 1.5 T2V with I2V DiT, vision checkpoint, encoder-bound
    VAE, and flow-match shift 7.0.  Strategy ``standard``.
    """

    return _h15_recipe(image_to_video=True)


def hunyuan_video_recipes() -> tuple[NativeDiffusionRecipe, ...]:
    """Return the four public HunyuanVideo recipes in registration order."""

    return (
        hunyuan_video_t2v_recipe(),
        hunyuan_video_i2v_recipe(),
        hunyuan_video15_t2v_recipe(),
        hunyuan_video15_i2v_recipe(),
    )


__all__ = [
    "HUNYUAN_VIDEO15_I2V_MODEL_ID",
    "HUNYUAN_VIDEO15_T2V_MODEL_ID",
    "HUNYUAN_VIDEO_I2V_MODEL_ID",
    "HUNYUAN_VIDEO_T2V_MODEL_ID",
    "hunyuan_video15_i2v_recipe",
    "hunyuan_video15_t2v_recipe",
    "hunyuan_video_i2v_recipe",
    "hunyuan_video_recipes",
    "hunyuan_video_t2v_recipe",
]
