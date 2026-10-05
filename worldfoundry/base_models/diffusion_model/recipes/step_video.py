"""Native declarative recipe for official Step-Video-T2V.

Returns a :class:`NativeDiffusionRecipe` only.  Does not construct a
runner.

Binds the Step-Video DiT, Step-LLM + Hunyuan-CLIP conditioner, family
latent initializer, Step-Video flow scheduler (``time_shift=13.0``,
forward in time), and the v2 VAE decoder.  Execution is the default
``standard`` five-role binding.  CFG is on; no dual expert or extra
refinement stage.  Latents are 64-channel with 16x spatial compression
and a 3-latent / 17-pixel temporal chunk.
"""

from __future__ import annotations

from ..components import ComponentKey, ComponentKind, ComponentSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.step_video import build_step_video_decoder
from ..models.denoisers.step_video import build_step_video_denoiser
from ..models.encoders.step_video import build_step_video_prompt_conditioner
from ..models.initializers.step_video import build_step_video_latent_initializer
from ..schedulers import build_step_video_flow_scheduler
from .spec import NativeDiffusionRecipe


STEP_VIDEO_MODEL_ID = "step-video-t2v"
STEP_VIDEO_REPO_ID = "stepfun-ai/stepvideo-t2v"
STEP_VIDEO_REVISION = "7a2b639ca2685350e87a4df7e4026285309f7fb6"


def step_video_t2v_recipe() -> NativeDiffusionRecipe:
    """Step-Video-T2V: DiT + dual text encoders + flow scheduler, CFG."""

    checkpoints = {
        "transformer": CheckpointSpec(
            repo_id=STEP_VIDEO_REPO_ID,
            revision=STEP_VIDEO_REVISION,
            files=("transformer/diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("transformer/**",),
        ),
        "vae": CheckpointSpec(
            repo_id=STEP_VIDEO_REPO_ID,
            revision=STEP_VIDEO_REVISION,
            files=("vae/vae_v2.safetensors",),
            allow_patterns=("vae/vae_v2.safetensors",),
        ),
        "resources": CheckpointSpec(
            repo_id=STEP_VIDEO_REPO_ID,
            revision=STEP_VIDEO_REVISION,
            files=(
                "step_llm/config.json",
                "step_llm/model.safetensors.index.json",
                "step_llm/step1_chat_tokenizer.model",
                "hunyuan_clip/clip_text_encoder/config.json",
                "hunyuan_clip/clip_text_encoder/pytorch_model.bin",
                "hunyuan_clip/tokenizer/tokenizer_config.json",
            ),
            allow_patterns=("step_llm/**", "hunyuan_clip/**"),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=STEP_VIDEO_MODEL_ID,
        aliases=("stepvideo", "step-video", "stepvideo-t2v", STEP_VIDEO_REPO_ID),
        components=(
            ComponentSpec(
                ComponentKey(ComponentKind.DENOISER),
                build_step_video_denoiser,
                {"weights": "transformer"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.CONDITIONER),
                build_step_video_prompt_conditioner,
                {"resources": "resources"},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.LATENT_INITIALIZER),
                build_step_video_latent_initializer,
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.SCHEDULER),
                build_step_video_flow_scheduler,
                options={"time_shift": 13.0, "reverse": False},
            ),
            ComponentSpec(
                ComponentKey(ComponentKind.DECODER),
                build_step_video_decoder,
                {"weights": "vae"},
            ),
        ),
        checkpoints=checkpoints,
        capabilities=frozenset({"text-to-video", "classifier-free-guidance"}),
        options={
            "latent_channels": 64,
            "spatial_compression": 16,
            "latent_frames_per_video_chunk": 3,
            "video_frames_per_chunk": 17,
        },
        metadata={
            "architecture": "step-video-t2v",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


__all__ = [
    "STEP_VIDEO_MODEL_ID",
    "STEP_VIDEO_REPO_ID",
    "STEP_VIDEO_REVISION",
    "step_video_t2v_recipe",
]
