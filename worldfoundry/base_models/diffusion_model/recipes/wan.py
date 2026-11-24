"""Native declarative recipes for official Wan 2.1 / 2.2 video variants.

Each public function returns a :class:`NativeDiffusionRecipe`: component
factories, Hub-pinned :class:`CheckpointSpec` values, and an execution
strategy id.  Recipes do not construct runners.

Family split
------------
- Wan 2.1 T2V / I2V: single DiT, ``standard`` strategy, UniPC flow scheduler.
- Wan 2.1 VACE: extra video-control conditioner + VACE initializer.
- Wan 2.2 TI2V-5B: text+image initializer, still ``standard``.
- Wan 2.2 A14B T2V / I2V: dual low/high-noise experts and
  ``wan22-dual-expert-guidance`` (timestep-routed CFG scales).
- Self-Forcing and Causal-Forcing (Wan 2.1 T2V 1.3B DMD): causal
  single-expert generators on ``autoregressive-window`` with distilled x0
  prediction; Causal-Forcing adds frame-wise blocks, an image-seeded first
  frame and the Causal Forcing++ first-block step list.

Official model sources are pinned to immutable Hub revisions.
"""

from __future__ import annotations

from collections.abc import Callable

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.wan import (
    build_diffusers_wan_video_codec,
    build_wan_video_decoder,
    build_wan_video_vae38_decoder,
)
from ..models.denoisers.wan import (
    build_fastvideo_causal_wan22_denoiser,
    build_self_forcing_wan21_denoiser,
    build_wan21_i2v_14b_denoiser,
    build_wan21_t2v_1p3b_denoiser,
    build_wan21_t2v_14b_denoiser,
    build_wan22_i2v_a14b_dual_denoiser,
    build_wan22_t2v_a14b_dual_denoiser,
    build_wan22_ti2v_5b_denoiser,
)
from ..models.denoisers.wan_vace import build_wan21_vace_14b_denoiser
from ..models.encoders.wan import (
    build_diffusers_wan_text_conditioner,
    build_wan_image_text_conditioner,
    build_wan_text_conditioner,
)
from ..models.initializers.wan import (
    build_wan_causal_forcing_latent_initializer,
    build_wan_causal_i2v_latent_initializer,
    build_wan_i2v_latent_initializer,
    build_wan_t2v_latent_initializer,
    build_wan_ti2v_latent_initializer,
    build_wan_vace_latent_initializer,
)
from ..schedulers import (
    build_fastvideo_causal_wan_self_forcing_scheduler,
    build_wan_flow_match_euler_scheduler,
    build_wan_flow_unipc_scheduler,
)
from .spec import NativeDiffusionRecipe

WAN21_T2V_1P3B_MODEL_ID = "wan2.1-t2v-1.3b"
WAN21_T2V_14B_MODEL_ID = "wan2.1-t2v-14b"
WAN21_I2V_14B_480P_MODEL_ID = "wan2.1-i2v-14b-480p"
WAN21_I2V_14B_720P_MODEL_ID = "wan2.1-i2v-14b-720p"
WAN22_TI2V_5B_MODEL_ID = "wan2.2-ti2v-5b"
WAN22_T2V_A14B_MODEL_ID = "wan2.2-t2v-a14b"
WAN22_I2V_A14B_MODEL_ID = "wan2.2-i2v-a14b"
WAN21_VACE_14B_MODEL_ID = "wan2.1-vace"
FASTVIDEO_CAUSAL_WAN22_MODEL_ID = "fastvideo-causal-wan2.2-i2v-14b"
SELF_FORCING_MODEL_ID = "self-forcing"
CAUSAL_FORCING_CHUNKWISE_MODEL_ID = "causal-forcing-chunkwise"
CAUSAL_FORCING_FRAMEWISE_MODEL_ID = "causal-forcing-framewise"
CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID = "causal-forcing-framewise-1step"
CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID = "causal-forcing-framewise-2step"

WAN21_T2V_1P3B_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
WAN21_T2V_14B_REPO_ID = "Wan-AI/Wan2.1-T2V-14B"
WAN21_I2V_14B_480P_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN21_I2V_14B_720P_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-720P"
WAN22_TI2V_5B_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_T2V_A14B_REPO_ID = "Wan-AI/Wan2.2-T2V-A14B"
WAN22_I2V_A14B_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B"
WAN21_VACE_14B_REPO_ID = "Wan-AI/Wan2.1-VACE-14B"
FASTVIDEO_CAUSAL_WAN22_REPO_ID = "FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers"
SELF_FORCING_REPO_ID = "gdhe17/Self-Forcing"
CAUSAL_FORCING_REPO_ID = "zhuhz22/Causal-Forcing"

WAN21_T2V_1P3B_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAN21_T2V_14B_REVISION = "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
WAN21_I2V_14B_480P_REVISION = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
WAN21_I2V_14B_720P_REVISION = "8823af45fcc58a8aa999a54b04be9abc7d2aac98"
WAN22_TI2V_5B_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
WAN22_T2V_A14B_REVISION = "c8c270b13ee05bfa474194ac9fb07a5868a97cea"
WAN22_I2V_A14B_REVISION = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
WAN21_VACE_14B_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"
FASTVIDEO_CAUSAL_WAN22_REVISION = "0977572ccf137da5e577d62e3231ca840151af38"
SELF_FORCING_REVISION = "2f8b779212da279d212c22a509b66ad6552f350e"
CAUSAL_FORCING_REVISION = "2f8eb8bb6eeb1238da9d13e5420d342a74d634a6"
# thu-ml/Causal-Forcing source revision these recipes follow.
CAUSAL_FORCING_UPSTREAM_SOURCE_REVISION = "da3ddf1a590f30c9edc568e11c2f77170995c31c"

# Architecture/training semantics were audited against the official Wan
# source.  Model assets are independently pinned to the immutable Hub commit
# above so a README-only source change cannot silently alter checkpoint
# identity.
WAN21_UPSTREAM_SOURCE_REVISION = "9737cba9c1c3c4d04b33fcad41c111989865d315"

WAN_TOKENIZER_FILES = (
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


def _keys() -> tuple[ComponentKey, ...]:
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


def _checkpoints(
    repo_id: str,
    revision: str,
    *,
    sharded_dit: bool,
    image_to_video: bool,
) -> dict[str, CheckpointSpec]:
    metadata = {
        "license": "Apache-2.0",
        "repository_revision": revision,
        "upstream_source_revision": WAN21_UPSTREAM_SOURCE_REVISION,
    }
    dit_file = (
        "diffusion_pytorch_model.safetensors.index.json" if sharded_dit else "diffusion_pytorch_model.safetensors"
    )
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(dit_file,),
            allow_patterns=("diffusion_pytorch_model*",),
            metadata=metadata,
        ),
        "text-encoder": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
            metadata=metadata,
        ),
        "tokenizer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
            metadata=metadata,
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("Wan2.1_VAE.pth",),
            metadata=metadata,
        ),
    }
    if image_to_video:
        checkpoints["image-encoder"] = CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",),
        )
    return checkpoints


def _wan21_recipe(
    *,
    model_id: str,
    repo_id: str,
    revision: str,
    denoiser_factory: Callable,
    parameter_scale: str,
    shift: float,
    aliases: tuple[str, ...],
    image_to_video: bool,
    resolution: str | None = None,
) -> NativeDiffusionRecipe:
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = _checkpoints(
        repo_id,
        revision,
        sharded_dit=parameter_scale == "14B",
        image_to_video=image_to_video,
    )
    conditioner_checkpoints = {
        "weights": "text-encoder",
        "tokenizer": "tokenizer",
    }
    if image_to_video:
        conditioner_checkpoints["image_weights"] = "image-encoder"
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(denoiser, denoiser_factory, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_wan_image_text_conditioner if image_to_video else build_wan_text_conditioner,
                conditioner_checkpoints,
            ),
            ComponentSpec(
                initializer,
                build_wan_i2v_latent_initializer if image_to_video else build_wan_t2v_latent_initializer,
            ),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": shift},
            ),
            ComponentSpec(codec, build_wan_video_decoder, {"weights": "vae"}),
        ),
        execution=_execution(image_to_video=image_to_video),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "image-to-video" if image_to_video else "text-to-video",
                "classifier-free-guidance",
            }
        ),
        options={
            "latent_channels": 16,
            "spatial_compression": 8,
            "temporal_compression": 4,
        },
        metadata={
            "architecture": "wan2.1",
            "parameter_scale": parameter_scale,
            "resolution": resolution,
            "native_inference": True,
            "output_layout": "BCTHW",
            "license": "Apache-2.0",
            "upstream_source_revision": WAN21_UPSTREAM_SOURCE_REVISION,
        },
    )


def wan21_t2v_1p3b_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 T2V 1.3B recipe."""

    return _wan21_recipe(
        model_id=WAN21_T2V_1P3B_MODEL_ID,
        repo_id=WAN21_T2V_1P3B_REPO_ID,
        revision=WAN21_T2V_1P3B_REVISION,
        denoiser_factory=build_wan21_t2v_1p3b_denoiser,
        parameter_scale="1.3B",
        shift=8.0,
        aliases=(
            "wan2.1",
            "wan-2.1",
            "wan2p1",
            "wan2.1-t2v",
            "wan2p1-t2v",
            "wan21-t2v-1.3b",
            WAN21_T2V_1P3B_REPO_ID,
        ),
        image_to_video=False,
    )


def wan21_t2v_14b_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 T2V 14B recipe."""

    return _wan21_recipe(
        model_id=WAN21_T2V_14B_MODEL_ID,
        repo_id=WAN21_T2V_14B_REPO_ID,
        revision=WAN21_T2V_14B_REVISION,
        denoiser_factory=build_wan21_t2v_14b_denoiser,
        parameter_scale="14B",
        shift=5.0,
        aliases=("wan21-t2v-14b", WAN21_T2V_14B_REPO_ID),
        image_to_video=False,
    )


def wan21_i2v_14b_480p_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 I2V 14B 480P recipe."""

    return _wan21_recipe(
        model_id=WAN21_I2V_14B_480P_MODEL_ID,
        repo_id=WAN21_I2V_14B_480P_REPO_ID,
        revision=WAN21_I2V_14B_480P_REVISION,
        denoiser_factory=build_wan21_i2v_14b_denoiser,
        parameter_scale="14B",
        shift=3.0,
        aliases=("wan2.1-i2v", "wan2p1-i2v", "wan21-i2v-480p", WAN21_I2V_14B_480P_REPO_ID),
        image_to_video=True,
        resolution="480P",
    )


def wan21_i2v_14b_720p_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.1 I2V 14B 720P recipe."""

    return _wan21_recipe(
        model_id=WAN21_I2V_14B_720P_MODEL_ID,
        repo_id=WAN21_I2V_14B_720P_REPO_ID,
        revision=WAN21_I2V_14B_720P_REVISION,
        denoiser_factory=build_wan21_i2v_14b_denoiser,
        parameter_scale="14B",
        shift=5.0,
        aliases=("wan21-i2v-720p", WAN21_I2V_14B_720P_REPO_ID),
        image_to_video=True,
        resolution="720P",
    )


def wan21_vace_14b_recipe() -> NativeDiffusionRecipe:
    """Return Wan2.1 VACE 14B on the framework-owned sampling loop."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=("diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("diffusion_pytorch_model*", "config.json"),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=WAN21_VACE_14B_REPO_ID,
            revision=WAN21_VACE_14B_REVISION,
            files=("Wan2.1_VAE.pth",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=WAN21_VACE_14B_MODEL_ID,
        aliases=("wan-vace", "wan2.1-vace-14b", WAN21_VACE_14B_REPO_ID),
        components=(
            ComponentSpec(denoiser, build_wan21_vace_14b_denoiser, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(initializer, build_wan_vace_latent_initializer),
            ComponentSpec(scheduler, build_wan_flow_unipc_scheduler, options={"shift": 16.0}),
            ComponentSpec(codec, build_wan_video_decoder, {"weights": "vae"}),
        ),
        execution=_execution(image_to_video=True),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "controlled-video-generation",
                "image-to-video",
                "reference-to-video",
                "video-to-video",
                "classifier-free-guidance",
            }
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": "wan2.1-vace",
            "parameter_scale": "14B",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


def _wan22_a14b_recipe(*, image_to_video: bool) -> NativeDiffusionRecipe:
    repo_id = WAN22_I2V_A14B_REPO_ID if image_to_video else WAN22_T2V_A14B_REPO_ID
    revision = WAN22_I2V_A14B_REVISION if image_to_video else WAN22_T2V_A14B_REVISION
    model_id = WAN22_I2V_A14B_MODEL_ID if image_to_video else WAN22_T2V_A14B_MODEL_ID
    boundary_ratio = 0.900 if image_to_video else 0.875
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    metadata = {
        "license": "Apache-2.0",
        "repository_revision": revision,
        "upstream_source_revision": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
    }
    checkpoints = {
        "high-dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("high_noise_model/diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("high_noise_model/*",),
            metadata=metadata,
        ),
        "low-dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("low_noise_model/diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("low_noise_model/*",),
            metadata=metadata,
        ),
        "text-encoder": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
            metadata=metadata,
        ),
        "tokenizer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
            metadata=metadata,
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("Wan2.1_VAE.pth",),
            metadata=metadata,
        ),
    }
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=(repo_id,),
        components=(
            ComponentSpec(
                denoiser,
                (build_wan22_i2v_a14b_dual_denoiser if image_to_video else build_wan22_t2v_a14b_dual_denoiser),
                {"high_weights": "high-dit", "low_weights": "low-dit"},
                {"boundary_ratio": boundary_ratio},
            ),
            ComponentSpec(
                conditioner,
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(
                initializer,
                build_wan_i2v_latent_initializer if image_to_video else build_wan_t2v_latent_initializer,
            ),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": 5.0 if image_to_video else 12.0},
            ),
            ComponentSpec(codec, build_wan_video_decoder, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(
            strategy="wan22-dual-expert-guidance",
            bindings=_execution(image_to_video=image_to_video).bindings,
            options={
                "boundary_ratio": boundary_ratio,
                "low_noise_guidance_scale": 3.5 if image_to_video else 3.0,
                "high_noise_guidance_scale": 3.5 if image_to_video else 4.0,
            },
        ),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "image-to-video" if image_to_video else "text-to-video",
                "classifier-free-guidance",
                "dual-expert",
            }
        ),
        options={"latent_channels": 16, "spatial_compression": 8, "temporal_compression": 4},
        metadata={
            "architecture": "wan2.2-a14b",
            "parameter_scale": "A14B",
            "native_inference": True,
            "output_layout": "BCTHW",
            "expert_boundary_ratio": boundary_ratio,
            "repository_revision": revision,
        },
    )


def wan22_t2v_a14b_recipe() -> NativeDiffusionRecipe:
    """Return the released Wan2.2 T2V A14B dual-expert recipe."""

    return _wan22_a14b_recipe(image_to_video=False)


def wan22_i2v_a14b_recipe() -> NativeDiffusionRecipe:
    """Return the released Wan2.2 I2V A14B dual-expert recipe."""

    return _wan22_a14b_recipe(image_to_video=True)


def fastvideo_causal_wan22_i2v_14b_recipe() -> NativeDiffusionRecipe:
    """Return FastVideo's eight-step, first-frame-conditioned CausalWan2.2 rollout."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    repo_id = FASTVIDEO_CAUSAL_WAN22_REPO_ID
    revision = FASTVIDEO_CAUSAL_WAN22_REVISION
    tokenizer_files = (
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    checkpoints = {
        "high-dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("transformer/diffusion_pytorch_model.safetensors",),
            allow_patterns=("transformer/config.json", "transformer/diffusion_pytorch_model*"),
        ),
        "low-dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("transformer_2/diffusion_pytorch_model.safetensors",),
            allow_patterns=("transformer_2/config.json", "transformer_2/diffusion_pytorch_model*"),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("text_encoder/model.safetensors.index.json",),
            allow_patterns=("text_encoder/config.json", "text_encoder/model*.safetensors*"),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=tokenizer_files,
            allow_patterns=("tokenizer/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("vae/diffusion_pytorch_model.safetensors",),
            allow_patterns=("vae/*",),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=FASTVIDEO_CAUSAL_WAN22_MODEL_ID,
        aliases=(repo_id,),
        components=(
            ComponentSpec(
                denoiser,
                build_fastvideo_causal_wan22_denoiser,
                {"high_weights": "high-dit", "low_weights": "low-dit"},
                {
                    "boundary_ratio": 0.875,
                    "block_size": 3,
                    "cache_window": 21,
                    "sink_size": 0,
                },
            ),
            ComponentSpec(
                conditioner,
                build_diffusers_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"tokenizer_subdir": "tokenizer", "text_length": 512},
            ),
            ComponentSpec(initializer, build_wan_causal_i2v_latent_initializer),
            ComponentSpec(
                scheduler,
                build_fastvideo_causal_wan_self_forcing_scheduler,
                options={
                    "raw_timesteps": (1000, 850, 700, 550, 350, 275, 200, 125),
                    "shift": 12.0,
                    "boundary_ratio": 0.875,
                },
            ),
            ComponentSpec(codec, build_diffusers_wan_video_codec, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(
            strategy="autoregressive-window",
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "latent_encoder": codec,
                "scheduler": scheduler,
                "decoder": codec,
            },
            options={"prediction_mode": "flow", "guidance_mode": "standard"},
        ),
        checkpoints=checkpoints,
        capabilities=frozenset({"image-to-video", "causal-generation", "dual-expert", "self-forcing"}),
        options={
            "latent_channels": 16,
            "spatial_compression": 8,
            "temporal_compression": 4,
            "default_height": 480,
            "default_width": 832,
            "default_num_frames": 81,
            "default_num_inference_steps": 8,
            "default_guidance_scale": 1.0,
            "default_fps": 16,
        },
        metadata={
            "architecture": "fastvideo-causal-wan2.2-a14b",
            "parameter_scale": "A14B",
            "native_inference": True,
            "output_layout": "BCTHW",
            "expert_boundary_ratio": 0.875,
            "latent_block_size": 3,
            "cache_window": 21,
            "repository_revision": revision,
            "upstream_flashdreams_revision": "289da6f1d232de5abaa30d686c977b9c0040fe76",
        },
    )


def _forcing_wan21_t2v_1p3b_recipe(
    *,
    model_id: str,
    aliases: tuple[str, ...],
    dit: CheckpointSpec,
    block_size: int,
    fixed_timesteps: tuple[int, ...],
    first_block_timesteps: tuple[int, ...] = (),
    renoise_dtype: str | None = None,
    image_to_video: bool = False,
    capability: str,
    metadata: dict[str, object],
) -> NativeDiffusionRecipe:
    """Assemble a DMD-distilled causal Wan2.1 T2V 1.3B rollout on ``autoregressive-window``.

    Self-Forcing and the Causal-Forcing releases share the causal Wan graph, the
    shift-5 distilled-x0 sampler, the 21-latent-frame global KV cache, the Wan2.1
    text encoder and the bf16 codec. They differ in the generator checkpoint, the
    latent block size, the released step lists, and whether an image may seed
    the first latent frame.
    """

    import torch

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    wan = _checkpoints(
        WAN21_T2V_1P3B_REPO_ID,
        WAN21_T2V_1P3B_REVISION,
        sharded_dit=False,
        image_to_video=False,
    )
    execution_options: dict[str, object] = {
        "prediction_mode": "distilled-x0",
        "fixed_timesteps": fixed_timesteps,
    }
    if first_block_timesteps:
        execution_options["first_block_timesteps"] = first_block_timesteps
    execution_options["context_timestep"] = 0
    if renoise_dtype is not None:
        execution_options["renoise_dtype"] = renoise_dtype
    capabilities = {"text-to-video", "causal-generation", capability}
    if image_to_video:
        capabilities.add("image-to-video")
    return NativeDiffusionRecipe(
        model_id=model_id,
        aliases=aliases,
        components=(
            ComponentSpec(
                denoiser,
                build_self_forcing_wan21_denoiser,
                {"weights": "dit"},
                {"block_size": block_size, "cache_frames": 21, "timestep_shift": 5.0},
            ),
            ComponentSpec(
                conditioner,
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(
                initializer,
                build_wan_causal_forcing_latent_initializer if image_to_video else build_wan_t2v_latent_initializer,
            ),
            ComponentSpec(
                scheduler,
                build_wan_flow_match_euler_scheduler,
                options={"shift": 5.0},
            ),
            ComponentSpec(
                codec,
                build_wan_video_decoder,
                {"weights": "vae"},
                {"dtype": torch.bfloat16, "std_reciprocal_in_codec_dtype": True},
            ),
        ),
        execution=ExecutionSpec(
            strategy="autoregressive-window",
            bindings=dict(_execution(image_to_video=image_to_video).bindings),
            options=execution_options,
        ),
        checkpoints={
            "dit": dit,
            "text-encoder": wan["text-encoder"],
            "tokenizer": wan["tokenizer"],
            "vae": wan["vae"],
        },
        capabilities=frozenset(capabilities),
        options={
            "latent_channels": 16,
            "spatial_compression": 8,
            "temporal_compression": 4,
            "default_height": 480,
            "default_width": 832,
            "default_num_frames": 81,
            "default_num_inference_steps": len(fixed_timesteps),
            "default_guidance_scale": 1.0,
            "default_fps": 16,
        },
        metadata={
            "parameter_scale": "1.3B",
            "native_inference": True,
            "output_layout": "BCTHW",
            "latent_block_size": block_size,
            "cache_window": 21,
            **metadata,
        },
    )


def self_forcing_wan21_t2v_1p3b_recipe() -> NativeDiffusionRecipe:
    """Return Self-Forcing's four-step causal Wan2.1 T2V 1.3B rollout (DMD checkpoint).

    The released sampler denoises three-latent-frame blocks at the unwarped
    indices (1000, 750, 500, 250), which the runner warps onto the shift-5
    flow calendar, and commits each clean block to the KV cache at timestep 0.
    The codec runs in bf16 with the official wrapper's bf16 latent scale.
    """

    return _forcing_wan21_t2v_1p3b_recipe(
        model_id=SELF_FORCING_MODEL_ID,
        aliases=(SELF_FORCING_REPO_ID, "self-forcing-dmd"),
        dit=CheckpointSpec(
            repo_id=SELF_FORCING_REPO_ID,
            revision=SELF_FORCING_REVISION,
            files=("checkpoints/self_forcing_dmd.pt",),
            metadata={"license": "Apache-2.0", "repository_revision": SELF_FORCING_REVISION},
        ),
        block_size=3,
        fixed_timesteps=(1000, 750, 500, 250),
        capability="self-forcing",
        metadata={
            "architecture": "self-forcing-causal-wan2.1",
            "repository_revision": SELF_FORCING_REVISION,
            "upstream_source_revision": "33593df3e81fa3ec10239271dd2c100facac6de1",
        },
    )


def _causal_forcing_recipe(
    *,
    model_id: str,
    aliases: tuple[str, ...] = (),
    checkpoint_file: str,
    block_size: int,
    fixed_timesteps: tuple[int, ...],
    first_block_timesteps: tuple[int, ...] = (),
    image_to_video: bool = False,
) -> NativeDiffusionRecipe:
    """Return one released Causal-Forcing generator on the shared forcing rollout.

    The official scheduler re-noises a one-frame block of one sample in bf16 (see
    ``AutoregressiveWindowRunner._renoise_in_dtype``), so frame-wise students
    re-noise in bf16; three-frame blocks keep the float32 re-noise.
    """

    return _forcing_wan21_t2v_1p3b_recipe(
        model_id=model_id,
        aliases=aliases,
        dit=CheckpointSpec(
            repo_id=CAUSAL_FORCING_REPO_ID,
            revision=CAUSAL_FORCING_REVISION,
            files=(checkpoint_file,),
            metadata={"license": "Apache-2.0", "repository_revision": CAUSAL_FORCING_REVISION},
        ),
        block_size=block_size,
        fixed_timesteps=fixed_timesteps,
        first_block_timesteps=first_block_timesteps,
        renoise_dtype="bfloat16" if block_size == 1 else None,
        image_to_video=image_to_video,
        capability="causal-forcing",
        metadata={
            "architecture": "causal-forcing-causal-wan2.1",
            "repository_revision": CAUSAL_FORCING_REVISION,
            "upstream_source_revision": CAUSAL_FORCING_UPSTREAM_SOURCE_REVISION,
        },
    )


def causal_forcing_chunkwise_recipe() -> NativeDiffusionRecipe:
    """Return Causal-Forcing's chunk-wise four-step student: three-latent-frame blocks, T2V.

    It is the default variant, so the public ``causal-forcing`` model ID resolves to it.
    """

    return _causal_forcing_recipe(
        model_id=CAUSAL_FORCING_CHUNKWISE_MODEL_ID,
        aliases=("causal-forcing",),
        checkpoint_file="chunkwise/causal_forcing.pt",
        block_size=3,
        fixed_timesteps=(1000, 750, 500, 250),
    )


def causal_forcing_framewise_recipe() -> NativeDiffusionRecipe:
    """Return Causal-Forcing's frame-wise four-step student: one-latent-frame blocks.

    An input image makes it the official I2V route: the encoded image is the
    clean first latent frame, committed to the KV cache before the remaining
    frames are rolled out.
    """

    return _causal_forcing_recipe(
        model_id=CAUSAL_FORCING_FRAMEWISE_MODEL_ID,
        checkpoint_file="framewise/causal_forcing.pt",
        block_size=1,
        fixed_timesteps=(1000, 750, 500, 250),
        image_to_video=True,
    )


def causal_forcing_framewise_1step_recipe() -> NativeDiffusionRecipe:
    """Return Causal Forcing++ frame-wise one-step: four steps on the first frame, then one per frame."""

    return _causal_forcing_recipe(
        model_id=CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID,
        checkpoint_file="causal-forcing++/framewise-1step.pt",
        block_size=1,
        fixed_timesteps=(1000,),
        first_block_timesteps=(1000, 750, 500, 250),
    )


def causal_forcing_framewise_2step_recipe() -> NativeDiffusionRecipe:
    """Return Causal Forcing++ frame-wise two-step: four steps on the first frame, then two per frame."""

    return _causal_forcing_recipe(
        model_id=CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID,
        checkpoint_file="causal-forcing++/framewise-2step.pt",
        block_size=1,
        fixed_timesteps=(1000, 500),
        first_block_timesteps=(1000, 750, 500, 250),
    )


def wan22_ti2v_5b_recipe() -> NativeDiffusionRecipe:
    """Return the native Wan2.2 5B recipe with optional first-frame locking."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=("diffusion_pytorch_model.safetensors.index.json",),
            allow_patterns=("diffusion_pytorch_model*",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
        ),
        "vae": CheckpointSpec(
            repo_id=WAN22_TI2V_5B_REPO_ID,
            revision=WAN22_TI2V_5B_REVISION,
            files=("Wan2.2_VAE.pth",),
        ),
    }
    bindings = dict(_execution(image_to_video=True).bindings)
    return NativeDiffusionRecipe(
        model_id=WAN22_TI2V_5B_MODEL_ID,
        aliases=(
            "wan2.2",
            "wan-2.2",
            "wan2p2",
            "wan2.2-ti2v-5b-1280x704-121f",
            WAN22_TI2V_5B_REPO_ID,
        ),
        components=(
            ComponentSpec(denoiser, build_wan22_ti2v_5b_denoiser, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_wan_text_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
            ),
            ComponentSpec(initializer, build_wan_ti2v_latent_initializer),
            ComponentSpec(
                scheduler,
                build_wan_flow_unipc_scheduler,
                options={"shift": 5.0},
            ),
            ComponentSpec(codec, build_wan_video_vae38_decoder, {"weights": "vae"}),
        ),
        execution=ExecutionSpec(strategy="masked-latent", bindings=bindings),
        checkpoints=checkpoints,
        capabilities=frozenset({"text-to-video", "image-to-video", "classifier-free-guidance", "per-token-timestep"}),
        options={
            "latent_channels": 48,
            "spatial_compression": 16,
            "temporal_compression": 4,
        },
        metadata={
            "architecture": "wan2.2-ti2v",
            "parameter_scale": "5B",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


__all__ = [
    "CAUSAL_FORCING_CHUNKWISE_MODEL_ID",
    "CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID",
    "CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID",
    "CAUSAL_FORCING_FRAMEWISE_MODEL_ID",
    "CAUSAL_FORCING_REPO_ID",
    "CAUSAL_FORCING_REVISION",
    "CAUSAL_FORCING_UPSTREAM_SOURCE_REVISION",
    "FASTVIDEO_CAUSAL_WAN22_MODEL_ID",
    "FASTVIDEO_CAUSAL_WAN22_REPO_ID",
    "FASTVIDEO_CAUSAL_WAN22_REVISION",
    "SELF_FORCING_MODEL_ID",
    "SELF_FORCING_REPO_ID",
    "SELF_FORCING_REVISION",
    "WAN21_I2V_14B_480P_MODEL_ID",
    "WAN21_I2V_14B_480P_REPO_ID",
    "WAN21_I2V_14B_480P_REVISION",
    "WAN21_I2V_14B_720P_MODEL_ID",
    "WAN21_I2V_14B_720P_REPO_ID",
    "WAN21_I2V_14B_720P_REVISION",
    "WAN21_T2V_1P3B_MODEL_ID",
    "WAN21_T2V_1P3B_REPO_ID",
    "WAN21_T2V_1P3B_REVISION",
    "WAN21_T2V_14B_MODEL_ID",
    "WAN21_T2V_14B_REPO_ID",
    "WAN21_T2V_14B_REVISION",
    "WAN21_UPSTREAM_SOURCE_REVISION",
    "WAN21_VACE_14B_MODEL_ID",
    "WAN21_VACE_14B_REPO_ID",
    "WAN21_VACE_14B_REVISION",
    "WAN22_I2V_A14B_MODEL_ID",
    "WAN22_I2V_A14B_REPO_ID",
    "WAN22_I2V_A14B_REVISION",
    "WAN22_T2V_A14B_MODEL_ID",
    "WAN22_T2V_A14B_REPO_ID",
    "WAN22_T2V_A14B_REVISION",
    "WAN22_TI2V_5B_MODEL_ID",
    "WAN22_TI2V_5B_REPO_ID",
    "WAN22_TI2V_5B_REVISION",
    "WAN_TOKENIZER_FILES",
    "causal_forcing_chunkwise_recipe",
    "causal_forcing_framewise_1step_recipe",
    "causal_forcing_framewise_2step_recipe",
    "causal_forcing_framewise_recipe",
    "fastvideo_causal_wan22_i2v_14b_recipe",
    "self_forcing_wan21_t2v_1p3b_recipe",
    "wan21_i2v_14b_480p_recipe",
    "wan21_i2v_14b_720p_recipe",
    "wan21_t2v_1p3b_recipe",
    "wan21_t2v_14b_recipe",
    "wan21_vace_14b_recipe",
    "wan22_i2v_a14b_recipe",
    "wan22_t2v_a14b_recipe",
    "wan22_ti2v_5b_recipe",
]
