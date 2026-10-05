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

Checkpoint identity is pinned to immutable Hub revisions plus file SHA-256
so a README-only upstream change cannot silently swap weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.wan import (
    build_diffusers_wan_video_codec,
    build_wan_video_decoder,
    build_wan_video_vae38_decoder,
)
from ..models.denoisers.wan import (
    build_fastvideo_causal_wan22_denoiser,
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
    build_wan_causal_i2v_latent_initializer,
    build_wan_i2v_latent_initializer,
    build_wan_t2v_latent_initializer,
    build_wan_ti2v_latent_initializer,
    build_wan_vace_latent_initializer,
)
from ..schedulers import (
    build_fastvideo_causal_wan_self_forcing_scheduler,
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

WAN21_T2V_1P3B_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
WAN21_T2V_14B_REPO_ID = "Wan-AI/Wan2.1-T2V-14B"
WAN21_I2V_14B_480P_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN21_I2V_14B_720P_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-720P"
WAN22_TI2V_5B_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_T2V_A14B_REPO_ID = "Wan-AI/Wan2.2-T2V-A14B"
WAN22_I2V_A14B_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B"
WAN21_VACE_14B_REPO_ID = "Wan-AI/Wan2.1-VACE-14B"
FASTVIDEO_CAUSAL_WAN22_REPO_ID = "FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers"

WAN21_T2V_1P3B_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAN21_T2V_14B_REVISION = "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
WAN21_I2V_14B_480P_REVISION = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
WAN21_I2V_14B_720P_REVISION = "8823af45fcc58a8aa999a54b04be9abc7d2aac98"
WAN22_TI2V_5B_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
WAN22_T2V_A14B_REVISION = "c8c270b13ee05bfa474194ac9fb07a5868a97cea"
WAN22_I2V_A14B_REVISION = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
WAN21_VACE_14B_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"
FASTVIDEO_CAUSAL_WAN22_REVISION = "0977572ccf137da5e577d62e3231ca840151af38"

# Architecture/training semantics were audited against the official Wan
# source.  Model assets are independently pinned to the immutable Hub commit
# above so a README-only source change cannot silently alter checkpoint
# identity.
WAN21_UPSTREAM_SOURCE_REVISION = "9737cba9c1c3c4d04b33fcad41c111989865d315"
WAN21_T2V_1P3B_FILE_SHA256 = {
    "diffusion_pytorch_model.safetensors": (
        "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f"
    ),
    "models_t5_umt5-xxl-enc-bf16.pth": (
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
    ),
    "Wan2.1_VAE.pth": (
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
    ),
    "google/umt5-xxl/spiece.model": (
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"
    ),
    "google/umt5-xxl/tokenizer.json": (
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
    ),
}
WAN21_T2V_1P3B_FILE_SIZE_BYTES = {
    "diffusion_pytorch_model.safetensors": 5_676_070_424,
    "models_t5_umt5-xxl-enc-bf16.pth": 11_361_920_418,
    "Wan2.1_VAE.pth": 507_609_880,
    "google/umt5-xxl/special_tokens_map.json": 6_623,
    "google/umt5-xxl/spiece.model": 4_548_313,
    "google/umt5-xxl/tokenizer.json": 16_837_417,
    "google/umt5-xxl/tokenizer_config.json": 61_728,
}

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
    file_sha256: Mapping[str, str] | None = None,
    file_size_bytes: Mapping[str, int] | None = None,
) -> dict[str, CheckpointSpec]:
    integrity = dict(file_sha256 or {})
    sizes = dict(file_size_bytes or {})
    metadata = {
        "license": "Apache-2.0",
        "repository_revision": revision,
        "upstream_source_revision": WAN21_UPSTREAM_SOURCE_REVISION,
    }
    dit_file = (
        "diffusion_pytorch_model.safetensors.index.json"
        if sharded_dit
        else "diffusion_pytorch_model.safetensors"
    )
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=(dit_file,),
            allow_patterns=("diffusion_pytorch_model*",),
            metadata=metadata,
            file_sha256=(
                {dit_file: integrity[dit_file]}
                if dit_file in integrity
                else {}
            ),
            file_size_bytes=(
                {dit_file: sizes[dit_file]}
                if dit_file in sizes
                else {}
            ),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("models_t5_umt5-xxl-enc-bf16.pth",),
            metadata=metadata,
            file_sha256=(
                {
                    "models_t5_umt5-xxl-enc-bf16.pth": integrity[
                        "models_t5_umt5-xxl-enc-bf16.pth"
                    ]
                }
                if "models_t5_umt5-xxl-enc-bf16.pth" in integrity
                else {}
            ),
            file_size_bytes=(
                {
                    "models_t5_umt5-xxl-enc-bf16.pth": sizes[
                        "models_t5_umt5-xxl-enc-bf16.pth"
                    ]
                }
                if "models_t5_umt5-xxl-enc-bf16.pth" in sizes
                else {}
            ),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=WAN_TOKENIZER_FILES,
            allow_patterns=("google/umt5-xxl/*",),
            metadata=metadata,
            file_sha256={
                name: integrity[name]
                for name in WAN_TOKENIZER_FILES
                if name in integrity
            },
            file_size_bytes={
                name: sizes[name]
                for name in WAN_TOKENIZER_FILES
                if name in sizes
            },
        ),
        "vae": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("Wan2.1_VAE.pth",),
            metadata=metadata,
            file_sha256=(
                {"Wan2.1_VAE.pth": integrity["Wan2.1_VAE.pth"]}
                if "Wan2.1_VAE.pth" in integrity
                else {}
            ),
            file_size_bytes=(
                {"Wan2.1_VAE.pth": sizes["Wan2.1_VAE.pth"]}
                if "Wan2.1_VAE.pth" in sizes
                else {}
            ),
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
        file_sha256=(
            WAN21_T2V_1P3B_FILE_SHA256
            if model_id == WAN21_T2V_1P3B_MODEL_ID
            else None
        ),
        file_size_bytes=(
            WAN21_T2V_1P3B_FILE_SIZE_BYTES
            if model_id == WAN21_T2V_1P3B_MODEL_ID
            else None
        ),
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
                build_wan_i2v_latent_initializer
                if image_to_video
                else build_wan_t2v_latent_initializer,
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
                (
                    build_wan22_i2v_a14b_dual_denoiser
                    if image_to_video
                    else build_wan22_t2v_a14b_dual_denoiser
                ),
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
                build_wan_i2v_latent_initializer
                if image_to_video
                else build_wan_t2v_latent_initializer,
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
        capabilities=frozenset(
            {"image-to-video", "causal-generation", "dual-expert", "self-forcing"}
        ),
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
        capabilities=frozenset(
            {"text-to-video", "image-to-video", "classifier-free-guidance", "per-token-timestep"}
        ),
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
    "FASTVIDEO_CAUSAL_WAN22_MODEL_ID",
    "FASTVIDEO_CAUSAL_WAN22_REPO_ID",
    "FASTVIDEO_CAUSAL_WAN22_REVISION",
    "WAN21_I2V_14B_480P_MODEL_ID",
    "WAN21_I2V_14B_480P_REPO_ID",
    "WAN21_I2V_14B_480P_REVISION",
    "WAN21_I2V_14B_720P_MODEL_ID",
    "WAN21_I2V_14B_720P_REPO_ID",
    "WAN21_I2V_14B_720P_REVISION",
    "WAN21_T2V_1P3B_MODEL_ID",
    "WAN21_T2V_1P3B_REPO_ID",
    "WAN21_T2V_1P3B_REVISION",
    "WAN21_T2V_1P3B_FILE_SHA256",
    "WAN21_T2V_14B_MODEL_ID",
    "WAN21_T2V_14B_REPO_ID",
    "WAN21_T2V_14B_REVISION",
    "WAN22_TI2V_5B_MODEL_ID",
    "WAN22_TI2V_5B_REPO_ID",
    "WAN22_TI2V_5B_REVISION",
    "WAN22_T2V_A14B_MODEL_ID",
    "WAN22_T2V_A14B_REPO_ID",
    "WAN22_T2V_A14B_REVISION",
    "WAN22_I2V_A14B_MODEL_ID",
    "WAN22_I2V_A14B_REPO_ID",
    "WAN22_I2V_A14B_REVISION",
    "WAN21_VACE_14B_MODEL_ID",
    "WAN21_VACE_14B_REPO_ID",
    "WAN21_VACE_14B_REVISION",
    "WAN_TOKENIZER_FILES",
    "WAN21_UPSTREAM_SOURCE_REVISION",
    "fastvideo_causal_wan22_i2v_14b_recipe",
    "wan21_i2v_14b_480p_recipe",
    "wan21_i2v_14b_720p_recipe",
    "wan21_t2v_1p3b_recipe",
    "wan21_t2v_14b_recipe",
    "wan22_ti2v_5b_recipe",
    "wan22_t2v_a14b_recipe",
    "wan22_i2v_a14b_recipe",
    "wan21_vace_14b_recipe",
]
