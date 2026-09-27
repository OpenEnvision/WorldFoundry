"""Native declarative recipes for the Sana image / video / world family.

Each public function returns a :class:`NativeDiffusionRecipe`.  Variant
metadata (task, runner, Hub path, defaults) lives in
:mod:`.sana_variants`; this module only binds components and a strategy.
Recipes do not construct runners.

Family split
------------
- Image (Sana / Sana 1.5): DC-AE codec, Gemma conditioner, flow-match
  scheduler, ``standard`` strategy, classifier-free guidance.
- Sprint: SCM scheduler, embedded guidance (no CFG), optional teacher
  checkpoint, ``decoder_input_scale=0.5``.
- ControlNet: HED-conditioned denoiser, encoded initialization so the
  control image is VAE-encoded before denoise.
- Video 480p: Wan VAE + Flow-DPM; 720p: LTX tensor codec + Flow-DPM.
- Long streaming: ``chunked-kv-cache`` with Euler, KV sink tokens.
- SANA-WM: world-model DiT, flow-match (shift 9.8), LTX-2 codec,
  ``standard`` with encoded initialization (camera / interactive I2V).
"""

from __future__ import annotations

from collections.abc import Mapping

from ..components import ComponentKey, ComponentKind, ComponentSpec, ExecutionSpec
from ..loaders import CheckpointSpec
from ..models.autoencoders.ltx import (
    build_diffusers_ltx2_tensor_video_codec,
    build_ltx_tensor_video_codec,
)
from ..models.autoencoders.sana import build_sana_dc_autoencoder
from ..models.autoencoders.wan import build_wan_video_decoder
from ..models.denoisers.sana import (
    build_sana_controlnet_denoiser,
    build_sana_image_denoiser,
    build_sana_sprint_denoiser,
    build_sana_streaming_denoiser,
    build_sana_video_denoiser,
    build_sana_world_denoiser,
    build_sana_world_streaming_denoiser,
)
from ..models.denoisers.sana_refiner import build_sana_wm_ltx2_refiner_processor
from ..models.encoders.sana import (
    build_sana_prompt_conditioner,
    build_sana_wm_refiner_conditioner,
)
from ..models.initializers.sana import (
    build_sana_controlnet_initializer,
    build_sana_image_initializer,
    build_sana_video_initializer,
    build_sana_video_to_video_initializer,
    build_sana_world_initializer,
)
from ..schedulers import (
    build_sana_flow_dpm_scheduler,
    build_sana_flow_match_scheduler,
    build_sana_longlive_scheduler,
    build_sana_scm_scheduler,
    build_sana_streaming_euler_scheduler,
    build_sana_wm_streaming_euler_scheduler,
)
from .sana_variants import SANA_ALIASES, SanaVariant, get_sana_variant
from .spec import NativeDiffusionRecipe

SANA_GEMMA_REPO_ID = "Efficient-Large-Model/gemma-2-2b-it"
SANA_GEMMA_REVISION = "569d9809d0c8b6722d4d31b5a77a2ec7a400650a"
SANA_GEMMA_WEIGHT_SHA256 = "bf06a1e6cfe1610beb98a2975e5602e7fc108d902b3ff9dd62282d749c7a2394"
SANA_GEMMA_WEIGHT_SIZE_BYTES = 5228717512
SANA_GEMMA_CONFIG_SHA256 = "b3195cba8e4a8e637f15bc0bab6b48fd0f4e02967fac72925e2dfb02c4d5b8d7"
SANA_GEMMA_CONFIG_SIZE_BYTES = 881
SANA_DCAE_REPO_ID = "mit-han-lab/dc-ae-f32c32-sana-1.1"
SANA_DCAE_REVISION = "64547fd6e83db7bec338319d05ac9fe26d449ec2"
SANA_DCAE_WEIGHT_SHA256 = "01a99a6d08b8e1d54bf9af28cab521432cdeee372720e84c38bac8b8abe27ba6"
SANA_DCAE_WEIGHT_SIZE_BYTES = 1249046372
SANA_LTX_REPO_ID = "Lightricks/LTX-2"
SANA_LTX_REVISION = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
SANA_WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
SANA_WAN_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
SANA_WM_REPO_ID = "Efficient-Large-Model/SANA-WM_bidirectional"
SANA_WM_REVISION = "90e0ff3b8f1f9b54a92b4b707edeaa27073aec84"
SANA_WM_STREAMING_MODEL_ID = "sana-wm-streaming"
SANA_WM_STREAMING_REPO_ID = "Efficient-Large-Model/SANA-WM_streaming"
SANA_WM_STREAMING_REVISION = "c7694938c854b1fcc29468e9514f1ea2d0c4ba8e"
SANA_SPRINT_TEACHER_CHECKPOINTS = {
    "sana-sprint-600m-1024px": CheckpointSpec(
        repo_id="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher",
        revision="141411fcb8360dc1e6ae91e7c7ed69371adb058c",
        files=("checkpoints/Sana_Sprint_0.6B_1024px_teacher.pth",),
        file_sha256={
            "checkpoints/Sana_Sprint_0.6B_1024px_teacher.pth": (
                "1afe15602c9a60d32510aed0a83e3d75b9c289aa41d202f6444e1892fc14cc1b"
            )
        },
        file_size_bytes={"checkpoints/Sana_Sprint_0.6B_1024px_teacher.pth": 2375214166},
    ),
    "sana-sprint-1600m-1024px": CheckpointSpec(
        repo_id="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_teacher",
        revision="f111cf761e05a1ad460cd5fb528c1a107327b459",
        files=("checkpoints/Sana_Sprint_1.6B_1024px_teacher.pth",),
        file_sha256={
            "checkpoints/Sana_Sprint_1.6B_1024px_teacher.pth": (
                "9829ee64eb6ae3cb372e0f7bd2b62372fe6f106ddf0bafcafa99356247d6a548"
            )
        },
        file_size_bytes={"checkpoints/Sana_Sprint_1.6B_1024px_teacher.pth": 6430670590},
    ),
}

SANA_TOKENIZER_FILES = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
SANA_TOKENIZER_SHA256 = {
    "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
    "tokenizer.json": "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89",
}
SANA_TOKENIZER_SIZE_BYTES = {
    "special_tokens_map.json": 636,
    "tokenizer.json": 17525357,
    "tokenizer.model": 4241003,
    "tokenizer_config.json": 46996,
}


def _aliases(model_id: str) -> tuple[str, ...]:
    return tuple(alias for alias, target in SANA_ALIASES.items() if target == model_id)


def _dit_file(variant: SanaVariant) -> str:
    prefix = f"hf://{variant.repo_id}/"
    if not variant.model_path.startswith(prefix):
        raise ValueError(f"Sana native recipe requires a repository-relative checkpoint: {variant.model_path}")
    return variant.model_path.removeprefix(prefix)


def _shared_checkpoints(variant: SanaVariant) -> dict[str, CheckpointSpec]:
    dit_file = _dit_file(variant)
    return {
        "dit": CheckpointSpec(
            repo_id=variant.repo_id,
            revision=variant.checkpoint_revision,
            files=(dit_file,),
            file_sha256=(
                {dit_file: variant.checkpoint_sha256}
                if variant.checkpoint_sha256 is not None
                else {}
            ),
            file_size_bytes=(
                {dit_file: variant.checkpoint_size_bytes}
                if variant.checkpoint_size_bytes is not None
                else {}
            ),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=SANA_GEMMA_REPO_ID,
            revision=SANA_GEMMA_REVISION,
            files=("gemma-2-2b-it.safetensors",),
            allow_patterns=("config.json", "gemma-2-2b-it.safetensors"),
            file_sha256={"gemma-2-2b-it.safetensors": SANA_GEMMA_WEIGHT_SHA256},
            file_size_bytes={"gemma-2-2b-it.safetensors": SANA_GEMMA_WEIGHT_SIZE_BYTES},
            resource_sha256={"config.json": SANA_GEMMA_CONFIG_SHA256},
            resource_size_bytes={"config.json": SANA_GEMMA_CONFIG_SIZE_BYTES},
        ),
        "tokenizer": CheckpointSpec(
            repo_id=SANA_GEMMA_REPO_ID,
            revision=SANA_GEMMA_REVISION,
            files=SANA_TOKENIZER_FILES,
            allow_patterns=SANA_TOKENIZER_FILES,
            file_sha256=SANA_TOKENIZER_SHA256,
            file_size_bytes=SANA_TOKENIZER_SIZE_BYTES,
        ),
    }


def _keys() -> tuple[ComponentKey, ...]:
    return (
        ComponentKey(ComponentKind.DENOISER),
        ComponentKey(ComponentKind.CONDITIONER),
        ComponentKey(ComponentKind.LATENT_INITIALIZER),
        ComponentKey(ComponentKind.SCHEDULER),
        ComponentKey(ComponentKind.LATENT_ENCODER, "codec"),
    )


def _execution(
    *,
    encoded_initialization: bool,
    strategy: str = "standard",
    options: Mapping[str, object] | None = None,
) -> ExecutionSpec:
    denoiser, conditioner, initializer, scheduler, codec = _keys()
    bindings = {
        "denoiser": denoiser,
        "conditioner": conditioner,
        "latent_initializer": initializer,
        "scheduler": scheduler,
        "decoder": codec,
    }
    if encoded_initialization:
        bindings["latent_encoder"] = codec
    return ExecutionSpec(strategy=strategy, bindings=bindings, options=options or {})


def _parameter_scale(model_id: str) -> str:
    tokens = model_id.lower().split("-")
    if "600m" in tokens:
        return "600M"
    if "4800m" in tokens:
        return "4800M"
    return "1600M"


def _image_recipe(variant: SanaVariant) -> NativeDiffusionRecipe:
    """Compose image / Sprint / ControlNet on ``standard`` (SCM vs flow-match)."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = _shared_checkpoints(variant)
    teacher_checkpoint = SANA_SPRINT_TEACHER_CHECKPOINTS.get(variant.model_id)
    if teacher_checkpoint is not None:
        checkpoints["teacher"] = teacher_checkpoint
    checkpoints["codec"] = CheckpointSpec(
        repo_id=SANA_DCAE_REPO_ID,
        revision=SANA_DCAE_REVISION,
        files=("model.safetensors",),
        file_sha256={"model.safetensors": SANA_DCAE_WEIGHT_SHA256},
        file_size_bytes={"model.safetensors": SANA_DCAE_WEIGHT_SIZE_BYTES},
        metadata={"license": "Apache-2.0"},
    )
    sprint = variant.runner == "sprint"
    controlnet = variant.runner == "controlnet"
    input_size = int(variant.resolution.removesuffix("px").replace("K", "000")) // 32
    if "2k" in variant.model_id:
        input_size = 64
    elif "4k" in variant.model_id:
        input_size = 128
    denoiser_factory = (
        build_sana_sprint_denoiser
        if sprint
        else build_sana_controlnet_denoiser
        if controlnet
        else build_sana_image_denoiser
    )
    initializer_factory = build_sana_controlnet_initializer if controlnet else build_sana_image_initializer
    scheduler_factory = build_sana_scm_scheduler if sprint else build_sana_flow_match_scheduler
    scheduler_options = {"sigma_data": 0.5} if sprint else {
        "shift": (
            6.0 if variant.model_id == "sana-1600m-4k-bf16"
            else 4.0 if _parameter_scale(variant.model_id) == "600M"
            else 3.0
        )
    }
    codec_options = {"decoder_input_scale": 0.5} if sprint else {}
    return NativeDiffusionRecipe(
        model_id=variant.model_id,
        aliases=_aliases(variant.model_id),
        components=(
            ComponentSpec(
                denoiser,
                denoiser_factory,
                {"weights": "dit"},
                {
                    "input_size": input_size,
                    "parameter_scale": _parameter_scale(variant.model_id),
                    "sana_1p5": variant.model_id.startswith("sana1p5"),
                },
            ),
            ComponentSpec(
                conditioner,
                build_sana_prompt_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"max_length": 300},
            ),
            ComponentSpec(
                initializer,
                initializer_factory,
                options={"noise_scale": 0.5 if sprint else 1.0},
            ),
            ComponentSpec(scheduler, scheduler_factory, options=scheduler_options),
            ComponentSpec(codec, build_sana_dc_autoencoder, {"weights": "codec"}, codec_options),
        ),
        execution=_execution(encoded_initialization=controlnet),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "control-image-to-image" if controlnet else "text-to-image",
                "embedded-guidance" if sprint else "classifier-free-guidance",
            }
        ),
        options={"latent_channels": 32, "spatial_compression": 32, "temporal_compression": 1},
        metadata={
            "architecture": "sana-sprint" if sprint else "sana-controlnet" if controlnet else "sana",
            "parameter_scale": _parameter_scale(variant.model_id),
            "resolution": variant.resolution,
            "native_inference": True,
            "output_layout": "BCHW",
        },
    )


def _video_recipe(variant: SanaVariant) -> NativeDiffusionRecipe:
    """Compose video or streaming; long-KV uses ``chunked-kv-cache``."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = _shared_checkpoints(variant)
    streaming = variant.runner == "streaming"
    longlive = variant.model_id == "longsana-video-2b-480p"
    long_streaming = streaming and variant.mode == "long_streaming"
    resolution = "720p" if "720p" in variant.model_id else "480p"
    if resolution == "480p":
        checkpoints["codec"] = CheckpointSpec(
            repo_id=SANA_WAN_REPO_ID,
            revision=SANA_WAN_REVISION,
            files=("Wan2.1_VAE.pth",),
        )
        codec_factory = build_wan_video_decoder
        codec_options: dict[str, object] = {"tiled": True}
        channels, spatial, temporal = 16, 8, 4
    elif streaming and variant.mode == "bidirectional_short":
        checkpoints["codec"] = CheckpointSpec(
            repo_id=SANA_LTX_REPO_ID,
            revision=SANA_LTX_REVISION,
            files=(
                "vae/diffusion_pytorch_model.safetensors",
                "vae/config.json",
            ),
        )
        codec_factory = build_diffusers_ltx2_tensor_video_codec
        codec_options = {"tiled": True}
        channels, spatial, temporal = 128, 32, 8
    else:
        checkpoints["codec"] = CheckpointSpec(
            repo_id=SANA_LTX_REPO_ID,
            revision=SANA_LTX_REVISION,
            files=("ltx-2-19b-dev.safetensors",),
        )
        codec_factory = build_ltx_tensor_video_codec
        codec_options = {
            "tiled": True,
            "spatial_tile_size": 768,
            "spatial_overlap": 64,
            "temporal_tile_size": 96,
            "temporal_overlap": 24,
        }
        channels, spatial, temporal = 128, 32, 8
    return NativeDiffusionRecipe(
        model_id=variant.model_id,
        aliases=_aliases(variant.model_id),
        components=(
            ComponentSpec(
                denoiser,
                build_sana_streaming_denoiser if streaming else build_sana_video_denoiser,
                {"weights": "dit"},
                {
                    "resolution": resolution,
                    "autoregressive": variant.mode == "long_streaming",
                    "longlive": longlive,
                },
            ),
            ComponentSpec(
                conditioner,
                build_sana_prompt_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"max_length": 300},
            ),
            ComponentSpec(
                initializer,
                build_sana_video_to_video_initializer if streaming else build_sana_video_initializer,
                options={
                    "channels": channels,
                    "spatial_compression": spatial,
                    "temporal_compression": temporal,
                    # 720 is not divisible by the LTX codec's 32x spatial
                    # compression. Diffuse on the next complete latent row
                    # and crop the decoded pixels back to the requested 720p.
                    "allow_spatial_padding": resolution == "720p",
                },
            ),
            ComponentSpec(
                scheduler,
                (
                    build_sana_longlive_scheduler
                    if longlive
                    else build_sana_streaming_euler_scheduler
                    if long_streaming
                    else build_sana_flow_dpm_scheduler
                ),
                options={"shift": 8.0 if resolution == "720p" else 7.0},
            ),
            ComponentSpec(codec, codec_factory, {"weights": "codec"}, codec_options),
        ),
        execution=_execution(
            encoded_initialization=streaming,
            strategy=(
                "chunked-additive-cache" if longlive
                else "chunked-kv-cache" if long_streaming else "standard"
            ),
            options={
                "base_chunk_frames": 3,
                "num_cached_chunks": 2,
                "sink_token": True,
            }
            if long_streaming
            else {"base_chunk_frames": 10, "num_cached_chunks": -1, "sink_token": False}
            if longlive
            else None,
        ),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "video-to-video" if streaming else "text-to-video",
                "classifier-free-guidance",
            }
        ),
        options={
            "latent_channels": channels,
            "spatial_compression": spatial,
            "temporal_compression": temporal,
        },
        metadata={
            "architecture": "sana-streaming" if streaming else "sana-video",
            "parameter_scale": "2B",
            "resolution": variant.resolution,
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


def sana_recipe(model_id: str) -> NativeDiffusionRecipe:
    """Resolve one public Sana ID into image or video role composition.

    Dispatches on :class:`SanaVariant.task`.  Image / Sprint / ControlNet
    bind DiT, Gemma, DC-AE, and either flow-match or SCM.  Video binds the
    video or streaming DiT, Gemma, Wan or LTX codec, and Flow-DPM (or
    streaming Euler for long-KV).  Strategy is ``standard`` unless
    ``mode=="long_streaming"`` (then ``chunked-kv-cache``).
    """

    variant = get_sana_variant(model_id)
    if variant.task in {"text-to-video", "video-to-video"}:
        return _video_recipe(variant)
    return _image_recipe(variant)


def sana_world_recipe() -> NativeDiffusionRecipe:
    """SANA-WM camera-controlled world model on the five native roles.

    Binds ``build_sana_world_denoiser``, Gemma conditioner, world
    initializer, flow-match scheduler (shift 9.8), and a tiled LTX-2
    tensor codec used as both encoder and decoder.  Strategy is
    ``standard`` with encoded initialization.  CFG is on; capabilities
    include camera-control and interactive world-model I2V.
    """

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=SANA_WM_REPO_ID,
            revision=SANA_WM_REVISION,
            files=("dit/sana_wm_1600m_720p.safetensors",),
        ),
        "text-encoder": CheckpointSpec(
            repo_id=SANA_GEMMA_REPO_ID,
            revision=SANA_GEMMA_REVISION,
            files=("gemma-2-2b-it.safetensors",),
            allow_patterns=("config.json", "gemma-2-2b-it.safetensors"),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=SANA_GEMMA_REPO_ID,
            revision=SANA_GEMMA_REVISION,
            files=SANA_TOKENIZER_FILES,
            allow_patterns=SANA_TOKENIZER_FILES,
        ),
        "codec": CheckpointSpec(
            repo_id=SANA_WM_REPO_ID,
            revision=SANA_WM_REVISION,
            files=(
                "vae/diffusion_pytorch_model.safetensors",
                "vae/config.json",
            ),
        ),
    }
    return NativeDiffusionRecipe(
        model_id="sana-wm",
        aliases=("sana-wm-2.6b", SANA_WM_REPO_ID),
        components=(
            ComponentSpec(denoiser, build_sana_world_denoiser, {"weights": "dit"}),
            ComponentSpec(
                conditioner,
                build_sana_prompt_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"max_length": 300},
            ),
            ComponentSpec(initializer, build_sana_world_initializer),
            ComponentSpec(scheduler, build_sana_flow_match_scheduler, options={"shift": 9.8}),
            ComponentSpec(
                codec,
                build_diffusers_ltx2_tensor_video_codec,
                {"weights": "codec"},
                {
                    "tiled": True,
                    "spatial_tile_size": 768,
                    "spatial_overlap": 64,
                    "temporal_tile_size": 96,
                    "temporal_overlap": 24,
                },
            ),
        ),
        execution=_execution(encoded_initialization=True),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {"image-to-video", "camera-control", "interactive-world-model", "classifier-free-guidance"}
        ),
        options={"latent_channels": 128, "spatial_compression": 32, "temporal_compression": 8},
        metadata={
            "architecture": "sana-world-camctrl",
            "parameter_scale": "2.6B",
            "resolution": "704x1280",
            "native_inference": True,
            "output_layout": "BCTHW",
        },
    )


def sana_wm_streaming_recipe() -> NativeDiffusionRecipe:
    """Chunk-causal SANA-WM Stage-1 plus the released LTX-2 refiner/VAE."""

    denoiser, conditioner, initializer, scheduler, codec = _keys()
    refiner = ComponentKey(ComponentKind.LATENT_PROCESSOR, "refiner")
    refiner_conditioner = ComponentKey(ComponentKind.CONDITIONER, "refiner")
    repo_id = SANA_WM_STREAMING_REPO_ID
    revision = SANA_WM_STREAMING_REVISION
    gemma_files = (
        "gemma3_12b/model.safetensors.index.json",
        "gemma3_12b/model-00001-of-00005.safetensors",
        "gemma3_12b/model-00002-of-00005.safetensors",
        "gemma3_12b/model-00003-of-00005.safetensors",
        "gemma3_12b/model-00004-of-00005.safetensors",
        "gemma3_12b/model-00005-of-00005.safetensors",
    )
    checkpoints = {
        "dit": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("sana_dit/model.pt",),
            metadata={"license": "Apache-2.0"},
        ),
        "text-encoder": CheckpointSpec(
            repo_id=SANA_GEMMA_REPO_ID,
            revision=SANA_GEMMA_REVISION,
            files=("gemma-2-2b-it.safetensors",),
            allow_patterns=("config.json", "gemma-2-2b-it.safetensors"),
        ),
        "tokenizer": CheckpointSpec(
            repo_id=SANA_GEMMA_REPO_ID,
            revision=SANA_GEMMA_REVISION,
            files=SANA_TOKENIZER_FILES,
            allow_patterns=SANA_TOKENIZER_FILES,
        ),
        "codec": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("ltx2_causal_vae/diffusion_pytorch_model.safetensors",),
            allow_patterns=("ltx2_causal_vae/config.json",),
        ),
        "refiner": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("refiner_diffusers/transformer/diffusion_pytorch_model.safetensors",),
            allow_patterns=("refiner_diffusers/transformer/config.json",),
        ),
        "refiner-connectors": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=("refiner_diffusers/connectors/diffusion_pytorch_model.safetensors",),
            allow_patterns=("refiner_diffusers/connectors/config.json",),
        ),
        "refiner-text-encoder": CheckpointSpec(
            repo_id=repo_id,
            revision=revision,
            files=gemma_files,
            allow_patterns=(
                "gemma3_12b/*.json",
                "gemma3_12b/tokenizer.model",
            ),
        ),
    }
    return NativeDiffusionRecipe(
        model_id=SANA_WM_STREAMING_MODEL_ID,
        aliases=(repo_id,),
        components=(
            ComponentSpec(
                denoiser,
                build_sana_world_streaming_denoiser,
                {"weights": "dit"},
            ),
            ComponentSpec(
                conditioner,
                build_sana_prompt_conditioner,
                {"weights": "text-encoder", "tokenizer": "tokenizer"},
                {"max_length": 300},
            ),
            ComponentSpec(initializer, build_sana_world_initializer),
            ComponentSpec(scheduler, build_sana_wm_streaming_euler_scheduler),
            ComponentSpec(
                codec,
                build_diffusers_ltx2_tensor_video_codec,
                {"weights": "codec"},
                {"tiled": True},
            ),
            ComponentSpec(
                refiner,
                build_sana_wm_ltx2_refiner_processor,
                {"weights": "refiner"},
                {"sigmas": (0.909375, 0.725, 0.421875, 0.0)},
            ),
            ComponentSpec(
                refiner_conditioner,
                build_sana_wm_refiner_conditioner,
                {
                    "weights": "refiner-text-encoder",
                    "connectors": "refiner-connectors",
                },
                {"max_length": 1024},
            ),
        ),
        execution=ExecutionSpec(
            strategy="prefix-recompute",
            bindings={
                "denoiser": denoiser,
                "conditioner": conditioner,
                "latent_initializer": initializer,
                "latent_encoder": codec,
                "scheduler": scheduler,
                "decoder": codec,
                "refiner": refiner,
                "refiner_conditioner": refiner_conditioner,
            },
            options={
                "guidance_mode": "standard",
                "chunk_size": 3,
                "history_frames": 6,
                "sink_frames": 1,
                "refiner_max_frames": 11,
            },
        ),
        checkpoints=checkpoints,
        capabilities=frozenset(
            {
                "image-to-video",
                "camera-control",
                "interactive-world-model",
                "causal-generation",
                "latent-refinement",
            }
        ),
        options={
            "latent_channels": 128,
            "spatial_compression": 32,
            "temporal_compression": 8,
            "default_height": 704,
            "default_width": 1280,
            "default_num_frames": 241,
            "default_num_inference_steps": 4,
            "default_guidance_scale": 1.0,
            "default_fps": 16,
        },
        metadata={
            "architecture": "sana-world-streaming-camctrl",
            "parameter_scale": "2.6B+LTX2",
            "resolution": "704x1280",
            "native_inference": True,
            "output_layout": "BCTHW",
            "latent_chunk_size": 3,
            "refiner_max_frames": 11,
            "repository_revision": revision,
            "upstream_flashdreams_revision": "289da6f1d232de5abaa30d686c977b9c0040fe76",
        },
    )


__all__ = [
    "SANA_WM_STREAMING_MODEL_ID",
    "SANA_WM_STREAMING_REPO_ID",
    "SANA_WM_STREAMING_REVISION",
    "sana_recipe",
    "sana_wm_streaming_recipe",
    "sana_world_recipe",
]
