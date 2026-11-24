"""
Model loader for LTX-2 trainer using the new ltx-core package.
This module provides a unified interface for loading LTX-2 model components
for training, using SingleGPUModelBuilder from ltx-core.
Example usage:
    # Load individual components
    vae_encoder = load_video_vae_encoder("/path/to/checkpoint.safetensors", device="cuda")
    vae_decoder = load_video_vae_decoder("/path/to/checkpoint.safetensors", device="cuda")
    text_encoder = load_text_encoder("/path/to/gemma", device="cuda")
    # Load all components from a monolith (or pass component paths for a split pack)
    components = load_model("/path/to/checkpoint.safetensors", text_encoder_path="/path/to/gemma")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import SafetensorError

from . import logger

# Type alias for device specification
Device = str | torch.device

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model.models.encoders.ltx.embeddings_processor import EmbeddingsProcessor
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.solarwm import SolarWMModel as LTXModel
    from worldfoundry.base_models.diffusion_model.schedulers.ltx_sigma import LTX2Scheduler

    from ..ltx_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from ..ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ..ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
    from ..ltx_core.types import SpatioTemporalScaleFactors


def _to_torch_device(device: Device) -> torch.device:
    """Convert device specification to torch.device."""
    return torch.device(device) if isinstance(device, str) else device


def read_video_scale_factors(model_path: str | Path) -> SpatioTemporalScaleFactors:
    """Read the video VAE compression factors from a checkpoint's embedded config.
    The VAE is not always loaded where these factors are needed (during training the
    latents are precomputed), so they are derived from the checkpoint metadata via
    ``SpatioTemporalScaleFactors.from_model_config``, which yields the default 32x32x8
    layout for checkpoints whose config carries no VAE block list (e.g. audio-only).
    A failure to *read* the metadata is deliberately not swallowed: it means the
    checkpoint path is wrong or the file is corrupt, and silently falling back to the
    default layout would train a non-default VAE (e.g. 16x16x4) on the wrong RoPE
    positions without ever surfacing the error. Fail loudly instead.
    """
    from ..ltx_core.loader import SafetensorsModelStateDictLoader
    from ..ltx_core.types import SpatioTemporalScaleFactors

    metadata = SafetensorsModelStateDictLoader().metadata(str(model_path))
    return SpatioTemporalScaleFactors.from_model_config(metadata.get("config", {}))


def is_split_transformer(checkpoint_path: str | Path) -> bool:
    """Whether ``checkpoint_path`` is the transformer file of a split checkpoint pack.
    A split pack ships one safetensors per component, and the transformer's metadata
    declares only ``transformer``/``scheduler``. Every monolith declares at least one
    VAE section — including audio-only checkpoints, which have ``audio_vae`` but no
    ``vae`` — so the absence of all of them is what identifies the split layout.
    Answers ``False`` for a checkpoint whose metadata cannot be read at all: that is not
    a layout question, and the caller is about to open the same file for real and raise
    with the actual reason.
    """
    from ..ltx_core.loader import SafetensorsModelStateDictLoader

    try:
        metadata = SafetensorsModelStateDictLoader().metadata(str(checkpoint_path))
    except (OSError, SafetensorError):
        return False
    config = metadata.get("config", {})
    sections = set(config) if isinstance(config, dict) else set()
    return "transformer" in sections and not sections & {"vae", "audio_vae", "vocoder"}


def resolve_video_vae_path(model_path: str | Path, video_vae_path: str | Path | None = None) -> str:
    """Return the file holding the video VAE (encoder, decoder and its scale factors)."""
    return _resolve_component_path(model_path, video_vae_path, "video VAE", "video_vae_path")


def resolve_audio_vae_path(model_path: str | Path, audio_vae_path: str | Path | None = None) -> str:
    """Return the file holding the audio VAE and vocoder."""
    return _resolve_component_path(model_path, audio_vae_path, "audio VAE", "audio_vae_path")


def embedding_weight_paths(
    transformer_path: str | Path,
    text_encoder_path: str | Path | None,
) -> str | tuple[str, str]:
    """Return the files that contain the embeddings-processor weights.
    Legacy monoliths store connectors and feature-extractor weights with the
    transformer. LTX-2.5 split packs move the text projections into the packed
    text-encoder safetensors, so both files must be read.
    """
    from ..ltx_core.text_encoders.gemma.gemma_assets import is_safetensors_file

    transformer_path = str(transformer_path)
    if text_encoder_path is not None and is_safetensors_file(text_encoder_path):
        return transformer_path, str(text_encoder_path)
    if is_split_transformer(transformer_path):
        raise ValueError(
            f"{transformer_path} is the transformer of a split checkpoint pack, whose "
            f"text_embedding_projection weights live in the packed text-encoder file. "
            f"Set text_encoder_path to that .safetensors instead of a Gemma directory."
        )
    return transformer_path


def _resolve_component_path(
    model_path: str | Path,
    component_path: str | Path | None,
    component: str,
    flag: str,
) -> str:
    """Resolve a component file, refusing to fall back to a split pack's transformer.
    Falling back would build the component from a checkpoint that holds neither its
    config nor its weights. That is not a hard failure downstream — the loader logs an
    "Uninitialized parameters" warning and hands back a model still on the meta device —
    so it has to be rejected here, where the cause is still visible.
    """
    if component_path is not None:
        return str(component_path)
    if is_split_transformer(model_path):
        raise ValueError(
            f"{model_path} is the transformer of a split checkpoint pack and carries no "
            f"{component}. Pass the standalone {component} safetensors via {flag}."
        )
    return str(model_path)


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LTXModel:
    """Load the LTX transformer model.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded LTXModel transformer
    """
    from ..ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ..ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> VideoEncoder:
    """Load the video VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoEncoder
    """
    from ..ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ..ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> VideoDecoder:
    """Load the video VAE decoder (for inference/validation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoDecoder
    """
    from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.loader.video_decoder import (
        build_cutlass_fna_diffusion_decoder_op,
    )

    from ..ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ..ltx_core.model.video_vae import (
        VideoDecoderConfigurator,
        is_diffusion_video_vae,
        video_decoder_sd_ops_for_checkpoint,
    )

    checkpoint_path = str(checkpoint_path)

    # ChunkedEager ModuleOps: deferred stage-4 + W-chunking + cutlass-fna; lower
    # peak VRAM than CombinedCompile during validation decode.
    diffusion_vae = is_diffusion_video_vae(checkpoint_path)
    module_ops = (build_cutlass_fna_diffusion_decoder_op(),) if diffusion_vae else ()

    return SingleGPUModelBuilder(
        model_path=checkpoint_path,
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(
            checkpoint_path,
            diffusion_vae=diffusion_vae,
        ),
        module_ops=module_ops,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AudioEncoder:
    """Load the audio VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights (default bfloat16, but float32 recommended for quality)
    Returns:
        Loaded AudioEncoder
    """
    from ..ltx_core.loader import SingleGPUModelBuilder
    from ..ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AudioDecoder:
    """Load the audio VAE decoder.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded AudioDecoder
    """
    from ..ltx_core.loader import SingleGPUModelBuilder
    from ..ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Vocoder:
    """Load the vocoder (for audio waveform generation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded Vocoder
    """
    from ..ltx_core.loader import SingleGPUModelBuilder
    from ..ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> LTXGemmaTextEncoder:
    """Load the Gemma text encoder.
    Args:
        gemma_model_path: Gemma model directory, or the packed text-encoder safetensors
            of a split pack
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to quantize the Gemma backbone to 8-bit with bitsandbytes.
            The full-precision weights are materialized on CPU first, so this trades host
            RAM for roughly half the VRAM.
    Returns:
        Loaded LTXGemmaTextEncoder
    """
    from ..ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ..ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, get_gemma_ops
    from ..ltx_core.text_encoders.gemma.gemma_assets import resolve_gemma_weight_paths

    torch_device = _to_torch_device(device)

    gemma_weight_paths = resolve_gemma_weight_paths(str(gemma_model_path))
    gemma_sd_ops, gemma_module_ops = get_gemma_ops(str(gemma_model_path))

    # Quantization needs the weights materialized before it can convert them, and doing
    # that on the target GPU would defeat the point of asking for 8-bit.
    build_device = torch.device("cpu") if load_in_8bit else torch_device

    text_encoder = SingleGPUModelBuilder(
        model_path=tuple(gemma_weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(gemma_model_path)),
        model_sd_ops=gemma_sd_ops,
        module_ops=gemma_module_ops,
    ).build(device=build_device, dtype=dtype)

    if load_in_8bit:
        from .gemma_8bit import quantize_gemma_to_8bit

        return quantize_gemma_to_8bit(text_encoder, device=torch_device)

    return text_encoder


def load_embeddings_processor(
    checkpoint_path: str | Path | Sequence[str | Path],
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> EmbeddingsProcessor:
    """Load the embeddings processor (feature extractor + video/audio connectors).
    Args:
        checkpoint_path: LTX transformer checkpoint, or transformer + packed
            text-encoder files for a split pack.
        gemma_model_path: Path to the Gemma model directory or packed
            text-encoder safetensors. Used to read the
            text encoder's hidden_size / num_hidden_layers when sizing the
            feature-extractor projections.
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded EmbeddingsProcessor with feature extractor and connectors
    """
    from ..ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ..ltx_core.text_encoders.gemma import EMBEDDINGS_PROCESSOR_KEY_OPS, EmbeddingsProcessorConfigurator

    torch_device = _to_torch_device(device)

    paths = (
        tuple(str(path) for path in checkpoint_path)
        if isinstance(checkpoint_path, (list, tuple))
        else str(checkpoint_path)
    )
    return SingleGPUModelBuilder(
        model_path=paths,
        model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(str(gemma_model_path)),
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    ).build(device=torch_device, dtype=dtype)


# =============================================================================
# Combined Component Loader
# =============================================================================


@dataclass
class LtxModelComponents:
    """Container for all LTX-2 model components."""

    transformer: LTXModel
    video_vae_encoder: VideoEncoder | None = None
    video_vae_decoder: VideoDecoder | None = None
    audio_vae_decoder: AudioDecoder | None = None
    vocoder: Vocoder | None = None
    text_encoder: LTXGemmaTextEncoder | None = None
    scheduler: LTX2Scheduler | None = None


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    video_vae_path: str | Path | None = None,
    audio_vae_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
) -> LtxModelComponents:
    """
    Load LTX-2 model components from a safetensors checkpoint.
    This is a convenience function that loads multiple components at once.
    For loading individual components, use the dedicated functions:
    - load_transformer()
    - load_video_vae_encoder()
    - load_video_vae_decoder()
    - load_audio_vae_decoder()
    - load_vocoder()
    - load_text_encoder()
    Args:
        checkpoint_path: Unified checkpoint or split transformer safetensors.
        text_encoder_path: Gemma directory or packed text-encoder safetensors
            (required if with_text_encoder=True).
        video_vae_path: Video VAE safetensors for a split pack. Defaults to
            ``checkpoint_path`` for a monolith.
        audio_vae_path: Audio VAE/vocoder safetensors for a split pack. Defaults
            to ``checkpoint_path`` for a monolith.
        device: Device to load models on ("cuda", "cpu", etc.)
        dtype: Data type for model weights
        with_video_vae_encoder: Whether to load the video VAE encoder (for preprocessing)
        with_video_vae_decoder: Whether to load the video VAE decoder (for inference/validation)
        with_audio_vae_decoder: Whether to load the audio VAE decoder
        with_vocoder: Whether to load the vocoder
        with_text_encoder: Whether to load the text encoder
    Returns:
        LtxModelComponents containing all loaded model components
    """
    from worldfoundry.base_models.diffusion_model.schedulers.ltx_sigma import LTX2Scheduler

    checkpoint_path = Path(checkpoint_path)

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    needs_video_vae = with_video_vae_encoder or with_video_vae_decoder
    needs_audio_vae = with_audio_vae_decoder or with_vocoder
    if needs_video_vae:
        video_vae_path = resolve_video_vae_path(checkpoint_path, video_vae_path)
    if needs_audio_vae:
        audio_vae_path = resolve_audio_vae_path(checkpoint_path, audio_vae_path)

    logger.info(f"Loading LTX-2 model from {checkpoint_path}")

    torch_device = _to_torch_device(device)

    # Load transformer
    logger.debug("Loading transformer...")
    transformer = load_transformer(checkpoint_path, torch_device, dtype)

    # Load video VAE encoder
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(video_vae_path, torch_device, dtype)

    # Load video VAE decoder
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(video_vae_path, torch_device, dtype)

    # Load audio VAE decoder
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(audio_vae_path, torch_device, dtype)

    # Load vocoder
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(audio_vae_path, torch_device, dtype)

    # Load text encoder
    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(text_encoder_path, torch_device, dtype)

    # Create scheduler (stateless, no loading needed)
    scheduler = LTX2Scheduler()

    return LtxModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
