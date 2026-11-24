"""LTX video / audio codec adapters for the native runner contracts.

Loader wrappers (:class:`LTXVideoEncoderModule`, decoder, vocoder) own
checkpoint config + key prefixes.  Runner-facing types:

- :class:`LTXMediaDecoder` (:class:`~...runners.MultiModalLatentDecoder`):
  unpatchify video+audio, decode pixels, vocode waveform.
- :class:`LTXVideoMediaDecoder`: video-only multimodal decode.
- :class:`LTXTensorVideoCodec`: :class:`~...contracts.LatentEncoder` +
  :class:`~...contracts.LatentDecoder` for BCTHW tensors (SANA-WM).
- :class:`DiffusersLTX2TensorVideoCodec`: same contract over a Diffusers
  LTX-2 VAE.

``DiffusionRequest`` supplies FPS / ``return_latent``; tiling comes from
component options.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as functional

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, ModalityState
from ....loaders import (
    CheckpointSpec,
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
    safetensors_json_metadata,
)
from ....optimizations import OffloadMode, QuantizationPolicy, RuntimePolicy
from ....runners import MultiModalLatentDecoder
from ...representations.ltx.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ...representations.ltx.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from .audio import AudioDecoderConfigurator, VocoderConfigurator, decode_audio
from .audio.vocoder import Snake, SnakeBeta, VocoderWithBWE
from .video import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)


class LTXVideoEncoderModule(torch.nn.Module):
    """Native-loader wrapper around the configured LTX video encoder CNN."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.encoder = VideoEncoderConfigurator.from_config(dict(checkpoint_config))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Encode BCTHW pixels to video latents."""
        return self.encoder(sample)


class LTXVideoDecoderModule(torch.nn.Module):
    """Native-loader wrapper around the configured LTX video decoder CNN."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.decoder = VideoDecoderConfigurator.from_config(dict(checkpoint_config))


class LTXAudioDecoderModule(torch.nn.Module):
    """Native-loader wrapper around the LTX audio VAE decoder."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.decoder = AudioDecoderConfigurator.from_config(dict(checkpoint_config))


class LTXVocoderModule(torch.nn.Module):
    """Native-loader wrapper around the LTX waveform vocoder."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.vocoder = VocoderConfigurator.from_config(dict(checkpoint_config))


def _prefixed_state_dict(
    state_dict: Mapping[str, object],
    *,
    prefixes: Mapping[str, str],
    lazy: bool = False,
) -> Mapping[str, object]:
    converted: dict[str, object] = {}
    for key in state_dict:
        for source, destination in prefixes.items():
            if key.startswith(source):
                suffix = key.removeprefix(source)
                if "per_channel_statistics." in destination and suffix not in {
                    "mean-of-means",
                    "std-of-means",
                }:
                    break
                converted[f"{destination}{suffix}"] = key if lazy else state_dict[key]
                break
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        return TransformedStateDict(state_dict, converted)
    return converted


def convert_ltx_video_encoder_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    return _prefixed_state_dict(
        state_dict,
        prefixes={
            "vae.encoder.": "encoder.",
            "vae.per_channel_statistics.": "encoder.per_channel_statistics.",
            "encoder.": "encoder.",
            "per_channel_statistics.": "encoder.per_channel_statistics.",
        },
        lazy=lazy,
    )


def convert_ltx_video_decoder_state_dict(
    state_dict: Mapping[str, object], *, lazy: bool = False
) -> Mapping[str, object]:
    return _prefixed_state_dict(
        state_dict,
        prefixes={
            "vae.decoder.": "decoder.",
            "vae.per_channel_statistics.": "decoder.per_channel_statistics.",
        },
        lazy=lazy,
    )


def convert_ltx_audio_decoder_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    return _prefixed_state_dict(
        state_dict,
        prefixes={
            "audio_vae.decoder.": "decoder.",
            "audio_vae.per_channel_statistics.": "decoder.per_channel_statistics.",
        },
        lazy=lazy,
    )


def convert_ltx_vocoder_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    return _prefixed_state_dict(state_dict, prefixes={"vocoder.": "vocoder."}, lazy=lazy)


def _load_configured_module(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
    *,
    module_class: type[torch.nn.Module],
    converter,
    config: Mapping[str, object] | None = None,
    lazy_state_dict: bool = False,
) -> torch.nn.Module:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    return NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            config=config or {},
            lazy_state_dict=lazy_state_dict,
            buffer_aware_offload=lazy_state_dict,
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=converter,
            vram_module_map=(
                {module_class: AutoWrappedModule}
                if lazy_state_dict and module_class is LTXDiffusionVideoDecoderModule
                else {
                    Snake: AutoWrappedModule,
                    SnakeBeta: AutoWrappedModule,
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.ConvTranspose1d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.GroupNorm: AutoWrappedModule,
                }
            ),
        ),
        checkpoint,
        replace(policy, quantization=QuantizationPolicy()),
    )


def load_ltx_video_encoder(context: ComponentBuildContext) -> LTXVideoEncoderModule:
    if context.component_options.get("split_checkpoint"):
        return _load_configured_module(
            context.require_checkpoint("weights"),
            context.policy,
            module_class=LTXVideoEncoderModule,
            converter=partial(convert_ltx_video_encoder_state_dict, lazy=True),
            lazy_state_dict=True,
        )
    return load_ltx_video_encoder_checkpoint(context.require_checkpoint("weights"), context.policy)


def load_ltx_video_encoder_checkpoint(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> LTXVideoEncoderModule:
    module = _load_configured_module(
        checkpoint,
        policy,
        module_class=LTXVideoEncoderModule,
        converter=convert_ltx_video_encoder_state_dict,
    )
    if not isinstance(module, LTXVideoEncoderModule):
        raise TypeError(f"expected LTXVideoEncoderModule, got {type(module).__name__}")
    return module


def load_ltx_video_decoder_checkpoint(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> LTXVideoDecoderModule:
    module = _load_configured_module(
        checkpoint,
        policy,
        module_class=LTXVideoDecoderModule,
        converter=convert_ltx_video_decoder_state_dict,
    )
    if not isinstance(module, LTXVideoDecoderModule):
        raise TypeError(f"expected LTXVideoDecoderModule, got {type(module).__name__}")
    return module


class LTXMediaDecoder(MultiModalLatentDecoder):
    """Decode final LTX modality states into video and synchronized audio."""

    def __init__(
        self,
        video: LTXVideoDecoderModule,
        audio: LTXAudioDecoderModule,
        vocoder: LTXVocoderModule,
        *,
        compute_dtype: torch.dtype,
        tiling: TilingConfig | None,
        precision: str = "autocast",
        spatial_divisor: int = 1,
        strip_reference_tokens: bool = False,
        continue_generator: bool = False,
        skip_redundant_video_autocast: bool = False,
        parallel_audio: bool = False,
        audio_cuda_graph: bool = False,
        compile_vocoder: bool = False,
    ) -> None:
        self.precision = precision
        self.spatial_divisor = spatial_divisor
        self.strip_reference_tokens = strip_reference_tokens
        self.uses_run_generator = continue_generator
        self.video = video
        self.audio = audio
        self.vocoder = vocoder
        self.compute_dtype = compute_dtype
        self.tiling = tiling
        self.skip_redundant_video_autocast = bool(skip_redundant_video_autocast)
        self.parallel_audio = bool(parallel_audio)
        self._audio_streams: dict[int, torch.cuda.Stream] = {}
        self._compiled_vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None
        if compile_vocoder and torch.cuda.is_available():
            from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

            configure_persistent_compile_cache(namespace="ltx-vocoder")
            self._compiled_vocoder = torch.compile(self.vocoder.vocoder, mode="default")
        self._audio_graph_runner = None
        if audio_cuda_graph:
            from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

            self._audio_graph_runner = InferenceCUDAGraphRunner(
                self._run_audio_pipeline,
                enabled=True,
                extra_key=(
                    f"ltx_audio_decode:{id(self.audio)}:{id(self.vocoder)}:"
                    f"{precision}:{self._compiled_vocoder is not None}"
                ),
            )

    def _run_audio_pipeline(self, lat: torch.Tensor) -> torch.Tensor:
        """Decode audio latents -> mel spectrogram -> waveform."""
        use_autocast = self.precision == "autocast" and self.compute_dtype in {torch.float16, torch.bfloat16}
        if self._compiled_vocoder is not None and lat.device.type == "cuda" and not torch.is_grad_enabled():
            if isinstance(self.vocoder.vocoder, VocoderWithBWE):
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.compute_dtype,
                    enabled=use_autocast,
                ):
                    decoded_mel = self.audio.decoder(lat)
                # Run compiled VocoderWithBWE outside outer BF16 autocast so its internal
                # FP32 autocast compiles cleanly without nested-autocast Inductor drift.
                return self._compiled_vocoder(decoded_mel).squeeze(0).float()
            with torch.autocast(
                device_type="cuda",
                dtype=self.compute_dtype,
                enabled=use_autocast,
            ):
                decoded_mel = self.audio.decoder(lat)
                return self._compiled_vocoder(decoded_mel).squeeze(0).float()
        with torch.autocast(
            device_type=lat.device.type,
            dtype=self.compute_dtype,
            enabled=use_autocast,
        ):
            return decode_audio(lat, self.audio.decoder, self.vocoder.vocoder).waveform

    def _get_audio_stream(self, device: torch.device) -> torch.cuda.Stream:
        idx = 0 if device.index is None else int(device.index)
        stream = self._audio_streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._audio_streams[idx] = stream
        return stream

    def _decode_audio_waveform(self, audio_in: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self._audio_graph_runner is not None and audio_in.device.type == "cuda" and not torch.is_grad_enabled():
            waveform = self._audio_graph_runner(audio_in)
            return waveform, int(self.vocoder.vocoder.output_sampling_rate)
        if self._compiled_vocoder is not None and audio_in.device.type == "cuda" and not torch.is_grad_enabled():
            waveform = self._run_audio_pipeline(audio_in)
            return waveform, int(self.vocoder.vocoder.output_sampling_rate)
        with torch.autocast(
            device_type=audio_in.device.type,
            dtype=self.compute_dtype,
            enabled=self.precision == "autocast" and self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            decoded_audio = decode_audio(audio_in, self.audio.decoder, self.vocoder.vocoder)
        return decoded_audio.waveform, int(decoded_audio.sampling_rate)

    def _use_video_autocast(self, device_type: str) -> bool:
        if self.precision != "autocast" or self.compute_dtype not in {torch.float16, torch.bfloat16}:
            return False
        # Only skip redundant autocast for uniform-BF16 deterministic decoders.
        # FP16 MUST retain autocast so PixelNorm's x**2 runs in FP32 and cannot
        # overflow 65504 to +inf (silent all-zero collapse).
        if (
            self.skip_redundant_video_autocast
            and device_type == "cuda"
            and self.compute_dtype == torch.bfloat16
            and not bool(getattr(self.video.decoder, "timestep_conditioning", False))
        ):
            params = tuple(self.video.decoder.parameters())
            if params and all(p.dtype == torch.bfloat16 for p in params):
                return False
        return True

    @staticmethod
    def _shapes(request: DiffusionRequest) -> tuple[VideoLatentShape, AudioLatentShape]:
        fps = float(request.inputs.get("frame_rate", 24.0))
        pixels = VideoPixelShape(
            batch=request.batch_size,
            frames=request.num_frames,
            height=request.height,
            width=request.width,
            fps=fps,
        )
        return VideoLatentShape.from_pixel_shape(pixels), AudioLatentShape.from_video_pixel_shape(pixels)

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> Mapping[str, object]:
        """Unpatchify, decode video chunks, and vocode audio from joint states."""
        video_shape, audio_shape = self._shapes(
            replace(request, height=request.height // self.spatial_divisor, width=request.width // self.spatial_divisor)
            if self.spatial_divisor != 1
            else request
        )
        patchifier = VideoLatentPatchifier(1)
        latent = states["video"].latent
        if self.strip_reference_tokens:
            latent = latent[:, : patchifier.get_token_count(video_shape)]
        video_latent = patchifier.unpatchify(latent, video_shape)
        audio_latent = AudioPatchifier(1).unpatchify(states["audio"].latent, audio_shape)

        video_in = video_latent.to(dtype=self.compute_dtype)
        audio_in = audio_latent.to(dtype=self.compute_dtype)
        video_gen = (
            generator
            if generator is not None
            else torch.Generator(device=video_latent.device).manual_seed(request.sampling.seed)
        )
        use_video_autocast = self._use_video_autocast(video_latent.device.type)

        if self.parallel_audio and video_latent.device.type == "cuda" and not torch.is_grad_enabled():
            main_stream = torch.cuda.current_stream(video_latent.device)
            audio_stream = self._get_audio_stream(video_latent.device)
            audio_stream.wait_stream(main_stream)
            audio_in.record_stream(audio_stream)
            audio_waveform: torch.Tensor | None = None
            try:
                with torch.cuda.stream(audio_stream):
                    audio_waveform, audio_sampling_rate = self._decode_audio_waveform(audio_in)
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.compute_dtype,
                    enabled=use_video_autocast,
                ):
                    chunks = list(self.video.decoder.decode_video(video_in, self.tiling, generator=video_gen))
            finally:
                main_stream.wait_stream(audio_stream)
                if audio_waveform is not None:
                    audio_waveform.record_stream(main_stream)
        else:
            with torch.autocast(
                device_type=video_latent.device.type,
                dtype=self.compute_dtype,
                enabled=use_video_autocast,
            ):
                chunks = list(self.video.decoder.decode_video(video_in, self.tiling, generator=video_gen))
            audio_waveform, audio_sampling_rate = self._decode_audio_waveform(audio_in)

        if not chunks:
            raise RuntimeError("LTX video decoder returned no chunks")
        return {
            "video": chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0),
            "audio": audio_waveform,
            "audio_sampling_rate": audio_sampling_rate,
        }


class LTXVideoMediaDecoder(MultiModalLatentDecoder):
    """Decode the video-only LTX-Video latent state."""

    def __init__(
        self,
        video: LTXVideoDecoderModule,
        *,
        tiling: TilingConfig | None,
        first_stage_scale: float | None = None,
        latent_upsample_factor: int = 1,
        tone_map_compression: float = 0.0,
        compute_dtype: torch.dtype | None = None,
        decode_autocast: bool = False,
    ) -> None:
        self.video = video
        self.tiling = tiling
        if first_stage_scale is not None and not 0.0 < float(first_stage_scale) <= 1.0:
            raise ValueError("LTX first-stage scale must be in (0, 1]")
        if int(latent_upsample_factor) < 1:
            raise ValueError("LTX latent upsample factor must be positive")
        if not 0.0 <= float(tone_map_compression) <= 1.0:
            raise ValueError("LTX tone-map compression must be in [0, 1]")
        self.first_stage_scale = None if first_stage_scale is None else float(first_stage_scale)
        self.latent_upsample_factor = int(latent_upsample_factor)
        self.tone_map_compression = float(tone_map_compression)
        self.compute_dtype = compute_dtype
        self.decode_autocast = bool(decode_autocast)

    @staticmethod
    def _round_down_to_vae(value: int) -> int:
        result = int(value) - int(value) % 32
        if result < 32:
            raise ValueError("LTX stage dimensions must be at least 32 pixels")
        return result

    def _latent_pixel_size(self, request: DiffusionRequest) -> tuple[int, int]:
        if self.first_stage_scale is None:
            return request.height, request.width
        return (
            self._round_down_to_vae(int(request.height * self.first_stage_scale)) * self.latent_upsample_factor,
            self._round_down_to_vae(int(request.width * self.first_stage_scale)) * self.latent_upsample_factor,
        )

    def _tone_map(self, latent: torch.Tensor) -> torch.Tensor:
        if self.tone_map_compression == 0.0:
            return latent
        scale = self.tone_map_compression * 0.75
        sigmoid = torch.sigmoid(4.0 * scale * (latent.abs() - 1.0))
        return latent * (1.0 - 0.8 * scale * sigmoid)

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, object]:
        latent_height, latent_width = self._latent_pixel_size(request)
        pixels = VideoPixelShape(
            batch=request.batch_size,
            frames=request.num_frames,
            height=latent_height,
            width=latent_width,
            fps=float(request.inputs.get("frame_rate", 24.0)),
        )
        video_shape = VideoLatentShape.from_pixel_shape(pixels)
        video_latent = VideoLatentPatchifier(1).unpatchify(states["video"].latent, video_shape)
        video_latent = self._tone_map(video_latent)
        # The timestep-conditioned VAE injects noise too. Keep it request-local
        # so reusing a pipeline or unrelated global RNG draws cannot change RGB.
        generator = torch.Generator(device=video_latent.device).manual_seed(request.sampling.seed)
        autocast_dtype = self.compute_dtype or video_latent.dtype
        with torch.autocast(
            device_type=video_latent.device.type,
            dtype=autocast_dtype,
            enabled=self.decode_autocast and autocast_dtype in {torch.float16, torch.bfloat16},
        ):
            chunks = list(self.video.decoder.decode_video(video_latent, self.tiling, generator=generator))
        if not chunks:
            raise RuntimeError("LTX-Video decoder returned no chunks")
        video = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        if (latent_height, latent_width) != (request.height, request.width):
            dtype = video.dtype
            video = (
                functional.interpolate(
                    video.permute(0, 3, 1, 2).float(),
                    size=(request.height, request.width),
                    mode="bilinear",
                    align_corners=False,
                )
                .to(dtype=dtype)
                .permute(0, 2, 3, 1)
            )
        return {"video": video}


class LTXTensorVideoCodec:
    """BCTHW encoder/decoder role for models that diffuse directly in LTX latents."""

    spatial_compression_factor = 32
    temporal_compression_factor = 8
    latent_ch = 128

    def __init__(
        self,
        encoder: LTXVideoEncoderModule,
        decoder: LTXVideoDecoderModule,
        *,
        tiling: TilingConfig | None,
        compute_device: str | torch.device | None = None,
        compute_dtype: torch.dtype | None = None,
        decode_autocast: bool = False,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.tiling = tiling
        self.compute_device = None if compute_device is None else torch.device(compute_device)
        self.compute_dtype = compute_dtype
        self.decode_autocast = bool(decode_autocast)

    def _input_target(self, module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters())
        return (
            self.compute_device if self.compute_device is not None else parameter.device,
            self.compute_dtype if self.compute_dtype is not None else parameter.dtype,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode BCTHW RGB to 128-channel LTX latents (optional tiling)."""
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError(f"LTX tensor codec expects BCTHW RGB video, got {tuple(images.shape)}")
        device, dtype = self._input_target(self.encoder)
        images = images.to(device=device, dtype=dtype)
        return self.encoder.encoder.tiled_encode(images, self.tiling)

    @torch.no_grad()
    def encode_posterior(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the normalized posterior used by the LTX-Video trainer."""

        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError(f"LTX tensor codec expects BCTHW RGB video, got {tuple(images.shape)}")
        device, dtype = self._input_target(self.encoder)
        images = images.to(device=device, dtype=dtype)
        return self.encoder.encoder.tiled_encode(
            images,
            self.tiling,
            sample_posterior=True,
            generator=generator,
        )

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        """Decode LTX latents to BCTHW pixels; center-crop to ``request`` size."""
        if request is not None and bool(request.inputs.get("return_latent", False)):
            return latents
        device, dtype = self._input_target(self.decoder)
        latents = latents.to(device=device, dtype=dtype)
        generator = (
            None if request is None else torch.Generator(device=latents.device).manual_seed(request.sampling.seed)
        )
        with torch.autocast(
            device_type=latents.device.type,
            dtype=dtype,
            enabled=self.decode_autocast and dtype in {torch.float16, torch.bfloat16},
        ):
            chunks = (
                [self.decoder.decoder(latents, generator=generator)]
                if self.tiling is None
                else list(self.decoder.decoder.tiled_decode(latents, self.tiling, generator=generator))
            )
        if not chunks:
            raise RuntimeError("LTX tensor decoder returned no chunks")
        video = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
        if request is not None:
            decoded_height, decoded_width = video.shape[-2:]
            if decoded_height < request.height or decoded_width < request.width:
                raise ValueError(
                    "LTX tensor decoder output is smaller than the requested crop: "
                    f"{decoded_height}x{decoded_width} < {request.height}x{request.width}"
                )
            if (decoded_height, decoded_width) != (request.height, request.width):
                top = (decoded_height - request.height) // 2
                left = (decoded_width - request.width) // 2
                video = video[
                    ...,
                    top : top + request.height,
                    left : left + request.width,
                ]
        return video.clamp_(-1.0, 1.0)


class DiffusersLTX2TensorVideoCodec:
    """BCTHW codec for official diffusers-format LTX-2 VAE directories."""

    spatial_compression_factor = 32
    temporal_compression_factor = 8
    latent_ch = 128

    def __init__(
        self,
        vae: torch.nn.Module,
        *,
        compute_device: str | torch.device | None = None,
        compute_dtype: torch.dtype | None = None,
        offload_after_call: bool = False,
    ) -> None:
        self.vae = vae
        parameter = next(vae.parameters())
        self.compute_device = torch.device(compute_device or parameter.device)
        self.compute_dtype = compute_dtype or parameter.dtype
        self.offload_after_call = bool(offload_after_call)

    def _activate(self) -> None:
        self.vae.to(device=self.compute_device, dtype=self.compute_dtype)

    def _deactivate(self) -> None:
        if not self.offload_after_call:
            return
        self.vae.to(device="cpu")
        if self.compute_device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.compute_device):
                torch.cuda.empty_cache()

    def _normalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        return (latents - mean) * float(self.vae.config.scaling_factor) / std

    def _denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        return latents * std / float(self.vae.config.scaling_factor) + mean

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self._activate()
        try:
            pixels = images.to(device=self.compute_device, dtype=self.compute_dtype)
            posterior = self.vae.encode(pixels, return_dict=False)[0]
            return self._normalize(posterior.mode())
        finally:
            self._deactivate()

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        if request is not None and bool(request.inputs.get("return_latent", False)):
            return latents
        self._activate()
        try:
            values = self._denormalize(latents.to(device=self.compute_device, dtype=self.compute_dtype))
            video = self.vae.decode(values, return_dict=False)[0]
            if request is not None and video.shape[-2:] != (request.height, request.width):
                decoded_height, decoded_width = video.shape[-2:]
                if decoded_height < request.height or decoded_width < request.width:
                    raise ValueError(
                        "diffusers LTX-2 decoder output is smaller than the requested crop: "
                        f"{decoded_height}x{decoded_width} < {request.height}x{request.width}"
                    )
                top = (decoded_height - request.height) // 2
                left = (decoded_width - request.width) // 2
                video = video[..., top : top + request.height, left : left + request.width]
            return video.clamp_(-1.0, 1.0)
        finally:
            self._deactivate()


def _tiling_from_options(options: Mapping[str, object]) -> TilingConfig | None:
    if options.get("tiled", True) is False:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=int(options.get("spatial_tile_size", 768)),
            tile_overlap_in_pixels=int(options.get("spatial_overlap", 64)),
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=int(options.get("temporal_tile_size", 80)),
            tile_overlap_in_frames=int(options.get("temporal_overlap", 24)),
        ),
    )


def _maybe_enable_vae_channels_last_3d(
    module: torch.nn.Module,
    context: ComponentBuildContext,
) -> None:
    """Apply ``channels_last_3d`` to LTX 3D VAE convolutions when opted in by policy."""
    enabled = bool(
        context.component_options.get(
            "vae_channels_last_3d",
            context.policy.options.get("vae_channels_last_3d", False),
        )
    )
    if not enabled or context.policy.offload.mode is not OffloadMode.NONE:
        return
    converter = getattr(module, "enable_channels_last_3d", None)
    if callable(converter):
        converter()


def _resolve_vae_tiling(context: ComponentBuildContext) -> TilingConfig | None:
    """Resolve VAE tiling, allowing policy ``vae_tiled=False`` to override recipe tiling."""
    if "vae_tiled" in context.component_options:
        if context.component_options["vae_tiled"] is False:
            return None
    elif "vae_tiled" in context.policy.options and context.policy.options["vae_tiled"] is False:
        return None
    return _tiling_from_options(context.component_options)


def _resolve_vae_compile_mode(context: ComponentBuildContext) -> str | None:
    raw = context.component_options.get("vae_compile", context.policy.options.get("vae_compile", False))
    if raw is None or raw is False:
        return None
    if raw is True:
        return str(
            context.component_options.get(
                "vae_compile_mode",
                context.policy.options.get("vae_compile_mode", "default"),
            )
        )
    normalized = str(raw).strip().casefold()
    if normalized in {"", "false", "0", "off", "none", "disabled"}:
        return None
    return str(raw).strip()


def _maybe_enable_vae_compile(
    module: torch.nn.Module,
    context: ComponentBuildContext,
    *,
    dynamic: bool | None = None,
) -> bool:
    """Enable ``channels_last_3d`` + ``torch.compile`` on deterministic LTX 3D VAE decoders."""
    mode = _resolve_vae_compile_mode(context)
    if mode is None or context.policy.offload.mode is not OffloadMode.NONE or context.policy.device.type != "cuda":
        return False
    if not bool(getattr(module, "_channels_last_3d", False)):
        converter = getattr(module, "enable_channels_last_3d", None)
        if callable(converter):
            converter()
    compiler = getattr(module, "enable_compile", None)
    if callable(compiler):
        return bool(compiler(mode=mode, dynamic=dynamic))
    return False


def _resolve_vae_decode_autocast(context: ComponentBuildContext) -> bool:
    return bool(
        context.component_options.get(
            "vae_decode_autocast",
            context.policy.options.get("vae_decode_autocast", False),
        )
    )


def build_ltx_media_decoder(context: ComponentBuildContext) -> LTXMediaDecoder:
    lazy = context.component_options.get("precision") == "weights"
    video = _load_configured_module(
        context.require_checkpoint("weights"),
        context.policy,
        module_class=LTXVideoDecoderModule,
        converter=partial(convert_ltx_video_decoder_state_dict, lazy=True)
        if lazy
        else convert_ltx_video_decoder_state_dict,
        lazy_state_dict=lazy,
    )
    audio = _load_configured_module(
        context.require_checkpoint("weights"),
        context.policy,
        module_class=LTXAudioDecoderModule,
        converter=partial(convert_ltx_audio_decoder_state_dict, lazy=True)
        if lazy
        else convert_ltx_audio_decoder_state_dict,
        lazy_state_dict=lazy,
    )
    vocoder = _load_configured_module(
        context.require_checkpoint("weights"),
        context.policy,
        module_class=LTXVocoderModule,
        converter=partial(convert_ltx_vocoder_state_dict, lazy=True) if lazy else convert_ltx_vocoder_state_dict,
        lazy_state_dict=lazy,
    )
    if not isinstance(video, LTXVideoDecoderModule):
        raise TypeError("LTX video decoder construction returned an unexpected module")
    if not isinstance(audio, LTXAudioDecoderModule):
        raise TypeError("LTX audio decoder construction returned an unexpected module")
    if not isinstance(vocoder, LTXVocoderModule):
        raise TypeError("LTX vocoder construction returned an unexpected module")
    tiling = _resolve_vae_tiling(context)
    _maybe_enable_vae_channels_last_3d(video.decoder, context)
    compiled = _maybe_enable_vae_compile(video.decoder, context, dynamic=(True if tiling is not None else None))
    resident_cuda = context.policy.offload.mode is OffloadMode.NONE and context.policy.device.type == "cuda"
    parallel_audio = bool(
        context.component_options.get(
            "vae_parallel_audio",
            context.policy.options.get("vae_parallel_audio", compiled and resident_cuda),
        )
    )
    audio_cuda_graph = bool(
        context.component_options.get(
            "vae_audio_cuda_graph",
            context.policy.options.get("vae_audio_cuda_graph", compiled and resident_cuda),
        )
    )
    skip_redundant_video_autocast = bool(
        context.component_options.get(
            "vae_skip_redundant_autocast",
            context.policy.options.get("vae_skip_redundant_autocast", compiled),
        )
    )
    compile_vocoder = bool(
        context.component_options.get(
            "vae_compile_vocoder",
            context.policy.options.get("vae_compile_vocoder", compiled and resident_cuda),
        )
    )
    return LTXMediaDecoder(
        video,
        audio,
        vocoder,
        compute_dtype=context.policy.dtype,
        tiling=tiling,
        precision=str(context.component_options.get("precision", "autocast")),
        spatial_divisor=int(context.component_options.get("spatial_divisor", 1)),
        strip_reference_tokens=bool(context.component_options.get("strip_reference_tokens", False)),
        continue_generator=bool(context.component_options.get("continue_generator", False)),
        skip_redundant_video_autocast=skip_redundant_video_autocast,
        parallel_audio=parallel_audio and resident_cuda,
        audio_cuda_graph=audio_cuda_graph and resident_cuda,
        compile_vocoder=compile_vocoder and resident_cuda,
    )


def build_ltx_video_media_decoder(context: ComponentBuildContext) -> LTXVideoMediaDecoder:
    video = load_ltx_video_decoder_checkpoint(
        context.require_checkpoint("weights"),
        context.policy,
    )
    tiling = _resolve_vae_tiling(context)
    _maybe_enable_vae_channels_last_3d(video.decoder, context)
    _maybe_enable_vae_compile(video.decoder, context, dynamic=(True if tiling is not None else None))
    raw_scale = context.component_options.get("first_stage_scale")
    return LTXVideoMediaDecoder(
        video,
        tiling=tiling,
        first_stage_scale=None if raw_scale is None else float(raw_scale),
        latent_upsample_factor=int(context.component_options.get("latent_upsample_factor", 1)),
        tone_map_compression=float(context.component_options.get("tone_map_compression", 0.0)),
        compute_dtype=context.policy.dtype,
        decode_autocast=_resolve_vae_decode_autocast(context),
    )


def build_ltx_tensor_video_codec(context: ComponentBuildContext) -> LTXTensorVideoCodec:
    """Build one shared LTX encoder/decoder pair for standard diffusion roles."""

    checkpoint = context.require_checkpoint("weights")
    encoder = load_ltx_video_encoder_checkpoint(checkpoint, context.policy)
    decoder = load_ltx_video_decoder_checkpoint(checkpoint, context.policy)
    tiling = _resolve_vae_tiling(context)
    _maybe_enable_vae_channels_last_3d(encoder.encoder, context)
    _maybe_enable_vae_channels_last_3d(decoder.decoder, context)
    _maybe_enable_vae_compile(decoder.decoder, context, dynamic=(True if tiling is not None else None))
    return LTXTensorVideoCodec(
        encoder,
        decoder,
        tiling=tiling,
        compute_device=context.policy.device,
        compute_dtype=context.policy.dtype,
        decode_autocast=_resolve_vae_decode_autocast(context),
    )


def build_diffusers_ltx2_tensor_video_codec(
    context: ComponentBuildContext,
) -> DiffusersLTX2TensorVideoCodec:
    """Load the official SANA-WM LTX-2 VAE without network fallback."""

    materialized = NativeCheckpointResolver().materialize(context.require_checkpoint("weights"))
    weight_path = next(
        (path for path in materialized.paths if path.suffix == ".safetensors"),
        None,
    )
    if weight_path is None:
        raise FileNotFoundError("diffusers LTX-2 checkpoint has no safetensors weights")
    component_root = weight_path.parent
    if not (component_root / "config.json").is_file():
        raise FileNotFoundError(f"diffusers LTX-2 config is missing: {component_root / 'config.json'}")

    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        component_root,
        local_files_only=True,
        torch_dtype=context.policy.dtype,
    )
    staged_offload = bool(
        context.policy.device.type != "cpu"
        and context.component_options.get(
            "offload",
            context.policy.offload.mode is not OffloadMode.NONE,
        )
    )
    if not staged_offload:
        vae.to(device=context.policy.device, dtype=context.policy.dtype)
    vae.eval()
    if bool(context.component_options.get("tiled", True)):
        vae.enable_tiling()
    return DiffusersLTX2TensorVideoCodec(
        vae,
        compute_device=context.policy.device,
        compute_dtype=context.policy.dtype,
        offload_after_call=staged_offload,
    )


__all__ = [
    "DiffusersLTX2TensorVideoCodec",
    "LTXMediaDecoder",
    "LTXTensorVideoCodec",
    "LTXVideoEncoderModule",
    "LTXVideoMediaDecoder",
    "build_diffusers_ltx2_tensor_video_codec",
    "build_ltx_media_decoder",
    "build_ltx_tensor_video_codec",
    "build_ltx_video_media_decoder",
    "convert_ltx_audio_decoder_state_dict",
    "convert_ltx_video_decoder_state_dict",
    "convert_ltx_video_encoder_state_dict",
    "convert_ltx_vocoder_state_dict",
    "load_ltx_video_decoder_checkpoint",
    "load_ltx_video_encoder",
    "load_ltx_video_encoder_checkpoint",
]


class LTXDiffusionVideoDecoderModule(torch.nn.Module):
    """Configured CausalDiffusionVAE decoder with its selected inference preset."""

    def __init__(self, checkpoint_config: Mapping[str, object], *, mode: str = "chunked_eager") -> None:
        super().__init__()
        from .video.diffusion_video_decoder import DiffusionVideoDecoderConfigurator
        from .video.transformer.apply import apply_diffvae_mode
        from .video.transformer.config import DiffVAEMode

        if checkpoint_config.get("vae", {}).get("_class_name") != "CausalDiffusionVAE":
            raise ValueError("LTX split diffusion decoding requires a CausalDiffusionVAE checkpoint")
        self.decoder = DiffusionVideoDecoderConfigurator.from_config(dict(checkpoint_config))
        apply_diffvae_mode(self.decoder, DiffVAEMode(mode))

    def forward(
        self,
        latent: torch.Tensor,
        tiling: TilingConfig | None,
        generator: torch.Generator,
    ) -> list[torch.Tensor]:
        """Keep decoder weights materialized until every tiled chunk has been emitted."""
        return list(self.decoder.decode_video(latent, tiling, generator))


class LTXDiffusionMediaDecoder(MultiModalLatentDecoder):
    """Decode split video/audio checkpoints using the run's continued noise generator."""

    uses_run_generator: bool = True

    def __init__(
        self,
        video: LTXDiffusionVideoDecoderModule,
        audio: LTXAudioDecoderModule,
        vocoder: LTXVocoderModule,
        *,
        compute_dtype: torch.dtype,
        mode: str,
        vae_config: Mapping[str, object],
        model_bytes: int,
        tiling: object = "auto",
    ) -> None:
        self.video = video
        self.audio = audio
        self.vocoder = vocoder
        self.compute_dtype = compute_dtype
        self.mode = mode
        self.vae_config = vae_config
        self.model_bytes = model_bytes
        self.tiling = tiling

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
    ) -> Mapping[str, object]:
        from worldfoundry.core.vram.memory import activation_budget_bytes

        from .video.diffusion_tiling import recommended_decode_tiling_config
        from .video.diffusion_video_decoder import diffvae_tiling_geometry_from_vae_config
        from .video.transformer.config import DiffVAEMode

        channels = int(
            self.vae_config.get("decoder", {}).get("in_channels", self.vae_config.get("latent_channels", 128))
        )
        video_shape, audio_shape = LTXMediaDecoder._shapes(request)
        video_shape = video_shape._replace(channels=channels)
        patchifier = VideoLatentPatchifier(1)
        latent = states["video"].latent[:, : patchifier.get_token_count(video_shape)]
        video = patchifier.unpatchify(latent, video_shape).to(self.compute_dtype)
        audio = AudioPatchifier(1).unpatchify(states["audio"].latent, audio_shape).to(self.compute_dtype)
        tiling = self.tiling
        if tiling == "auto":
            tiling = recommended_decode_tiling_config(
                **diffvae_tiling_geometry_from_vae_config(dict(self.vae_config)),
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                mode=DiffVAEMode(self.mode),
                free_bytes=activation_budget_bytes(video.device),
                model_bytes=self.model_bytes,
            )
        chunks = self.video(video, tiling, generator)
        if not chunks:
            raise RuntimeError("LTX diffusion video decoder returned no chunks")
        decoded_audio = decode_audio(audio, self.audio.decoder, self.vocoder.vocoder)
        return {
            "video": torch.cat(chunks, dim=0),
            "audio": decoded_audio.waveform,
            "audio_sampling_rate": decoded_audio.sampling_rate,
        }


def build_ltx_diffusion_media_decoder(context: ComponentBuildContext) -> LTXDiffusionMediaDecoder:
    """Load the diffusion video VAE, audio VAE and BWE vocoder from split files."""
    from safetensors import safe_open

    from .video.state_dict import convert_ltx_diffusion_decoder_state_dict

    mode = str(context.component_options.get("diffvae_optimization", "chunked_eager"))
    video_checkpoint = context.require_checkpoint("video_vae")
    materialized = NativeCheckpointResolver().materialize(video_checkpoint)
    config = safetensors_json_metadata(materialized)
    model_bytes = 0
    with safe_open(materialized.paths[0], framework="pt", device="cpu") as handle:
        import math

        for key in handle.keys():  # noqa: SIM118  # safetensors handles are not iterable
            if key.startswith(("vae.decoder.", "decoder.")) or "per_channel_statistics" in key:
                model_bytes += math.prod(handle.get_slice(key).get_shape()) * 2
    video = _load_configured_module(
        video_checkpoint,
        context.policy,
        module_class=LTXDiffusionVideoDecoderModule,
        converter=partial(convert_ltx_diffusion_decoder_state_dict, mode=mode),
        config={"mode": mode},
        lazy_state_dict=True,
    )
    audio_checkpoint = context.require_checkpoint("audio_vae")
    audio = _load_configured_module(
        audio_checkpoint,
        context.policy,
        module_class=LTXAudioDecoderModule,
        converter=partial(convert_ltx_audio_decoder_state_dict, lazy=True),
        lazy_state_dict=True,
    )
    vocoder = _load_configured_module(
        audio_checkpoint,
        context.policy,
        module_class=LTXVocoderModule,
        converter=partial(convert_ltx_vocoder_state_dict, lazy=True),
        lazy_state_dict=True,
    )
    return LTXDiffusionMediaDecoder(
        video,
        audio,
        vocoder,
        compute_dtype=context.policy.dtype,
        mode=mode,
        vae_config=config["vae"],
        model_bytes=model_bytes,
        tiling=context.component_options.get("tiling", "auto"),
    )
