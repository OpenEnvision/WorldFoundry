"""Public BF16, FP8 and FP4 R2V DMD inference entrypoint for Echo 1.5."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import torch
import yaml

from worldfoundry.core.io import resolve_data_path

from .ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from .ltx_distillation.audio_voice_filter import VoiceFilterConfig
from .ltx_distillation.generator_loader import GeneratorLoadReport, load_inference_generator
from .ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalR2VInferencePipeline
from .ltx_distillation.layerwise_offload import DiTLayerwiseOffload
from .ltx_distillation.models.vae_wrapper import create_vae_wrappers
from .ltx_distillation.r2v_conditioning import (
    R2VConditionBundle,
    encode_r2v_requests,
    load_r2v_conditioning,
    r2v_conditioning_cache_path,
    save_r2v_conditioning,
)
from .ltx_distillation.release_checkpoint import ReleaseCheckpoint, resolve_release_checkpoint
from .ltx_distillation.text_conditioning import artifact_fingerprint
from .ltx_distillation.utils import add_noise, compute_latent_shapes, decode_generated_sample, write_generated_media
from .r2v_schema import MAX_MEMORY_SLOTS, R2VRequest, load_r2v_request

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(resolve_data_path("models/runtime/configs/joyai-echo-longvideo/inference.bf16.yaml"))


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(path_value: str, *, required: bool = True) -> str | None:
    value = str(path_value or "").strip()
    if not value:
        if required:
            raise ValueError("required path is empty")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def resolve_video_vae_decode_mode(
    requested_mode: str,
    *,
    device_type: str,
    platform_name: str,
    total_memory_bytes: int | None,
) -> str:
    """Resolve ``auto`` without coupling decode policy to checkpoint precision."""
    mode = requested_mode.strip().lower()
    if mode not in {"auto", "tiled", "untiled"}:
        raise ValueError("video_vae.decode_mode must be auto, tiled, or untiled")
    if mode != "auto":
        return mode
    if device_type == "cuda" and platform_name == "win32":
        return "tiled"
    if device_type == "cuda" and (total_memory_bytes is None or total_memory_bytes < 48 * 2**30):
        return "tiled"
    return "untiled"


class InferenceConfig:
    """Validated YAML configuration with optional CLI overrides."""

    def __init__(self, config_path: Path, **cli_overrides: Any) -> None:
        config = _load_yaml_config(config_path)
        paths = config.get("paths", {})
        video = config.get("video", {})
        denoising = config.get("denoising", {})
        memory = config.get("memory", {})
        voice_filter = memory.get("voice_filter", {})
        runtime = config.get("inference", {})
        video_vae = config.get("video_vae", {}) or {}
        self.checkpoint = _resolve_path(paths.get("checkpoint", "checkpoints/echo15_full_dmd"))
        self.gemma_path = _resolve_path(paths.get("gemma_path", "checkpoints/gemma-3-12b"))
        self.requests_dir = _resolve_path(paths.get("requests_dir", "examples/the_last_visa/requests"))
        self.requests_glob = str(paths.get("requests_glob", "*.json"))
        self.output_root = _resolve_path(paths.get("output_root", "inference_result"))
        self.conditioning_cache_dir = _resolve_path(paths.get("conditioning_cache_dir", ""), required=False)

        self.num_frames = int(video.get("num_frames", 241))
        self.video_height = int(video.get("height", 736))
        self.video_width = int(video.get("width", 1280))
        self.video_fps = int(video.get("fps", 25))
        self.seed = int(video.get("seed", 42))

        self.video_vae_decode_mode = str(video_vae.get("decode_mode", "tiled")).lower()
        self.video_vae_tile_size_frames = int(video_vae.get("tile_size_frames", 64))
        self.video_vae_tile_overlap_frames = int(video_vae.get("tile_overlap_frames", 24))
        self.video_vae_tile_size_pixels = int(video_vae.get("tile_size_pixels", 512))
        self.video_vae_tile_overlap_pixels = int(video_vae.get("tile_overlap_pixels", 64))

        self.denoising_steps = [int(value) for value in denoising.get("steps", [])]
        self.denoising_sigmas = [float(value) for value in denoising.get("sigmas", [])]

        self.memory_max_size = int(memory.get("max_size", MAX_MEMORY_SLOTS))
        self.memory_downscale_factor = int(memory.get("downscale_factor", 1))
        self.memory_position_mode = str(memory.get("position_mode", "slot_center"))
        self.memory_position_offset = float(memory.get("position_offset", 500.0))
        self.memory_position_slot_stride = float(memory.get("position_slot_stride", 50.0))
        self.enable_audio_memory = bool(memory.get("enable_audio", True))
        self.voice_filter = VoiceFilterConfig(
            enabled=bool(voice_filter.get("enabled", True)),
            backend=str(voice_filter.get("backend", "msst_speech")),
        )

        self.device = str(runtime.get("device", "cuda"))
        self.dtype = str(runtime.get("dtype", "bfloat16")).lower()
        self.prompt_max_chars = int(runtime.get("prompt_max_chars", 1500))
        self.text_batch_size = int(runtime.get("text_batch_size", 1))
        self.image_batch_size = int(runtime.get("image_batch_size", 1))
        self.audio_batch_size = int(runtime.get("audio_batch_size", 1))
        dit_offload = runtime.get("dit_layerwise_offload", {}) or {}
        self.dit_layerwise_offload = bool(dit_offload.get("enabled", False))
        self.dit_resident_blocks = int(dit_offload.get("resident_blocks", 0))
        self.dit_prefetch_blocks = int(dit_offload.get("prefetch_blocks", 1))
        self.dit_pin_memory = bool(dit_offload.get("pin_memory", True))

        for key, value in cli_overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        self.validate()

    def validate(self) -> None:
        if self.dtype not in {"bfloat16", "bf16"}:
            raise ValueError("Echo 1.5 uses BF16 activations for every checkpoint mode")
        if len(self.denoising_sigmas) < 2:
            raise ValueError("denoising.sigmas must contain at least two values")
        if self.denoising_steps and len(self.denoising_steps) != len(self.denoising_sigmas):
            raise ValueError("denoising.steps and denoising.sigmas must have equal length")
        if self.memory_position_mode != "slot_center":
            raise ValueError("Echo 1.5 R2V requires memory.position_mode=slot_center")
        if not 0 <= self.memory_max_size <= MAX_MEMORY_SLOTS:
            raise ValueError(f"memory.max_size must be between 0 and {MAX_MEMORY_SLOTS}")
        if self.num_frames <= 0 or self.video_height <= 0 or self.video_width <= 0:
            raise ValueError("video dimensions and frame count must be positive")
        if self.prompt_max_chars <= 0:
            raise ValueError("inference.prompt_max_chars must be positive")
        if min(self.text_batch_size, self.image_batch_size, self.audio_batch_size) <= 0:
            raise ValueError("conditioning batch sizes must be positive")
        if self.video_vae_decode_mode not in {"auto", "tiled", "untiled"}:
            raise ValueError("video_vae.decode_mode must be auto, tiled, or untiled")
        # Reuse the core validators so public configuration follows exactly the
        # same alignment constraints as the decoder implementation.
        self.video_vae_tiling_config()
        if self.dit_resident_blocks < 0:
            raise ValueError("inference.dit_layerwise_offload.resident_blocks must be non-negative")
        if self.dit_prefetch_blocks < 1:
            raise ValueError("inference.dit_layerwise_offload.prefetch_blocks must be positive")

    def video_vae_tiling_config(self) -> TilingConfig:
        return TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=self.video_vae_tile_size_pixels,
                tile_overlap_in_pixels=self.video_vae_tile_overlap_pixels,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=self.video_vae_tile_size_frames,
                tile_overlap_in_frames=self.video_vae_tile_overlap_frames,
            ),
        )


class InferenceEngine:
    """Prepare complete R2V conditions first, then run the shared DMD pipeline."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16
        self.release_checkpoint: ReleaseCheckpoint = resolve_release_checkpoint(config.checkpoint)
        self.model_checkpoint = self.release_checkpoint.model_path
        self.gemma_path = Path(config.gemma_path)
        if not self.gemma_path.exists():
            raise FileNotFoundError(f"Gemma directory not found: {self.gemma_path}")

        self.generator = None
        self.video_vae = None
        self.audio_vae = None
        self.pipeline = None
        self.audio_sample_rate: int | None = None
        self.generator_load_report: GeneratorLoadReport | None = None
        self.dit_offload: DiTLayerwiseOffload | None = None
        self.generator_location = "unloaded"
        self.decoder_location = "unloaded"
        total_memory_bytes = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            total_memory_bytes = torch.cuda.get_device_properties(self.device).total_memory
        self.video_vae_decode_mode = resolve_video_vae_decode_mode(
            config.video_vae_decode_mode,
            device_type=self.device.type,
            platform_name=sys.platform,
            total_memory_bytes=total_memory_bytes,
        )
        self.video_vae_tiling_config = (
            config.video_vae_tiling_config() if self.video_vae_decode_mode == "tiled" else None
        )
        print(
            f"[VAE] video decode={self.video_vae_decode_mode} (configured={config.video_vae_decode_mode})",
            flush=True,
        )

    def prepare_conditions(
        self,
        request_files: list[Path],
        requests: list[R2VRequest],
        *,
        encode_cache_misses: bool = False,
    ) -> dict[Path, R2VConditionBundle]:
        config = self.config
        checkpoint_id = artifact_fingerprint(self.release_checkpoint.root)
        gemma_id = artifact_fingerprint(self.gemma_path)
        cached: dict[Path, R2VConditionBundle] = {}
        pending_files: list[Path] = []
        pending_requests: list[R2VRequest] = []
        if config.conditioning_cache_dir:
            print("[Stage 1] Loading precomputed R2V conditioning", flush=True)
            for request_file, request in zip(request_files, requests, strict=True):
                cache_path = r2v_conditioning_cache_path(
                    config.conditioning_cache_dir,
                    config.requests_dir,
                    request_file,
                )
                try:
                    cached[request_file] = load_r2v_conditioning(
                        cache_path,
                        request=request,
                        checkpoint_fingerprint=checkpoint_id,
                        gemma_fingerprint=gemma_id,
                    )
                except (FileNotFoundError, ValueError):
                    if not encode_cache_misses:
                        raise
                    pending_files.append(request_file)
                    pending_requests.append(request)
            if not pending_requests:
                return cached
        else:
            pending_files = list(request_files)
            pending_requests = list(requests)

        print(
            f"[Stage 1] Encoding {len(pending_requests)} R2V conditions "
            f"text_batch={config.text_batch_size} image_batch={config.image_batch_size} "
            f"audio_batch={config.audio_batch_size}",
            flush=True,
        )
        bundles = encode_r2v_requests(
            pending_requests,
            checkpoint_path=str(self.model_checkpoint),
            gemma_path=str(self.gemma_path),
            device=self.device,
            voice_filter_config=config.voice_filter,
            dtype=self.dtype,
            text_batch_size=config.text_batch_size,
            image_batch_size=config.image_batch_size,
            audio_batch_size=config.audio_batch_size,
            enable_audio_memory=config.enable_audio_memory,
            memory_position_mode=config.memory_position_mode,
            memory_position_offset=config.memory_position_offset,
            memory_position_slot_stride=config.memory_position_slot_stride,
        )
        for request_file, request, bundle in zip(pending_files, pending_requests, bundles, strict=True):
            cached[request_file] = bundle
            if config.conditioning_cache_dir:
                save_r2v_conditioning(
                    r2v_conditioning_cache_path(
                        config.conditioning_cache_dir,
                        config.requests_dir,
                        request_file,
                    ),
                    bundle,
                    request=request,
                    checkpoint_fingerprint=checkpoint_id,
                    gemma_fingerprint=gemma_id,
                )
        print("[Stage 1] Complete DiT conditioning ready on CPU", flush=True)
        return {request_file: cached[request_file] for request_file in request_files}

    def load_generator(self) -> None:
        if self.generator is not None:
            return
        config = self.config
        print(
            f"[Stage 2] Loading {self.release_checkpoint.name} precision={self.release_checkpoint.precision}",
            flush=True,
        )
        # On Windows, reopening a large BF16 safetensors file for the VAE after
        # materializing a large CPU weight pool can crash inside torch_cpu.dll.
        # Build the decoder modules before attaching the layerwise offload pool.
        precision = self.release_checkpoint.precision
        preload_vaes = config.dit_layerwise_offload and precision in {"bf16", "fp4"}
        if preload_vaes and (self.video_vae is None or self.audio_vae is None):
            print(
                f"[Stage 2] Preloading VAEs before {precision.upper()} generator restore",
                flush=True,
            )
            self.video_vae, self.audio_vae = create_vae_wrappers(
                checkpoint_path=str(self.model_checkpoint),
                device=torch.device("cpu"),
                dtype=self.dtype,
                with_video_encoder=False,
                with_audio_encoder=False,
                decoder_device=torch.device("cpu"),
            )

        self.generator, self.generator_load_report = load_inference_generator(
            checkpoint=self.release_checkpoint,
            gemma_path=self.gemma_path,
            device=self.device,
            dtype=self.dtype,
            video_height=config.video_height,
            video_width=config.video_width,
            load_on_cpu=config.dit_layerwise_offload,
        )
        if config.dit_layerwise_offload:
            self.dit_offload = DiTLayerwiseOffload(
                self.generator,
                execution_device=self.device,
                resident_blocks=config.dit_resident_blocks,
                prefetch_blocks=config.dit_prefetch_blocks,
                pin_memory=config.dit_pin_memory,
            )
            report = self.dit_offload.report
            print(
                f"[Stage 2] DiT layerwise offload blocks={report.block_count} "
                f"resident={report.resident_blocks} prefetch={report.prefetch_blocks} "
                f"cpu_weights={report.cpu_weight_bytes / 2**30:.2f}GiB "
                f"pinned={report.pinned_weight_bytes / 2**30:.2f}GiB",
                flush=True,
            )
        print(
            f"[Stage 2] format={self.generator_load_report.format} "
            f"quantized_modules={self.generator_load_report.quantized_modules} "
            f"missing={len(self.generator_load_report.missing_keys)} "
            f"unexpected={len(self.generator_load_report.unexpected_keys)}",
            flush=True,
        )

        if self.video_vae is None or self.audio_vae is None:
            self.video_vae, self.audio_vae = create_vae_wrappers(
                checkpoint_path=str(self.model_checkpoint),
                device=torch.device("cpu"),
                dtype=self.dtype,
                with_video_encoder=False,
                with_audio_encoder=False,
                decoder_device=torch.device("cpu"),
            )
        self.video_vae.eval()
        self.audio_vae.eval()
        sigmas = torch.tensor(config.denoising_sigmas, device=self.device, dtype=torch.float32)
        self.pipeline = BidirectionalR2VInferencePipeline(
            self.generator,
            add_noise,
            sigmas,
            memory_downscale_factor=config.memory_downscale_factor,
        )
        self.audio_sample_rate = self.audio_vae.get_output_sample_rate() or 24000
        self.generator_location = "cpu" if config.dit_layerwise_offload else str(self.device)
        self.decoder_location = "cpu"
        print("[Stage 2] Generator and decode VAEs ready", flush=True)

    def unload_generator(self) -> None:
        """Release all resident generation/decode weights owned by this engine."""

        self.release_generator_weights()
        self.video_vae = None
        self.audio_vae = None
        self.audio_sample_rate = None
        self.generator_load_report = None
        self.decoder_location = "unloaded"
        gc.collect()
        self._empty_cuda_cache()

    def release_generator_weights(self) -> None:
        """Release only DiT/pipeline weights while preserving loaded decoders."""

        if self.dit_offload is not None:
            self.dit_offload.close()
        self.pipeline = None
        self.dit_offload = None
        self.generator = None
        self.generator_location = "unloaded"
        gc.collect()
        self._empty_cuda_cache()

    def generation_storage_bytes(self) -> int:
        """Approximate unique tensor storage retained by generation modules."""

        storages: dict[tuple[str, int], int] = {}
        for module in (self.generator, self.video_vae, self.audio_vae):
            if module is None:
                continue
            for tensor in (*module.parameters(), *module.buffers()):
                storage = tensor.untyped_storage()
                key = (str(tensor.device), int(storage.data_ptr()))
                storages.setdefault(key, int(storage.nbytes()))
        return sum(storages.values())

    def stage_generator_for_conditioning(self, policy: str) -> None:
        """Place warm generation weights for a conditioning cache miss."""

        if policy not in {"gpu", "cpu", "release"}:
            raise ValueError(f"unsupported conditioning generator policy: {policy}")
        if policy == "release":
            self.unload_generator()
            return
        if self.generator is None:
            return
        if policy == "gpu":
            return
        if self.dit_offload is not None:
            self.dit_offload.deactivate()
        else:
            self._move(self.generator, "cpu")
        if self.video_vae is not None:
            self._move(self.video_vae, "cpu")
        if self.audio_vae is not None:
            self._move(self.audio_vae, "cpu")
        self.generator_location = "cpu"
        self.decoder_location = "cpu"
        self._empty_cuda_cache()

    @staticmethod
    def _move(module, target_device: str | torch.device) -> None:
        if module is not None:
            module.to(target_device)

    def _empty_cuda_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _stage_for_denoise(self) -> None:
        already_co_resident = self.generator_location.startswith("cuda") and self.decoder_location.startswith("cuda")
        if not already_co_resident:
            self._move(self.video_vae.decoder, "cpu")
            self._move(self.audio_vae.decoder, "cpu")
            self._move(self.audio_vae.vocoder, "cpu")
            self.decoder_location = "cpu"
            self._empty_cuda_cache()
        if self.dit_offload is not None:
            self.dit_offload.activate()
            self.generator_location = "layerwise"
        else:
            self._move(self.generator, self.device)
            self.generator_location = str(self.device)

    def _stage_for_decode(self, generator_policy: str = "cpu") -> str:
        if generator_policy not in {"gpu", "cpu", "release"}:
            raise ValueError(f"unsupported decode generator policy: {generator_policy}")
        if generator_policy == "release":
            self.release_generator_weights()
        elif self.dit_offload is not None:
            self.dit_offload.deactivate()
        elif generator_policy == "cpu":
            self._move(self.generator, "cpu")
        self.generator_location = (
            "cpu" if self.dit_offload is not None or generator_policy == "cpu" else self.generator_location
        )
        self._empty_cuda_cache()
        try:
            self._move(self.video_vae.decoder, self.device)
            self._move(self.audio_vae.decoder, self.device)
            self._move(self.audio_vae.vocoder, self.device)
        except torch.OutOfMemoryError:
            if generator_policy != "gpu":
                raise
            # Live free-memory sampling is advisory. If another process races
            # us, release DiT and retry decoder placement without failing the job.
            self._move(self.video_vae, "cpu")
            self._move(self.audio_vae, "cpu")
            self.release_generator_weights()
            self._move(self.video_vae.decoder, self.device)
            self._move(self.audio_vae.decoder, self.device)
            self._move(self.audio_vae.vocoder, self.device)
            generator_policy = "release"
        self.decoder_location = str(self.device)
        return generator_policy

    def run_request(
        self,
        request_file: Path,
        request: R2VRequest,
        output_dir: Path,
        bundle: R2VConditionBundle,
        *,
        stage_callback=None,
        decode_generator_policy=None,
    ) -> dict[str, Any]:
        if self.generator is None or self.pipeline is None or self.audio_sample_rate is None:
            raise RuntimeError("load_generator() must be called before inference")
        if len(request.memory_slots) > self.config.memory_max_size:
            raise ValueError(
                f"request has {len(request.memory_slots)} memory slots; configured maximum is "
                f"{self.config.memory_max_size}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=request.num_frames,
            video_height=request.height,
            video_width=request.width,
            batch_size=1,
            video_fps=float(self.config.video_fps),
        )
        latent_height, latent_width = int(video_shape[-2]), int(video_shape[-1])
        self.generator.latent_height = latent_height
        self.generator.latent_width = latent_width
        self.generator.video_frame_seqlen = latent_height * latent_width

        condition = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in bundle.text.items()
        }
        memory_audio_kwargs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in bundle.memory_audio_kwargs.items()
        }
        if stage_callback is not None:
            stage_callback("inferring")
        self._stage_for_denoise()
        started = time.perf_counter()
        denoise_started = time.perf_counter()
        fork_devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(request.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(request.seed)
            video_latent, audio_latent = self.pipeline.generate(
                video_shape=tuple(video_shape),
                audio_shape=tuple(audio_shape),
                conditional_dict=condition,
                memory_video=bundle.memory_video,
                first_frame_latent=bundle.first_frame_latent,
                seed=request.seed,
                **memory_audio_kwargs,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        denoise_seconds = time.perf_counter() - denoise_started

        if stage_callback is not None:
            stage_callback("decoding")
        generator_policy = str(decode_generator_policy()) if decode_generator_policy is not None else "cpu"
        generator_policy = self._stage_for_decode(generator_policy)
        decode_started = time.perf_counter()
        video_uint8, audio_waveform = decode_generated_sample(
            self.video_vae,
            self.audio_vae,
            video_latent,
            audio_latent,
            video_tiling_config=self.video_vae_tiling_config,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        if stage_callback is not None:
            stage_callback("writing")
        output_path = output_dir / "result.mp4"
        write_result = write_generated_media(
            output_path=output_path,
            video_uint8=video_uint8,
            audio_waveform=audio_waveform,
            fps=self.config.video_fps,
            audio_sr=self.audio_sample_rate,
        )
        metadata = {
            "schema": "echo15.r2v.result.v1",
            "request_file": str(request_file),
            "request": request.as_payload(),
            "model_checkpoint": str(self.release_checkpoint.root),
            "generator": asdict(self.generator_load_report),
            "dit_layerwise_offload": (asdict(self.dit_offload.report) if self.dit_offload else {"enabled": False}),
            "decode_generator_policy": generator_policy,
            "video_vae": {
                "decode_mode": self.video_vae_decode_mode,
                "tile_size_frames": self.config.video_vae_tile_size_frames,
                "tile_overlap_frames": self.config.video_vae_tile_overlap_frames,
                "tile_size_pixels": self.config.video_vae_tile_size_pixels,
                "tile_overlap_pixels": self.config.video_vae_tile_overlap_pixels,
            },
            "conditioning": {
                "memory_slots": len(request.memory_slots),
                "has_first_frame": bundle.first_frame_latent is not None,
                "has_memory_audio": "memory_audio" in bundle.memory_audio_kwargs,
                "input_fingerprints": bundle.input_fingerprints,
            },
            "output_path": str(output_path),
            "audio_latent_shape": list(audio_latent.shape) if audio_latent is not None else None,
            "audio_stats": write_result["audio_stats"],
            "timing": {
                "denoise_seconds": round(denoise_seconds, 3),
                "decode_seconds": round(decode_seconds, 3),
                "total_seconds": round(time.perf_counter() - started, 3),
            },
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[Inference] {request.shot_id} done denoise={denoise_seconds:.1f}s "
            f"decode={decode_seconds:.1f}s output={output_path}",
            flush=True,
        )
        del video_latent, audio_latent, video_uint8, audio_waveform, condition
        self._empty_cuda_cache()
        return {"output_path": str(output_path), "metadata": metadata}


def load_request_files(config: InferenceConfig, single_request: str | None = None) -> list[Path]:
    if single_request:
        path = Path(single_request).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"R2V request not found: {path}")
        return [path]
    requests_dir = Path(config.requests_dir)
    if Path(config.requests_glob).is_absolute():
        files = sorted(Path(value) for value in glob(config.requests_glob, recursive=True))
    else:
        files = sorted(requests_dir.glob(config.requests_glob))
    if not files:
        raise FileNotFoundError(f"no R2V requests matched {requests_dir / config.requests_glob}")
    return files


def load_requests(config: InferenceConfig, request_files: list[Path]) -> list[R2VRequest]:
    return [
        load_r2v_request(
            path,
            default_num_frames=config.num_frames,
            default_width=config.video_width,
            default_height=config.video_height,
            default_seed=config.seed,
            prompt_max_chars=config.prompt_max_chars,
        )
        for path in request_files
    ]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Echo 1.5 BF16/FP8/FP4 DMD R2V inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--request", help="run one production-compatible R2V JSON request")
    parser.add_argument("--checkpoint")
    parser.add_argument("--gemma-path")
    parser.add_argument("--requests-dir")
    parser.add_argument("--requests-glob")
    parser.add_argument("--output-root")
    parser.add_argument("--conditioning-cache-dir")
    parser.add_argument(
        "--condition-encode",
        "--condition_encode",
        "--text-encode",
        dest="condition_encode",
        action="store_true",
        help="batch text, condition-image, memory-image and memory-audio encoding, then exit",
    )
    parser.add_argument("--overwrite-condition-cache", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--video-height", type=int)
    parser.add_argument("--video-width", type=int)
    parser.add_argument("--video-fps", type=int)
    parser.add_argument("--video-vae-decode-mode", choices=("auto", "tiled", "untiled"))
    parser.add_argument("--video-vae-tile-size-frames", type=int)
    parser.add_argument("--video-vae-tile-overlap-frames", type=int)
    parser.add_argument("--video-vae-tile-size-pixels", type=int)
    parser.add_argument("--video-vae-tile-overlap-pixels", type=int)
    parser.add_argument("--text-batch-size", type=int)
    parser.add_argument("--image-batch-size", type=int)
    parser.add_argument("--audio-batch-size", type=int)
    parser.add_argument("--dit-layerwise-offload", type=str_to_bool)
    parser.add_argument("--dit-resident-blocks", type=int)
    parser.add_argument("--dit-prefetch-blocks", type=int)
    parser.add_argument("--dit-pin-memory", type=str_to_bool)
    parser.add_argument("--memory-max-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "request", "condition_encode", "overwrite_condition_cache"} and value is not None
    }
    for path_key in (
        "checkpoint",
        "gemma_path",
        "requests_dir",
        "output_root",
        "conditioning_cache_dir",
    ):
        if path_key in overrides:
            overrides[path_key] = str(Path(overrides[path_key]).expanduser().resolve())
    config = InferenceConfig(config_path, **overrides)
    request_files = load_request_files(config, args.request)
    requests = load_requests(config, request_files)

    if args.condition_encode:
        from .scripts.precompute_conditioning import main as precompute_main

        precompute_args = ["--config", str(config_path)]
        for option, value in (
            ("--request", args.request),
            ("--checkpoint", args.checkpoint),
            ("--gemma-path", args.gemma_path),
            ("--requests-dir", args.requests_dir),
            ("--requests-glob", args.requests_glob),
            ("--output-dir", args.conditioning_cache_dir),
            ("--text-batch-size", args.text_batch_size),
            ("--image-batch-size", args.image_batch_size),
            ("--audio-batch-size", args.audio_batch_size),
        ):
            if value is not None:
                precompute_args.extend((option, str(value)))
        if args.overwrite_condition_cache:
            precompute_args.append("--overwrite")
        precompute_main(precompute_args)
        return

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "multi-process generation is unsupported; use plain Python for inference or "
            "--condition-encode with torchrun"
        )
    engine = InferenceEngine(config)
    bundles = engine.prepare_conditions(request_files, requests)
    engine.load_generator()
    output_root = Path(config.output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for request_file, request in zip(request_files, requests, strict=True):
        engine.run_request(
            request_file,
            request,
            output_root / request.work_id / request.shot_id / f"inference_{timestamp}",
            bundles[request_file],
        )
    print(f"[Inference] Processed {len(request_files)} R2V request(s)", flush=True)


if __name__ == "__main__":
    main()
