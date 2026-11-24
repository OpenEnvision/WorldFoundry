"""Batch R2V conditioning that stops exactly at the DiT input boundary."""

from __future__ import annotations

import base64
import binascii
import gc
import hashlib
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
from urllib.request import Request, urlopen

import torch
import torchaudio
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torchvision.transforms import functional as TVF

from ..r2v_schema import R2VRequest
from .audio_voice_filter import VoiceFilterConfig, filter_voice_only
from .models.vae_wrapper import create_vae_wrappers
from .text_conditioning import TextCondition, encode_prompts_two_stage

CONDITIONING_SCHEMA_VERSION = "echo15.r2v.conditioning.v1"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
TensorDict: TypeAlias = dict[str, torch.Tensor | None]


@dataclass
class R2VConditionBundle:
    """All CPU tensors and scalar arguments required immediately before DiT."""

    text: TextCondition
    first_frame_latent: torch.Tensor | None
    memory_video: torch.Tensor | None
    memory_audio_kwargs: dict[str, Any]
    input_fingerprints: dict[str, str]


def _validate_bundle(bundle: R2VConditionBundle, request: R2VRequest) -> None:
    """Reject incomplete caches before they reach the generator."""

    for key in ("video_context", "attention_mask"):
        value = bundle.text.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
            raise ValueError(f"R2V conditioning requires batched text.{key}")
    audio_context = bundle.text.get("audio_context")
    if audio_context is not None and (not isinstance(audio_context, torch.Tensor) or audio_context.shape[0] != 1):
        raise ValueError("R2V conditioning text.audio_context must have batch size 1")

    first_frame = bundle.first_frame_latent
    if (first_frame is not None) != (request.condition_img is not None):
        raise ValueError("condition_img and first_frame_latent must either both exist or both be absent")
    if first_frame is not None and (first_frame.ndim != 5 or first_frame.shape[0] != 1 or first_frame.shape[1] != 1):
        raise ValueError(f"first_frame_latent must have shape [1, 1, C, H, W], got {tuple(first_frame.shape)}")

    slot_count = len(request.memory_slots)
    memory_video = bundle.memory_video
    if (memory_video is not None) != bool(slot_count):
        raise ValueError("memory slots and memory_video must either both exist or both be absent")
    if memory_video is not None and (
        memory_video.ndim != 5 or memory_video.shape[0] != 1 or memory_video.shape[1] != slot_count
    ):
        raise ValueError(f"memory_video must have shape [1, {slot_count}, C, H, W], got {tuple(memory_video.shape)}")

    audio = bundle.memory_audio_kwargs.get("memory_audio")
    timestep = bundle.memory_audio_kwargs.get("memory_audio_timestep")
    lengths = bundle.memory_audio_kwargs.get("memory_audio_segment_lengths")
    if audio is None:
        if timestep is not None or lengths is not None:
            raise ValueError("memory audio timestep/segments require memory_audio")
        return
    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError("memory_audio must have shape [1, T, C]")
    if not isinstance(timestep, torch.Tensor) or tuple(timestep.shape) != tuple(audio.shape[:2]):
        raise ValueError("memory_audio_timestep must match memory_audio batch/time dimensions")
    if (
        not isinstance(lengths, tuple)
        or len(lengths) != 1
        or len(lengths[0]) != slot_count
        or sum(int(value) for value in lengths[0]) != audio.shape[1]
    ):
        raise ValueError("memory_audio_segment_lengths must align one-to-one with memory slots")


def r2v_conditioning_cache_path(
    cache_root: str | Path,
    requests_root: str | Path,
    request_file: str | Path,
) -> Path:
    requests_root = Path(requests_root).resolve()
    request_file = Path(request_file).resolve()
    try:
        relative = request_file.relative_to(requests_root)
    except ValueError:
        relative = Path(request_file.name)
    return Path(cache_root) / relative.with_suffix(".safetensors")


class _ResourceStore:
    def __init__(self) -> None:
        self._bytes: dict[str, bytes] = {}
        self.fingerprints: dict[str, str] = {}

    def read(self, source: str, *, kind: str, max_bytes: int) -> bytes:
        if source in self._bytes:
            return self._bytes[source]
        if source.startswith("data:"):
            header, separator, encoded = source.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ValueError(f"{kind} has an invalid data URL")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{kind} has invalid base64 data") from exc
        elif source.startswith(("http://", "https://")):
            request = Request(source, headers={"User-Agent": "JoyAI-Echo15/1.0"})
            with urlopen(request, timeout=30) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes:
                    raise ValueError(f"{kind} exceeds {max_bytes} bytes: {source}")
                data = response.read(max_bytes + 1)
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"{kind} not found: {path}")
            if path.stat().st_size > max_bytes:
                raise ValueError(f"{kind} exceeds {max_bytes} bytes: {path}")
            data = path.read_bytes()
        if not data or len(data) > max_bytes:
            raise ValueError(f"{kind} must contain 1..{max_bytes} bytes: {source}")
        self._bytes[source] = data
        self.fingerprints[source] = hashlib.sha256(data).hexdigest()
        return data

    def image(self, source: str) -> Image.Image:
        raw = self.read(source, kind="R2V image", max_bytes=MAX_IMAGE_BYTES)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB")

    def audio(self, source: str) -> tuple[torch.Tensor, int]:
        raw = self.read(source, kind="R2V audio", max_bytes=MAX_AUDIO_BYTES)
        waveform, sample_rate = torchaudio.load(io.BytesIO(raw))
        return waveform.detach().cpu().float().contiguous(), int(sample_rate)


def _release_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _request_cache_fingerprint(request: R2VRequest) -> str:
    if request.request_sha256:
        return request.request_sha256
    payload = request.as_payload()
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _image_tensor(image: Image.Image, *, height: int, width: int) -> torch.Tensor:
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return (TVF.to_tensor(image) * 2.0 - 1.0).unsqueeze(1).contiguous()


def _normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(waveform).detach().cpu().float()
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim > 2:
        value = value.reshape(value.shape[-2], value.shape[-1])
    if value.ndim != 2 or value.shape[-1] <= 1:
        raise ValueError(f"R2V audio has no usable samples: shape={tuple(value.shape)}")
    if value.shape[0] == 1:
        value = value.repeat(2, 1)
    elif value.shape[0] > 2:
        value = value[:2]
    return value.contiguous()


def _chunked(values: list[Any], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _encode_unique_images(
    requests: list[R2VRequest],
    *,
    video_vae,
    resources: _ResourceStore,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[tuple[str, int, int], torch.Tensor]:
    grouped: dict[tuple[int, int], list[tuple[str, int, int]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for request in requests:
        sources = [request.condition_img] + [slot.image_url for slot in request.memory_slots]
        for source in sources:
            if not source:
                continue
            key = (source, request.height, request.width)
            if key not in seen:
                seen.add(key)
                grouped[(request.height, request.width)].append(key)

    encoded: dict[tuple[str, int, int], torch.Tensor] = {}
    if not grouped:
        return encoded
    video_vae.encoder.to(device=device, dtype=dtype)
    with torch.inference_mode():
        for (height, width), keys in grouped.items():
            for batch in _chunked(keys, batch_size):
                pixels = torch.stack(
                    [_image_tensor(resources.image(source), height=height, width=width) for source, _, _ in batch],
                    dim=0,
                ).to(device=device, dtype=dtype)
                latents = video_vae.encode(pixels).permute(0, 2, 1, 3, 4)
                for index, key in enumerate(batch):
                    encoded[key] = latents[index : index + 1].detach().cpu().contiguous()
                del pixels, latents
    video_vae.encoder.to("cpu")
    _release_cuda(device)
    return encoded


def _encode_unique_audio(
    requests: list[R2VRequest],
    *,
    audio_vae,
    resources: _ResourceStore,
    voice_filter_config: VoiceFilterConfig,
    device: torch.device,
    batch_size: int,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    if not enabled:
        return {}
    sources = list(
        dict.fromkeys(slot.audio_url for request in requests for slot in request.memory_slots if slot.audio_url)
    )
    if not sources:
        return {}
    prepared: dict[str, tuple[torch.Tensor, int]] = {}
    for source in sources:
        waveform, sample_rate = resources.audio(source)
        filtered = filter_voice_only(waveform, sample_rate, voice_filter_config)
        if filtered is None:
            raise ValueError(f"voice filter unexpectedly removed R2V audio: {source}")
        prepared[source] = (_normalize_audio(filtered), sample_rate)

    groups: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for source, (waveform, sample_rate) in prepared.items():
        groups[(sample_rate, tuple(waveform.shape))].append(source)

    encoded: dict[str, torch.Tensor] = {}
    audio_vae.encoder.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        for (sample_rate, _shape), group_sources in groups.items():
            for batch in _chunked(group_sources, batch_size):
                waveforms = torch.stack([prepared[source][0] for source in batch], dim=0)
                latents = audio_vae.encode(waveforms, sample_rate)
                for index, source in enumerate(batch):
                    encoded[source] = latents[index : index + 1].detach().cpu().contiguous()
                del waveforms, latents
    audio_vae.encoder.to("cpu")
    _release_cuda(device)
    return encoded


def encode_r2v_requests(
    requests: list[R2VRequest],
    *,
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    voice_filter_config: VoiceFilterConfig,
    dtype: torch.dtype = torch.bfloat16,
    text_batch_size: int = 1,
    image_batch_size: int = 1,
    audio_batch_size: int = 1,
    enable_audio_memory: bool = True,
    memory_position_mode: str = "slot_center",
    memory_position_offset: float = 500.0,
    memory_position_slot_stride: float = 50.0,
) -> list[R2VConditionBundle]:
    """Batch all modalities, then return one fully assembled DiT input per request."""

    if min(text_batch_size, image_batch_size, audio_batch_size) <= 0:
        raise ValueError("conditioning batch sizes must be positive")
    if not requests:
        return []
    for request in requests:
        unresolved = [slot.shot_id for slot in request.memory_slots if slot.shot_id]
        if unresolved:
            raise ValueError(
                "offline inference cannot resolve shot_id memory slots; provide image_url/audio_url "
                f"instead (shot={request.shot_id}, references={unresolved})"
            )

    unique_prompts = list(dict.fromkeys(request.prompt for request in requests))
    unique_text_conditions = encode_prompts_two_stage(
        unique_prompts,
        checkpoint_path=checkpoint_path,
        gemma_path=gemma_path,
        device=device,
        dtype=dtype,
        batch_size=text_batch_size,
    )
    text_by_prompt = dict(zip(unique_prompts, unique_text_conditions, strict=True))
    resources = _ResourceStore()
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        dtype=dtype,
        with_video_encoder=True,
        with_audio_encoder=True,
        with_decoders=False,
    )
    try:
        images = _encode_unique_images(
            requests,
            video_vae=video_vae,
            resources=resources,
            device=device,
            dtype=dtype,
            batch_size=image_batch_size,
        )
        audios = _encode_unique_audio(
            requests,
            audio_vae=audio_vae,
            resources=resources,
            voice_filter_config=voice_filter_config,
            device=device,
            batch_size=audio_batch_size,
            enabled=enable_audio_memory,
        )
    finally:
        del video_vae, audio_vae
        _release_cuda(device)

    bundles: list[R2VConditionBundle] = []
    for request in requests:
        text_condition = text_by_prompt[request.prompt]
        first_frame = images[(request.condition_img, request.height, request.width)] if request.condition_img else None
        video_slices = [
            images[(slot.image_url, request.height, request.width)] for slot in request.memory_slots if slot.image_url
        ]
        if len(video_slices) != len(request.memory_slots):
            raise ValueError(f"every offline R2V memory slot needs image_url: {request.shot_id}")
        memory_video = torch.cat(video_slices, dim=1).contiguous() if video_slices else None

        audio_slices = [audios.get(slot.audio_url) if slot.audio_url else None for slot in request.memory_slots]
        template = next((item for item in audio_slices if item is not None), None)
        memory_audio_kwargs: dict[str, Any] = (
            {
                "memory_position_mode": str(memory_position_mode),
                "memory_position_offset": float(memory_position_offset),
                "memory_position_slot_stride": float(memory_position_slot_stride),
            }
            if memory_video is not None
            else {}
        )
        if template is not None:
            aligned = [item if item is not None else torch.zeros_like(template) for item in audio_slices]
            memory_audio = torch.cat(aligned, dim=1).contiguous()
            memory_audio_kwargs.update(
                {
                    "memory_audio": memory_audio,
                    "memory_audio_timestep": torch.zeros(memory_audio.shape[:2], dtype=torch.float32),
                    "memory_audio_segment_lengths": (tuple(int(item.shape[1]) for item in aligned),),
                }
            )
        bundle = R2VConditionBundle(
            text=text_condition,
            first_frame_latent=first_frame,
            memory_video=memory_video,
            memory_audio_kwargs=memory_audio_kwargs,
            input_fingerprints=dict(resources.fingerprints),
        )
        _validate_bundle(bundle, request)
        bundles.append(bundle)
    return bundles


def save_r2v_conditioning(
    path: str | Path,
    bundle: R2VConditionBundle,
    *,
    request: R2VRequest,
    checkpoint_fingerprint: str,
    gemma_fingerprint: str,
) -> None:
    destination = Path(path)
    _validate_bundle(bundle, request)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for key, value in bundle.text.items():
        if isinstance(value, torch.Tensor):
            tensors[f"text.{key}"] = value.detach().cpu().contiguous()
    if bundle.first_frame_latent is not None:
        tensors["first_frame_latent"] = bundle.first_frame_latent.detach().cpu().contiguous()
    if bundle.memory_video is not None:
        tensors["memory_video"] = bundle.memory_video.detach().cpu().contiguous()
    for key in ("memory_audio", "memory_audio_timestep"):
        value = bundle.memory_audio_kwargs.get(key)
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().contiguous()

    scalar_audio_kwargs = {
        key: value for key, value in bundle.memory_audio_kwargs.items() if not isinstance(value, torch.Tensor)
    }
    metadata = {
        "schema_version": CONDITIONING_SCHEMA_VERSION,
        "request_sha256": _request_cache_fingerprint(request),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "gemma_fingerprint": gemma_fingerprint,
        "shot_id": request.shot_id,
        "memory_slot_count": str(len(request.memory_slots)),
        "memory_audio_kwargs": json.dumps(scalar_audio_kwargs, separators=(",", ":")),
        "input_fingerprints": json.dumps(bundle.input_fingerprints, sort_keys=True, separators=(",", ":")),
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, destination)


def load_r2v_conditioning(
    path: str | Path,
    *,
    request: R2VRequest,
    checkpoint_fingerprint: str,
    gemma_fingerprint: str,
) -> R2VConditionBundle:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"R2V conditioning cache not found: {source}")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected = {
            "schema_version": CONDITIONING_SCHEMA_VERSION,
            "request_sha256": _request_cache_fingerprint(request),
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "gemma_fingerprint": gemma_fingerprint,
            "shot_id": request.shot_id,
            "memory_slot_count": str(len(request.memory_slots)),
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {wanted!r})" for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(f"stale or incompatible R2V cache {source}: {details}")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118 - safe_open handles are not iterable.

    required_text = {"text.video_context", "text.attention_mask"}
    missing = required_text - tensors.keys()
    if missing:
        raise ValueError(f"R2V cache {source} is missing tensors: {sorted(missing)}")
    audio_kwargs = json.loads(metadata.get("memory_audio_kwargs", "{}"))
    segment_lengths = audio_kwargs.get("memory_audio_segment_lengths")
    if segment_lengths is not None:
        audio_kwargs["memory_audio_segment_lengths"] = tuple(
            tuple(int(value) for value in row) for row in segment_lengths
        )
    for key in ("memory_audio", "memory_audio_timestep"):
        if key in tensors:
            audio_kwargs[key] = tensors[key]
    bundle = R2VConditionBundle(
        text={
            "video_context": tensors["text.video_context"],
            "audio_context": tensors.get("text.audio_context"),
            "attention_mask": tensors["text.attention_mask"],
        },
        first_frame_latent=tensors.get("first_frame_latent"),
        memory_video=tensors.get("memory_video"),
        memory_audio_kwargs=audio_kwargs,
        input_fingerprints=json.loads(metadata.get("input_fingerprints", "{}")),
    )
    _validate_bundle(bundle, request)
    return bundle
