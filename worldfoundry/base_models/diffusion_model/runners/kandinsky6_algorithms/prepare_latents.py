from __future__ import annotations

import math

import torch
from torch import Tensor

from worldfoundry.base_models.diffusion_model.kandinsky6_types import LatentBundle


def prepare_video_latents(
    bs: int,
    duration: int,
    H_lat: int,
    W_lat: int,
    C: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> LatentBundle:
    """Sample random video noise latent and compute cu_seqlens.

    Returns a LatentBundle with video=(bs*duration, H_lat, W_lat, C) and
    uniform video_cu_seqlens=(bs+1,) int32.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    video = torch.randn(bs * duration, H_lat, W_lat, C, device=device, dtype=dtype, generator=g)
    cu = duration * torch.arange(bs + 1, dtype=torch.int32, device=device)
    return LatentBundle(video=video, audio=None, video_cu_seqlens=cu, audio_cu_seqlens=None)


def audio_latent_duration(
    video_latent_frames: int,
    *,
    fps: float = 24.0,
    audio_fps: int = 44100,
    downsample_factor: int = 1024,
) -> int:
    """Audio latent length matching K5 T2VA: ceil(sample_frames/fps * audio_fps / downsample)."""
    sample_frames = (video_latent_frames - 1) * 4 + 1
    return int(math.ceil(sample_frames / fps * audio_fps / downsample_factor))


def prepare_audio_latents(
    bundle: LatentBundle,
    audio_duration: int,
    audio_dim: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> LatentBundle:
    """Attach random audio noise latent to an existing LatentBundle.

    audio_duration: number of latent audio frames per batch item.
    """
    bs = bundle.video_cu_seqlens.shape[0] - 1 if bundle.video_cu_seqlens is not None else 1
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)  # offset from video seed for independence
    audio = torch.randn(bs * audio_duration, audio_dim, device=device, dtype=dtype, generator=g)
    cu = audio_duration * torch.arange(bs + 1, dtype=torch.int32, device=device)
    return LatentBundle(
        video=bundle.video,
        audio=audio,
        video_cu_seqlens=bundle.video_cu_seqlens,
        audio_cu_seqlens=cu,
    )


def prepare_visual_rope_pos(
    video_cu_seqlens: Tensor,
    H_patches: int,
    W_patches: int,
    device: torch.device | str | None = None,
) -> list[Tensor]:
    """Compute packed visual RoPE positions [T_pos, H_pos, W_pos].

    T_pos is concatenated per-sequence: [0..T_0-1, 0..T_1-1, ...].
    H and W positions are shared (same grid for all batch items).
    """
    n_frames = torch.diff(video_cu_seqlens).cpu()  # cu_seqlens is in frames, not tokens
    t_pos = torch.cat([torch.arange(n.item()) for n in n_frames])
    if device is not None:
        t_pos = t_pos.to(device)
    h_pos = torch.arange(H_patches, device=t_pos.device)
    w_pos = torch.arange(W_patches, device=t_pos.device)
    return [t_pos, h_pos, w_pos]


def prepare_audio_rope_pos(audio_cu_seqlens: Tensor, device: torch.device | str | None = None) -> Tensor:
    """Compute packed audio RoPE positions — 1-D, concatenated per sequence."""
    durations = torch.diff(audio_cu_seqlens).cpu()
    pos = torch.cat([torch.arange(d.item()) for d in durations])
    if device is not None:
        pos = pos.to(device)
    return pos


def prepare_text_rope_pos(text_cu_seqlens: Tensor, device: torch.device | str | None = None) -> Tensor:
    """Compute packed text RoPE positions from cumulative sequence lengths."""
    lengths = torch.diff(text_cu_seqlens).cpu()
    pos = torch.cat([torch.arange(l.item()) for l in lengths])
    if device is not None:
        pos = pos.to(device)
    return pos


def encode_video(
    frames: Tensor,
    vae,
    scaling_factor: float | None = None,
) -> Tensor:
    """Encode pixel-space frames into video latents (for I2V conditioning).

    frames: (bs, 3, T, H, W) float in [-1, 1] or uint8.
    Returns: (bs*T, H_lat, W_lat, C) — packed latent in THWC layout.
    """
    if frames.dtype == torch.uint8:
        frames = frames.float() / 127.5 - 1.0

    bs, _, T, H, W = frames.shape
    # VAE expects (bs, C, T, H, W)
    latent = vae.encode(frames)[0]  # (bs, C_lat, T_lat, H_lat, W_lat)

    sf = scaling_factor if scaling_factor is not None else vae.config.scaling_factor
    latent = latent * sf

    # Reshape to packed THWC layout
    _, C_lat, T_lat, H_lat, W_lat = latent.shape
    return latent.permute(0, 2, 3, 4, 1).reshape(bs * T_lat, H_lat, W_lat, C_lat)


@torch.no_grad()
def resize_image(
    image: Tensor,
    max_area: int,
    divisibility: int = 16,
    world_size: int = 1,
) -> tuple[Tensor, float]:
    """Aspect-preserving resize to fit ``max_area`` with sides divisible by ``divisibility``.

    K5 ``i2v_pipeline.resize_image`` parity. ``image`` is ``(B, C, H, W)``.
    Returns ``(resized, scale_k)``.
    """
    from math import sqrt

    h, w = image.shape[2:]
    area = h * w
    div = divisibility
    if div == 16:
        if world_size in (2, 4):
            div *= 2
        elif world_size == 8:
            div *= 4
    k = sqrt(max_area / area) / div
    new_h = int(round(h * k) * div)
    new_w = int(round(w * k) * div)
    try:
        import torchvision.transforms.functional as TF
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "I2VA image processing requires torchvision. Install it with `pip install torchvision`."
        ) from exc

    return TF.resize(image, (new_h, new_w)), k


def _load_pil_rgb(image: str | object):
    from PIL import Image

    if isinstance(image, str):
        try:
            pil_image = Image.open(image).convert("RGB")
            pil_image.load()
        except Exception as exc:
            raise ValueError(f"Cannot decode i2va input image {image!r}: {exc}") from exc
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"i2va image must be a path or PIL image, got {type(image).__name__}")
    return pil_image


@torch.no_grad()
def encode_i2va_first_frame(
    image: str | object,
    vae,
    device: torch.device | str,
    height: int | None = None,
    width: int | None = None,
    *,
    max_area: int | None = None,
    divisibility: int = 16,
    world_size: int = 1,
) -> tuple[Tensor, int, int]:
    """VAE-encode a reference image to one latent frame ``(1, H_lat, W_lat, C)``.

    Two modes (K5 parity):
      - ``max_area`` set (and height/width omitted): aspect-preserving resize like
        K5 ``resize_image`` / ``get_first_frame_from_image`` — output HxW from input.
      - ``height`` + ``width`` set: cover-resize + center-crop to that canvas
        (K5 ``encode_i2va_first_frame``).

    Returns ``(latent, pixel_height, pixel_width)``.
    """
    try:
        import torchvision.transforms.functional as TF
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "I2VA image processing requires torchvision. Install it with `pip install torchvision`."
        ) from exc

    pil_image = _load_pil_rgb(image)
    tensor = TF.pil_to_tensor(pil_image).unsqueeze(0)

    if max_area is not None and (height is None or width is None):
        tensor, _ = resize_image(
            tensor, max_area=max_area, divisibility=divisibility, world_size=world_size
        )
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    else:
        if height is None or width is None:
            raise ValueError("encode_i2va_first_frame needs height/width or max_area")
        src_h, src_w = tensor.shape[-2:]
        scale = min(src_h / height, src_w / width)
        tensor = TF.resize(tensor, (int(src_h / scale), int(src_w / scale)))
        cur_h, cur_w = tensor.shape[-2:]
        tensor = TF.crop(
            tensor, (cur_h - height) // 2, (cur_w - width) // 2, height, width
        )

    tensor = tensor / 127.5 - 1.0
    vae_dtype = next(vae.parameters()).dtype
    # (1, 3, H, W) → (1, 3, 1, H, W) video layout for Hunyuan VAE
    tensor = tensor.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
    latent = vae.encode(tensor, opt_tiling=False).latent_dist.sample()
    latent = latent.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
    return latent, int(height), int(width)


def _append_i2va_tail_condition_batch(
    latent_visual: Tensor,
    first_frames: Tensor,
    batch_size: int,
    video_duration: int,
):
    if latent_visual.shape[0] != batch_size or latent_visual.shape[1] != video_duration:
        raise ValueError(
            "generated visual latent shape mismatch: expected "
            f"({batch_size}, {video_duration}, H, W, C), got {tuple(latent_visual.shape)}"
        )

    _, _, height, width, dim = latent_visual.shape
    first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
    expected = (batch_size, height, width, dim)
    if tuple(first_frames.shape) != expected:
        raise ValueError(
            f"first-frame latent shape mismatch: expected {expected}, "
            f"got {tuple(first_frames.shape)}"
        )

    latent_visual = torch.cat([latent_visual, first_frames[:, None]], dim=1)
    token_types = torch.cat(
        [
            torch.zeros((batch_size, video_duration), dtype=torch.long, device=latent_visual.device),
            torch.ones((batch_size, 1), dtype=torch.long, device=latent_visual.device),
        ],
        dim=1,
    )
    return latent_visual, token_types, token_types == 0


def append_i2va_tail_condition(
    latent_visual: Tensor,
    first_frames: Tensor,
    batch_size: int,
    video_duration: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append one clean first-frame latent and build generated/reference token ids.

    Returns ``(latent, token_types, generated_mask)`` where ``token_types`` is
    0=generated / 1=reference and ``generated_mask`` selects frames to decode.
    """
    if latent_visual.ndim == 5:
        return _append_i2va_tail_condition_batch(
            latent_visual, first_frames, batch_size, video_duration
        )

    _, height, width, dim = latent_visual.shape
    first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
    expected = (batch_size, height, width, dim)
    if tuple(first_frames.shape) != expected:
        raise ValueError(
            f"first-frame latent shape mismatch: expected {expected}, got {tuple(first_frames.shape)}"
        )
    if latent_visual.shape[0] != batch_size * video_duration:
        raise ValueError(
            "generated visual latent length mismatch: expected "
            f"{batch_size * video_duration}, got {latent_visual.shape[0]}"
        )

    latent_visual = torch.cat(
        [
            latent_visual.reshape(batch_size, video_duration, height, width, dim),
            first_frames[:, None],
        ],
        dim=1,
    ).reshape(batch_size * (video_duration + 1), height, width, dim)

    token_types = torch.cat([
        torch.zeros(video_duration, dtype=torch.long, device=latent_visual.device),
        torch.ones(1, dtype=torch.long, device=latent_visual.device),
    ]).repeat(batch_size)
    return latent_visual, token_types, token_types == 0
