# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan video geometry helpers (latent grid, frame alignment, debug I/O).

Wan-specific latent-grid and artifact conventions remain with the Wan model
family. Used when a resize, tile, or mask must match the DiT patch
grid, and when official Wan recipes dump a debug mp4/png.

Public surface:

- :func:`best_output_size` — largest ``dw``×``dh``-aligned rectangle
  under *expected_area* that stays close to the source aspect.
- :func:`masks_like` — paired latent masks for Wan schedulers
  (first-slice vs YUME trailing-slice).
- :func:`save_video` / :func:`save_image` / :func:`merge_video_audio`
  — debug artifact writers (errors are logged, not always raised;
  that is the vendored contract).
"""

import binascii
import logging
import os
import os.path as osp
import shutil
import subprocess

import imageio
import torch
import torchvision

__all__ = ['save_video', 'save_image']

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Latent-grid helpers and debug writers — vendored Wan contract (log, not raise)
# ──────────────────────────────────────────────────────────────────────────


def rand_name(length=8, suffix=''):
    """Return a hex filename stem of *length* random bytes, plus *suffix*."""
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        logger.info("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logger.info(f"Merge completed, saved to {video_path}")

    finally:
        # Failures propagate to the caller; only the temp mux output is
        # cleaned up (the original video stays in place).
        if os.path.exists(temp_output):
            os.remove(temp_output)


def save_video(tensor,
               save_file=None,
               fps=30,
               suffix='.mp4',
               nrow=8,
               normalize=True,
               value_range=(-1, 1)):
    """Write a ``[C, T, H, W]`` (or batched grid) tensor as an H.264 mp4.

    Each time slice is passed through ``make_grid`` so a batch becomes
    one panel. Failures are logged and swallowed (vendored Wan contract).
    When *save_file* is omitted the file lands in ``/tmp/<rand>.mp4``.
    """
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    try:
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
                             dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(
            cache_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        # Vendored contract swallows the error; log at ERROR so a missing
        # output file is at least visible in logs.
        logger.error(f'save_video failed, error: {e}')


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    """Write a tensor as an image via ``torchvision.utils.save_image``.

    Unknown suffixes become ``.png``. Failures are logged and the
    function returns ``None`` (vendored Wan contract).
    """
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        torchvision.utils.save_image(
            tensor,
            save_file,
            nrow=nrow,
            normalize=normalize,
            value_range=value_range)
        return save_file
    except Exception as e:
        # Vendored contract swallows the error (returns None); log at ERROR.
        logger.error(f'save_image failed, error: {e}')


def masks_like(tensor, zero=False, generator=None, p=0.2, current_latent_num=None):
    """Build paired latent masks for Wan-family schedulers.

    Args:
        tensor: Latent tensor list.
        zero: Whether to apply zero/masked regions.
        generator: Optional torch RNG.
        p: Probability of applying stochastic mask when ``generator`` is set.
        current_latent_num: When set, mask ``[:, :-N]`` (YUME-1.5). Otherwise mask
            only the first latent slice (default Wan behavior).
    """
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u_target = u[:, :-current_latent_num] if current_latent_num else u[:, 0]
                    v_target = v[:, :-current_latent_num] if current_latent_num else v[:, 0]
                    u_target[:] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u_target).exp()
                    v_target[:] = torch.zeros_like(v_target)
        else:
            for u, v in zip(out1, out2):
                if current_latent_num:
                    u[:, :-current_latent_num] = torch.zeros_like(u[:, :-current_latent_num])
                    v[:, :-current_latent_num] = torch.zeros_like(v[:, :-current_latent_num])
                else:
                    u[:, 0] = torch.zeros_like(u[:, 0])
                    v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    """Largest ``dw``×``dh``-aligned size under *expected_area* near *w*/*h*.

    Tries snapping width first vs height first and keeps the candidate
    whose aspect is closer to the source. Used to pick a latent-grid-
    aligned decode size without exceeding a pixel budget.
    """
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2
