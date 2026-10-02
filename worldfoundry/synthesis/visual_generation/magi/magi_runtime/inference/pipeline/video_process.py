# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import tempfile

import ffmpeg
import torch
from einops import rearrange

import worldfoundry.core.distributed.model_parallel.groups as mpu
from inference.common.config import MagiConfig
from inference.model.vae import AutoModel, DiagonalGaussianDistribution, VideoTokenizerABC
from worldfoundry.core.distributed.runtime.logging import distributed_logger as magi_logger


############################################
# VaeHelper
###########################################
class SingletonMeta(type):
    """
    Singleton metaclass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class VaeHelper(metaclass=SingletonMeta):
    def __init__(self):
        # Initialize cache dict
        if not hasattr(self, "vae_cache_dict"):
            self.vae_cache_dict = {}

    @staticmethod
    def get_vae(vae_ckpt: str) -> VideoTokenizerABC:
        """
        Load a pretrained VAE model.

        Args:
            vae_ckpt (str): Path to the pretrained VAE checkpoint.

        Returns:
            VideoTokenizerABC: Pretrained VAE model.
        """
        vae_helper = VaeHelper()

        if vae_ckpt not in vae_helper.vae_cache_dict:
            vae = AutoModel.from_pretrained(vae_ckpt)
            vae.encode = vae_helper.patch_vae_encode.__get__(vae)
            vae.cuda()
            vae.eval()
            vae.bfloat16()
            if os.environ.get("OFFLOAD_VAE_CACHE") == "true":
                return vae
            vae_helper.vae_cache_dict[vae_ckpt] = vae
        return vae_helper.vae_cache_dict[vae_ckpt]

    @staticmethod
    @torch.no_grad()
    def patch_vae_encode(vae: callable, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input video.

        Args:
            x (torch.Tensor): Input video tensor with shape (N, C, T, H, W).
            sample_posterior (bool): Whether to sample from the posterior.

        Returns:
            torch.Tensor: Encoded tensor with additional information.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input x to be torch.Tensor, but got {type(x)}.")
        if len(x.shape) != 5:
            raise ValueError(f"Expected input tensor x to have shape (N, C, T, H, W), but got {x.shape}.")

        if not hasattr(vae, "encoder") or not callable(vae.encoder):
            raise AttributeError("Encoder is not defined or callable. Please initialize 'self.encoder'.")

        # for setting vae encoding to deterministic
        N, C, T, H, W = x.shape
        if T == 1:
            x = x.expand(-1, -1, 4, -1, -1)
            x = vae.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()

            return z[:, :, :1, :, :].type(x.dtype)
        else:
            x = vae.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()

            return z.type(x.dtype)

    @staticmethod
    def encode(
        video: torch.Tensor,
        vae: VideoTokenizerABC,
        tile_sample_min_length: int = 16,
        tile_sample_min_height: int = 256,
        tile_sample_min_width: int = 256,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        allow_spatial_tiling: bool = True,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """
        Encode the input tensor.
        Args:
            video (torch.Tensor): Input tensor with shape (N, T, C, H, W).
            vae (VideoTokenizerABC): Pretrained VAE model.
            tile_sample_min_length (int): Minimum length of the tile sample.
            tile_sample_min_height (int): Minimum height of the tile sample.
            tile_sample_min_width (int): Minimum width of the tile sample.
            spatial_tile_overlap_factor (float): Spatial tile overlap factor.
            allow_spatial_tiling (bool): Allow spatial tiling.
            parallel_group (ProcessGroup): Distributed encoding group.
        Returns:
            torch.Tensor: Encoded tensor.
        """
        assert video.dim() == 5, f"Expected input tensor to have shape (N, T, C, H, W), but got {video.shape}."
        video = video.cuda()
        video = (video / 127.5) - 1.0
        video = video.bfloat16()
        moments = vae.tiled_encode_3d(
            video,
            tile_sample_min_length=tile_sample_min_length,
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            spatial_tile_overlap_factor=spatial_tile_overlap_factor,
            temporal_tile_overlap_factor=temporal_tile_overlap_factor,
            allow_spatial_tiling=allow_spatial_tiling,
            parallel_group=parallel_group,
        )

        return moments

    @staticmethod
    def decode(
        chunk: torch.Tensor,
        vae: VideoTokenizerABC,
        tile_sample_min_height: int = 256,
        tile_sample_min_width: int = 256,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        tile_sample_min_length: int = 16,
        allow_spatial_tiling: bool = True,
        uint8_output: bool = True,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """
        Decode the input tensor.
        Args:
            chunk (torch.Tensor): Input tensor with shape (N, C, T, H, W).
            vae (VideoTokenizerABC): Pretrained VAE model.
            tile_sample_min_length (int): Minimum length of the tile sample.
            tile_sample_min_height (int): Minimum height of the tile sample.
            tile_sample_min_width (int): Minimum width of the tile sample.
            spatial_tile_overlap_factor (float): Spatial tile overlap factor.
            temporal_tile_overlap_factor (float): Temporal tile overlap factor.
            allow_spatial_tiling (bool): Allow spatial tiling.
            uint8_output (bool): Whether to output uint8 tensor.
            parallel_group (ProcessGroup): Distributed decoding group.
        Returns:
            torch.Tensor: Decoded tensor.
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = vae.tiled_decode_3d(
                chunk,
                tile_sample_min_height=tile_sample_min_height,
                tile_sample_min_width=tile_sample_min_width,
                spatial_tile_overlap_factor=spatial_tile_overlap_factor,
                temporal_tile_overlap_factor=temporal_tile_overlap_factor,
                tile_sample_min_length=tile_sample_min_length,
                allow_spatial_tiling=allow_spatial_tiling,
                parallel_group=parallel_group,
            )
        chunk = rearrange(chunk, "b c t h w -> (b t) c h w")
        if uint8_output:
            chunk = (chunk * 127.5) + 127.5
            chunk = chunk.clamp(0, 255)
            chunk = chunk.type(torch.uint8)
        return chunk


############################################
# Process to get prefix video
###########################################


def ffmpeg_i2v(image_path, w=384, h=224, aspect_policy="fit"):
    r = ffmpeg.input("pipe:0", format="image2pipe")
    if aspect_policy == "crop":
        r = r.filter("scale", w, h, force_original_aspect_ratio="increase").filter("crop", w, h)
    elif aspect_policy == "pad":
        r = r.filter("scale", w, h, force_original_aspect_ratio="decrease").filter(
            "pad", w, h, "(ow-iw)/2", "(oh-ih)/2", color="black"
        )
    elif aspect_policy == "fit":
        r = r.filter("scale", w, h)
    else:
        magi_logger.warning(f"Unknown aspect policy: {aspect_policy}, using fit as fallback")
        r = r.filter("scale", w, h)
    image_byte = open(image_path, "rb").read()
    try:
        out, _ = r.output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1).run(
            input=image_byte, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as e:
        print(f"Error occurred: {e.stderr.decode()}")
        raise e

    video = torch.frombuffer(out, dtype=torch.uint8).view(1, h, w, 3)
    return video


def ffmpeg_v2v(video_path, fps, w=384, h=224, prefix_frame=None, prefix_video_max_chunk=5):
    if video_path is None:
        return None
    out, _ = (
        ffmpeg.input(video_path, ss=0, format="mp4")
        .filter("fps", fps=fps)
        .filter("scale", w, h)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", nostdin=None)
        .run(capture_stdout=True, capture_stderr=True)
    )

    video = torch.frombuffer(out, dtype=torch.uint8).view(-1, h, w, 3)

    if prefix_frame is not None:
        return video[:prefix_frame]
    else:
        num_frames_to_read = video.shape[0]
        if num_frames_to_read < fps:
            clip_length = 1
        else:
            PREFIX_VIDEO_MAX_FRAMES = prefix_video_max_chunk * fps
            clip_length = min(num_frames_to_read // fps * fps, PREFIX_VIDEO_MAX_FRAMES)
        return video[-clip_length:]


def save_video_to_disk(video: torch.Tensor, save_path: str, fps: int) -> bytes:
    # TCHW -> THWC
    video = video.permute(0, 2, 3, 1).cpu().numpy()
    _, H, W, _ = video.shape
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video.tobytes())
        temp_file.flush()
        temp_file_path = temp_file.name

    output, _ = (
        ffmpeg.input(temp_file_path, format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}", r=fps)
        .output(save_path, format='mp4', vcodec='libx264', pix_fmt='yuv420p')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    os.remove(temp_file_path)
    return output


def encode_prefix_video(prefix_video, fps, vae_ckpt, scale_factor, parallel_group):
    if prefix_video is None:
        return None
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated before vae encode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved before vae encode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    # THWC -> NCTHW
    prefix_video = prefix_video.permute(3, 0, 1, 2).unsqueeze(0)
    magi_logger.debug(f"prefix_video.shape: {prefix_video.shape}")
    vae_model = VaeHelper.get_vae(vae_ckpt)
    tile_sample_min_length = fps // 2
    prefix_video = VaeHelper.encode(
        prefix_video,
        vae_model,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        spatial_tile_overlap_factor=0.25,
        temporal_tile_overlap_factor=0,
        tile_sample_min_length=tile_sample_min_length,
        allow_spatial_tiling=True,
        parallel_group=parallel_group,
    )
    prefix_video = prefix_video * scale_factor
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated after vae encode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved after vae encode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )
    return prefix_video


def process_image(image_path: str, config: MagiConfig) -> torch.Tensor:
    prefix_video = ffmpeg_i2v(image_path, w=config.runtime_config.video_size_w, h=config.runtime_config.video_size_h)
    prefix_video = encode_prefix_video(
        prefix_video,
        config.runtime_config.fps,
        config.runtime_config.vae_pretrained,
        config.runtime_config.scale_factor,
        parallel_group=mpu.get_tp_group(with_context_parallel=True),
    )
    return prefix_video


def process_prefix_video(prefix_video_path: str, config: MagiConfig) -> torch.Tensor:
    prefix_video = ffmpeg_v2v(
        prefix_video_path,
        fps=config.runtime_config.fps,
        prefix_frame=32,
        w=config.runtime_config.video_size_w,
        h=config.runtime_config.video_size_h,
    )
    prefix_video = encode_prefix_video(
        prefix_video,
        config.runtime_config.fps,
        config.runtime_config.vae_pretrained,
        config.runtime_config.scale_factor,
        parallel_group=mpu.get_tp_group(with_context_parallel=True),
    )
    return prefix_video


############################################
# Process to get final video
############################################
def decode_chunk(chunk, vae_ckpt, scale_factor, tile_sample_min_length, parallel_group):
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated before vae decode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved before vae decode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    vae_model = VaeHelper.get_vae(vae_ckpt)
    decoded_chunk = VaeHelper.decode(
        chunk / scale_factor,
        vae_model,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        spatial_tile_overlap_factor=0.25,
        temporal_tile_overlap_factor=0,
        tile_sample_min_length=tile_sample_min_length,
        allow_spatial_tiling=True,
        parallel_group=parallel_group,
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated after vae decode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved after vae decode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )
    return decoded_chunk


def post_chunk_process(chunk: torch.Tensor, config: MagiConfig):
    # A one-latent VAE tile is decoded as a still image. Keep decode tiles at
    # least two latents long even for low output frame rates.
    tile_sample_min_length = max(
        config.runtime_config.fps // 2,
        2 * config.runtime_config.temporal_downsample_factor,
    )
    chunk = decode_chunk(
        chunk,
        config.runtime_config.vae_pretrained,
        config.runtime_config.scale_factor,
        tile_sample_min_length,
        parallel_group=mpu.get_tp_group(with_context_parallel=True),
    )
    gc.collect()
    torch.cuda.empty_cache()
    return chunk
