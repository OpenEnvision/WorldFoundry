from __future__ import annotations
from random import randint
from typing import Dict, Optional, List
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from einops import rearrange

# Lazy import torchcodec to avoid environment issues at module load
_VideoDecoder = None

to_tensor = transforms.ToTensor()



def filter_video_files(file_names: List, xdof: bool = False) -> List:
    if xdof:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
            and "320_240" in str(f).lower()
        ]
    else:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
        ]


class VideoDataset(Dataset):
    def __init__(
        self,
        randomize: bool = True,
        num_frames: int = 13,
        fps: int = 30,
        *,
        seek_mode: str = "exact",
        ffmpeg_threads: int = 1,
        video_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.randomize = randomize
        self.num_frames = num_frames
        self.fps = fps
        self.seek_mode = seek_mode
        self.ffmpeg_threads = ffmpeg_threads

        # Get all the file path based on the split path
        mp4_list = list(Path(video_root).rglob("*.mp4"))
        if len(mp4_list) > 0:
            if "xdof" in video_root:
                self.episodes = filter_video_files(mp4_list, xdof=True)
            else:
                self.episodes = filter_video_files(mp4_list)
        else:
            raise ValueError(f"No video files found in {video_root}")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict:
        while True:
            video_path = self.episodes[idx]
            try:
                video_pil, start_idx = self._load_video_slice_torchcodec(
                    video_path,
                    num_frames=self.num_frames,
                )
                T = len(video_pil)

                video_frames = [to_tensor(img) for img in video_pil]
                video_tensor = torch.stack(video_frames, dim=0)
                video_tensor = torch.clamp(video_tensor * 255.0, 0, 255).to(torch.uint8)
                video_tensor = video_tensor.transpose(0, 1)  # (T, C, H, W) -> (C, T, H, W)
                target_ratio = 640 / 480
                if video_tensor.shape[3] / video_tensor.shape[2] > target_ratio:
                    target_height = video_tensor.shape[2]
                    target_width = int(video_tensor.shape[2] * target_ratio)
                elif video_tensor.shape[3] / video_tensor.shape[2] < target_ratio:
                    target_height = int(video_tensor.shape[3] / target_ratio)
                    target_width = video_tensor.shape[3]
                else:
                    target_height = video_tensor.shape[2]
                    target_width = video_tensor.shape[3]
                h_crop = (video_tensor.shape[2] - target_height) // 2
                w_crop = (video_tensor.shape[3] - target_width) // 2
                video_tensor = video_tensor[:, :, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
                video_tensor = F.interpolate(video_tensor, (480, 640), mode="bilinear")

                lam_frames = F.interpolate(video_tensor, (240, 320), mode="bilinear")
                lam_frames = torch.clamp(lam_frames / 255.0, 0, 1)
                lam_frames = torch.repeat_interleave(lam_frames, 2, dim=1)[:, 1:-1, :, :]
                lam_frames = rearrange(lam_frames, "c t h w -> t h w c")

                gt_actions = torch.zeros(self.num_frames - 1, 352, dtype=torch.float32)
                latent_actions = torch.ones(self.num_frames - 1, 32, dtype=torch.float32)
                action_seq = torch.cat([gt_actions, latent_actions], dim=-1)
                
                key = torch.ones((1, 29), dtype=torch.float32) * idx

                out: Dict[str, object] = {
                    "video": video_tensor,
                    "lam_video": lam_frames,
                    "action": action_seq,
                    "dataset": "human_video",
                    "fps": self.fps,
                    "num_frames": self.num_frames,

                    "__key__": key,
                    "padding_mask": torch.zeros(1, 256, 256).cuda(),
                    "image_size": 256 * torch.ones(4).cuda(),
                    "ai_caption": "",
                }
                return out
            except Exception as e:
                print("Error loading sample, retrying with a different index...")
                print(e)
                idx = randint(0, len(self) - 1)

    def _load_video_slice_torchcodec(self, video_path: str, num_frames: int) -> tuple[list[Image.Image], int]:
        """
        Decode exactly `num_frames` contiguous frames using TorchCodec.
        Returns (list[PIL.Image] length T, start_index).
        Optimized for high resolution and long videos.
        """
        global _VideoDecoder
        if _VideoDecoder is None:
            from torchcodec.decoders import VideoDecoder as _VD
            _VideoDecoder = _VD
        
        # Use more threads for better performance (default was 1, which is too low)
        # For high-res videos, more threads significantly improve decoding speed
        num_threads = max(self.ffmpeg_threads, 4) if self.ffmpeg_threads == 1 else self.ffmpeg_threads
        
        decoder = _VideoDecoder(
            video_path,
            dimension_order="NHWC",
            num_ffmpeg_threads=num_threads,
            seek_mode=self.seek_mode,
        )

        total = len(decoder)
        if total < num_frames:
            raise ValueError(f"Video shorter than requested window: {total} < {num_frames}, {video_path}")

        start = randint(0, total - num_frames - 1) if self.randomize else 0
        stop = start + num_frames

        batch = decoder.get_frames_in_range(start, stop).data
        if batch.device.type != "cpu":
            batch = batch.to("cpu", non_blocking=True)

        # Optimized batch conversion: convert entire tensor to numpy once, then to PIL
        # This is much faster than converting frame-by-frame, especially for high-res videos
        batch_np = batch.numpy()  # Shape: (T, H, W, C)
        
        # Ensure uint8 format for PIL (handle both uint8 and float inputs)
        if batch_np.dtype != "uint8":
            # Clamp values to [0, 255] range if float
            batch_np = batch_np.clip(0, 255).astype("uint8")
        
        # Batch convert all frames - much faster than individual conversions
        # Pre-allocate list and use efficient numpy slicing
        pil_list = [Image.fromarray(batch_np[i], mode="RGB") for i in range(batch_np.shape[0])]
        return pil_list, start
