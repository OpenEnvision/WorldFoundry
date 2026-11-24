import math
import os

import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu
from PIL import Image


def read_video_frames(
    video_path,
    process_length,
    stride,
    max_res,
    dataset="open",
    width: int | None = None,
    height: int | None = None,
    center_crop: bool = False,
):
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid0 = VideoReader(video_path, ctx=cpu(0))
        first0 = vid0.get_batch([0]).asnumpy()
        original_height, original_width = first0.shape[1:3]
        print("==> original video shape: ", (len(vid0), original_height, original_width, first0.shape[-1]))

        decode_width: int | None = None
        decode_height: int | None = None
        if bool(center_crop) and width is not None and height is not None:
            target_w = int(width)
            target_h = int(height)

            if original_width <= 0 or original_height <= 0:
                decode_width, decode_height = target_w, target_h
            else:
                scale_cover_w = float(target_w) / float(original_width)
                scale_cover_h = float(target_h) / float(original_height)
                # Minimal-cover resize: decode just large enough so we can center-crop to (target_h, target_w)
                # without distortion (no pad / no stretch).
                scale = max(scale_cover_w, scale_cover_h)

                decode_width = math.ceil(float(original_width) * scale)
                decode_height = math.ceil(float(original_height) * scale)

                # Prefer even decode sizes for video codecs / decoders.
                decode_width = max(16, ((decode_width + 1) // 2) * 2)
                decode_height = max(16, ((decode_height + 1) // 2) * 2)
        elif width is not None and height is not None:
            decode_width, decode_height = int(width), int(height)
        else:
            max_res_i = int(max_res) if max_res is not None else 0
            if max_res_i <= 0:
                max_res_i = 1024
            scale = min(1.0, float(max_res_i) / float(max(original_height, original_width)))
            decode_height = max(16, round(original_height * scale))
            decode_width = max(16, round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=int(decode_width), height=int(decode_height))
    else:
        vid = VideoReader(video_path, ctx=cpu(0), width=int(width), height=int(height))

    frames_idx = list(range(0, len(vid), stride))
    print(f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}")
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(f"==> final processing shape: {(len(frames_idx), *vid.get_batch([0]).shape[1:])}")
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    if bool(center_crop) and width is not None and height is not None:
        target_w = int(width)
        target_h = int(height)
        _, h, w, _ = frames.shape
        if w < target_w or h < target_h:
            raise ValueError(f"center_crop requires decoded frames to cover target {target_h}x{target_w}, got {h}x{w}")
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        frames = frames[:, top : top + target_h, left : left + target_w, :]

    return frames


def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data, strict=False)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=fps, video_codec="h264", options={"crf": "10"})
