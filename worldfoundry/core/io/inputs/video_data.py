"""Lazy PIL-based video/image dataset helpers for diffusion pipelines.

This module complements ``worldfoundry.core.media.codecs.video``, which provides numpy/torch
read/write primitives (``load_video_frames``, ``write_video``, etc.).  The APIs
here load frames on demand as PIL images and write PIL frame sequences.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────
# Lazy PIL readers — decode one frame at a time; natural sort for image folders
# ──────────────────────────────────────────────────────────────────────────


class LowMemoryVideo:
    """Lazy video reader that loads frames on demand."""

    def __init__(self, file_name: str):
        """Open an imageio reader; frames stay on disk until ``__getitem__``."""

        import imageio

        self.reader = imageio.get_reader(file_name)

    def __len__(self) -> int:
        """Frame count from the container; may be expensive on some codecs."""

        return self.reader.count_frames()

    def __getitem__(self, item: int) -> Image.Image:
        """Decode frame *item* as RGB PIL (drops alpha)."""

        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        """Close the imageio reader if the object is collected."""

        self.reader.close()


def split_file_name(file_name):
    """Tokenize a name into strings and ints so ``2.png`` sorts before ``10.png``."""

    result = []
    number = -1
    for i in file_name:
        if ord(i) >= ord("0") and ord(i) <= ord("9"):
            if number == -1:
                number = 0
            number = number * 10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    return tuple(result)


def search_for_images(folder):
    """Return naturally sorted JPG/PNG paths from one image-sequence folder.

    Embedded digit runs are compared numerically, so ``2.png`` sorts before
    ``10.png``. The search is non-recursive.
    """
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    return [os.path.join(folder, i) for i in file_list]


class LowMemoryImageFolder:
    """Lazy image-folder reader that loads frames on demand."""

    def __init__(self, folder, file_list=None):
        """Index JPG/PNG paths; *file_list* if given is joined under *folder* as-is."""

        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]

    def __len__(self):
        """Number of indexed image files."""

        return len(self.file_list)

    def __getitem__(self, item):
        """Open one image as RGB; the file handle is released after convert."""

        return Image.open(self.file_list[item]).convert("RGB")

    def __del__(self):
        """No open handles to close; kept for API symmetry with :class:`LowMemoryVideo`."""

        pass


def crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    """Center-crop and resize a PIL image to ``(width, height)``."""
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left : left + croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left : left + croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image


class VideoData:
    """Lazy video or image-folder dataset with optional resize."""

    def __init__(self, video_file=None, image_folder=None, height=None, width=None, **kwargs):
        """Open exactly one of *video_file* or *image_folder*; raise if both/neither."""

        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        """Materialize every frame (defeats laziness; used by small debug clips)."""

        return [self.__getitem__(i) for i in range(self.__len__())]

    def set_length(self, length):
        """Cap or override reported length without truncating the underlying reader."""

        self.length = length

    def set_shape(self, height, width):
        """Set the optional crop-resize target; ``None`` keeps native size."""

        self.height = height
        self.width = width

    def __len__(self):
        """Override length if set; otherwise the underlying reader length."""

        if self.length is None:
            return len(self.data)
        return self.length

    def shape(self):
        """Return ``(H, W)``; probes frame 0 when no explicit size was set."""

        if self.height is not None and self.width is not None:
            return self.height, self.width
        frame = self.__getitem__(0)
        if isinstance(frame, Image.Image):
            width, height = frame.size
            return height, width
        height, width, _ = np.asarray(frame).shape
        return height, width

    def __getitem__(self, item):
        """Fetch one PIL frame and crop-resize only when the stored size differs."""

        frame = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self):
        """Underlying readers close themselves; this is a no-op hook."""

        pass

    def save_images(self, folder):
        """Dump every frame as ``{i}.png`` (0-based) under *folder*."""

        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            self.__getitem__(i).save(os.path.join(folder, f"{i}.png"))


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    """Write a sequence of PIL frames to a video file."""
    import imageio

    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        writer.append_data(np.array(frame))
    writer.close()


def save_frames(frames, save_path):
    """Write a sequence of PIL frames to numbered PNG files."""
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"{i}.png"))


def merge_video_audio(video_path: str, audio_path: str) -> None:
    """Merge video and audio with ffmpeg; overwrite ``video_path`` on success.

    Raises on failure (missing inputs, ffmpeg errors). The temporary mux
    output is always removed, so a failed merge leaves the original video
    untouched. Historically this function swallowed its own errors and
    returned normally, which made callers treat missing audio tracks as
    success.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            temp_output,
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg execute failed: {result.stderr}")
        shutil.move(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)


def save_video_with_audio(frames, save_path, audio_path, fps=16, quality=9, ffmpeg_params=None):
    """Write PIL frames, then mux an external audio track with ffmpeg.

    Args:
        frames: Iterable of PIL images.
        save_path: Destination video path, replaced after muxing.
        audio_path: Existing audio file.
        fps: Output frame rate.
        quality: ImageIO encoder quality.
        ffmpeg_params: Optional ImageIO ffmpeg arguments.

    Notes:
        ``merge_video_audio`` raises on ffmpeg failures after removing its
        temporary file, so an exception here means ``save_path`` still holds
        the silent video.
    """
    save_video(frames, save_path, fps, quality, ffmpeg_params)
    merge_video_audio(save_path, audio_path)


__all__ = [
    "LowMemoryImageFolder",
    "LowMemoryVideo",
    "VideoData",
    "crop_and_resize",
    "merge_video_audio",
    "save_frames",
    "save_video",
    "save_video_with_audio",
    "search_for_images",
    "split_file_name",
]
