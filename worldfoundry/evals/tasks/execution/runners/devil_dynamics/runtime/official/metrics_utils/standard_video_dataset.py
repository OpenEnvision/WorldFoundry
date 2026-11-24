import os
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize

from worldfoundry.core.media.codecs.video import read_video_torchvision as read_video


def change_fps(video_tensor, fps, target_fps):
    if target_fps == -1:
        return video_tensor

    num_frames, height, width, _ = video_tensor.shape

    if fps > target_fps:
        # Subsample frames with an integer stride.
        interval = int(fps / target_fps)
        output_tensor = video_tensor[::interval]
    else:
        # Repeat each frame an integer number of times.
        interval = int(target_fps / fps)
        output_frames = []
        for frame in video_tensor:
            output_frames.append(frame)
            for _ in range(interval - 1):
                output_frames.append(frame)
        output_tensor = torch.stack(output_frames)
    return output_tensor


def resize_long_side(image_tensor, target_long_side):
    """
    Resizes a tensor image to have the specified size for its longest side while maintaining the aspect ratio.

    Args:
        image_tensor (torch.Tensor): Input image batch with shape (N, C, H, W).
        target_long_side (int): Desired size of the longest side of the resized image.

    Returns:
        torch.Tensor: The resized float32 image batch.
    """
    height, width = image_tensor.shape[-2:]

    if width > height:
        scale = target_long_side / width
        new_width = target_long_side
        new_height = int(height * scale)
    else:
        scale = target_long_side / height
        new_height = target_long_side
        new_width = int(width * scale)

    resized_image = F.interpolate(
        image_tensor.float(), size=(new_height, new_width), mode="bilinear", align_corners=False
    )

    return resized_image


def find_video_files(root_dir):
    """Return the paths of all ``.mp4`` and ``.avi`` files under ``root_dir``, recursively."""
    video_files = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith((".mp4", ".avi")):
                full_path = os.path.join(dirpath, filename)
                video_files.append(full_path)
    return video_files


class StandardVidoDataset(Dataset):
    def __init__(self, video_dir, transform=None, transform_no_resize=None, return_video=True, target_fps=8.0):
        """
        Args:
            video_dir (string): Directory searched recursively for videos; files starting with ``._`` are skipped.
            transform (callable, optional): Transform applied to the (T, C, H, W) float video in [0, 1].
            transform_no_resize (callable, optional): Stored only; ``video_no_resize`` uses ``resize_long_side``.
            return_video (bool): Include the video tensors in each sample.
            target_fps (float): Frame rate the videos are resampled to; ``-1`` keeps the source rate.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.transform_no_resize = transform_no_resize
        video_paths = find_video_files(video_dir)
        self.video_paths = [v for v in video_paths if not os.path.basename(v).startswith("._")]
        self.return_video = return_video
        self.target_fps = target_fps

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        try:
            video, _, video_metas = read_video(video_path, pts_unit="sec")
            video = change_fps(video, video_metas["video_fps"], self.target_fps)
            # interval = round(video_metas['video_fps'] / self.target_fps)
            # if interval > 1:
            #     video = video[::interval]
        except BaseException as e:
            print(video_path)
            print(f"{e}")
            return None
        org_video = video.clone()

        sample = {
            "video_name": os.path.basename(video_path),
            "length": video.shape[0],
            "video_path": video_path,
        }

        if self.return_video:
            video_no_resize = resize_long_side(video.permute(0, 3, 1, 2).float() / 255, 512)
            sample.update({"video_no_resize": video_no_resize})
            video = self.transform(video.permute(0, 3, 1, 2).float() / 255)
            sample.update({"video": video})
            sample.update({"org_video": org_video})

        return sample


def standard_collate_fn(batch):
    video_names = []
    videos = []
    videos_no_resize = []
    video_lengths = []
    org_videos = []
    video_paths = []
    for item in batch:
        if item is not None:  # skip videos that failed to load
            video_names.append(item["video_name"])
            video_lengths.append(item["length"])
            video_paths.append(item["video_path"])

            if "video" in item:
                videos.append(item["video"])
            if "video_no_resize" in item:
                videos_no_resize.append(item["video_no_resize"])
            if "org_video" in item:
                org_videos.append(item["org_video"])

    # Concatenate the (T, C, H, W) videos along the frame dimension.
    if "video" in item:
        videos = torch.cat(videos, dim=0)

    return {
        "video_names": video_names,
        "videos": videos,
        "video_lengths": video_lengths,
        "org_videos": org_videos,
        "videos_no_resize": videos_no_resize,
        "video_paths": video_paths,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect videos with StandardVidoDataset")
    parser.add_argument("video_folder", help="Directory containing input videos")
    args = parser.parse_args()
    video_folder = args.video_folder
    transform = Compose(
        [
            Resize((224, 224)),
        ]
    )

    dataset = StandardVidoDataset(video_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=standard_collate_fn)

    for i, data in enumerate(dataloader):
        videos = data["videos"]
        org_videos = data["org_videos"]
        pdb.set_trace()
        print(f"Batch {i + 1}:")
        print(f"Videos shape: {videos.shape}")
        print(f"len(org_videos): {len(org_videos)}")
        if i == 1:  # inspect two batches only
            break
