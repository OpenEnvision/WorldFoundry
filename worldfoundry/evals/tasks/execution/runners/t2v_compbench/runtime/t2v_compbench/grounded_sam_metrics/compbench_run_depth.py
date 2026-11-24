import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dpt import DepthAnything
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.grounded_sam_metrics.video_io import (
    write_video,
)


class Video_preprocess:
    def __init__(self):
        pass

    def extract_frames(self, video_path, num_frames=16):
        frames = []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames", total_frames)
        if total_frames <= num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def rgb_to_yuv(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def frames_to_video(self, frames, output_path, fps=8):
        yuv_frames = [self.rgb_to_yuv(frame) for frame in frames]
        video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
        write_video(output_path, video_tensor, fps, video_codec="h264", options={"crf": "18"})

    def convert_video(self, input_path, output_path, num_frames):
        frames = self.extract_frames(input_path, num_frames=num_frames)
        self.frames_to_video(frames, output_path)

    def merge_grid(self, image_list):
        # Concatenate the images horizontally
        row1 = np.concatenate((image_list[0], image_list[1]), axis=1)
        row2 = np.concatenate((image_list[2], image_list[3]), axis=1)
        row3 = np.concatenate((image_list[4], image_list[5]), axis=1)
        # Concatenate the rows vertically
        grid = np.concatenate((row1, row2, row3), axis=0)
        return grid

    def read_video_path(self, video_path):
        if os.path.isdir(video_path):  # if video_path is a list of videos
            video = os.listdir(video_path)
        elif os.path.isfile(video_path):  # else if video_path is a single video
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        video.sort()
        return video, video_path

    def convert_video_to_frames(self, video_path, num_frames=16):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to {num_frames} frames from path:", video_path)

        output_path = os.path.join(os.path.dirname(video_path), "frames", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            vid_id = v.split(".")[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path, num_frames=num_frames)
            for frame_count, frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f"{vid_id}_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, frame)
        print("finish converting from path: ", video_path)
        print("video frames stored in: ", output_path)
        return output_path

    def convert_video_to_standard_video(self, video_path, num_frames):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to video with 16 frames from path:", video_path)

        output_path = os.path.join(os.path.dirname(video_path), "video_standard", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            v_mp4 = v.split(".")[0] + ".mp4"
            self.convert_video(os.path.join(video_path, f"{v}"), os.path.join(output_path, f"{v_mp4}"), num_frames)
        print("finish converting from path: ", video_path)
        print("standard video stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path, num_image=6):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to image grid with 6 frames from path:", video_path)

        output_path = os.path.join(os.path.dirname(video_path), "image_grid", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path)
            frame_indices = np.linspace(
                0, len(frames) - 1, num_image, dtype=int
            )  # take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            grid_filename = os.path.join(output_path, f"{vid_id}.png")
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path


def run_depth(video_path, t2v_model, output_dir, meta_file, encoder="vitl", pred_only=True, grayscale=True):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    depth_anything = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14").to(DEVICE).eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    vid_process = Video_preprocess()
    with open(meta_file) as json_data:
        prompts = json.load(json_data)

    frame_folder = vid_process.convert_video_to_frames(video_path)
    print("frame_folder: ", frame_folder)

    output_dir = os.path.join(output_dir, t2v_model)
    videos = os.listdir(frame_folder)
    videos.sort(key=lambda x: int(x))

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(videos)):
        vid = videos[i]
        images = os.listdir(os.path.join(frame_folder, vid))
        images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        num = int(videos[i]) - 1

        _prompt = prompts[num]["prompt"]
        spatial = prompts[num]["spatial"]  # A is on the left of B
        _phrase_0 = prompts[num]["object_1"]
        _phrase_1 = prompts[num]["object_2"]
        spatial = prompts[num]["spatial"]
        if spatial not in ["left", "right", "above", "on", "under", "below", "in front of", "behind"]:
            print(spatial, "spatial not included!!!, index: ", vid)
            break

        if spatial in ["behind", "in front of"]:
            for frame in images:
                filename = os.path.join(frame_folder, vid, frame)
                raw_image = cv2.imread(filename)
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

                h, w = image.shape[:2]

                image = transform({"image": image})["image"]
                image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    depth = depth_anything(image)

                depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

                depth = depth.cpu().numpy().astype(np.uint8)

                if grayscale:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

                filename = os.path.basename(filename)

                if pred_only:
                    os.makedirs(os.path.join(output_dir, vid), exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir, vid, frame), depth)
    return frame_folder
