import torch
import torch.nn as nn
from PIL import Image

# isort: split
from q_align.model.builder import load_pretrained_model

# isort: split
from q_align.constants import IMAGE_TOKEN_INDEX

# isort: split
from q_align.mm_utils import tokenizer_image_token

# isort: split
import json
import os

import numpy as np
from decord import VideoReader
from tqdm import tqdm


class QAlignVideoScorer(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = (
            "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"
        )

        self.preferential_ids_ = [
            id_[1] for id_ in tokenizer(["excellent", "good", "fair", "poor", "bad"])["input_ids"]
        ]
        self.weight_tensor = torch.Tensor([1, 0.75, 0.5, 0.25, 0.0]).half().to(model.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, video: list[list[Image.Image]]):
        video = [
            [self.expand2square(frame, tuple(int(x * 255) for x in self.image_processor.image_mean)) for frame in vid]
            for vid in video
        ]
        with torch.inference_mode():
            video_tensors = [
                self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].half().to(self.model.device)
                for vid in video
            ]
            input_tensors = self.input_ids.repeat(len(video_tensors), 1)
            output = self.model(input_tensors, images=video_tensors)
            output_logits = output["logits"][:, -1, self.preferential_ids_]
            return (
                output_logits,
                torch.softmax(output_logits, -1),
                torch.softmax(output_logits, -1) @ self.weight_tensor,
            )


# Read video in sliding window manner, splitting video into segments with number of frames as window_size
def load_video_sliding_window(video_file, window_size=5):
    vr = VideoReader(video_file)
    total_frames = len(vr)
    frames_by_group = []

    # Calculate the left and right extension of the window
    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend

    for current_frame in range(total_frames):
        # Calculate the start and end frame of the window
        start_frame = max(0, current_frame - left_extend)
        end_frame = min(total_frames, current_frame + right_extend + 1)

        frame_indices = list(range(start_frame, end_frame))

        # If there are not enough frames, pad frames on both ends
        while len(frame_indices) < window_size:
            if start_frame == 0:
                frame_indices.append(frame_indices[-1])
            else:
                frame_indices.insert(0, frame_indices[0])

        frames = vr.get_batch(frame_indices).asnumpy()

        # Special handling for the beginning frames to ensure consistency with window_size frames
        if current_frame < left_extend:
            frames_by_group.append([Image.fromarray(frames[0])] * window_size)
        else:
            frames_by_group.append([Image.fromarray(frame) for frame in frames])

    return frames_by_group


# Set threshold for image score fluctuation based on camera movement amplitude
def set_threshold(camera_movement):
    if camera_movement is None:
        return 0.01
    if camera_movement < 0.1:
        return 0.01
    elif 0.1 <= camera_movement < 0.3:
        return 0.015
    elif 0.3 <= camera_movement < 0.5:
        return 0.025
    else:  # camera_movement >= 0.5
        return 0.03


# Get frames with poor quality artifacts based on score differences
def get_artifacts_frames(scores, threshold=0.025):
    # Calculate score differences between adjacent frames
    score_diffs = np.abs(np.diff(scores))

    # Identify frames where score differences exceed the threshold
    artifact_indices = np.where(score_diffs > threshold)[0]

    # Return both the current frame and the next frame as significant score difference may be caused by either
    artifacts_frames = np.unique(np.concatenate([artifact_indices, artifact_indices + 1]))

    return artifacts_frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("WORLDFOUNDRY_QALIGN_ONE_ALIGN_MODEL", ".cache/q-future/one-align"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--meta_info_path", type=str, default="meta_info.json")
    args = parser.parse_args()

    scorer = QAlignVideoScorer(pretrained=args.model_path, device=args.device)

    # Read video information from JSON file
    with open(args.meta_info_path) as f:
        meta_infos = json.load(f)

    # Iterate over videos for Q-Align scoring
    for meta_info in tqdm(meta_infos, desc="Q-Align Scoring"):
        # Get video filepath
        video_file = meta_info["filepath"]
        # Get video scores
        _, _, scores = scorer(load_video_sliding_window(video_file, args.window_size))
        scores = scores.tolist()
        # Set score fluctuation threshold based on video motion amplitude
        threshold = set_threshold(meta_info["perceptible_amplitude_score"])
        # Get frames with issues based on score and threshold
        artifacts_frames = get_artifacts_frames(scores, threshold)
        # Calculate final score: proportion of normal frames out of total frames
        final_score = 1 - len(artifacts_frames) / len(scores)
        # Save score to JSON file
        meta_info["motion_smoothness_score"] = final_score
        # Write data back to JSON file
        with open(args.meta_info_path, "w") as f:
            json.dump(meta_infos, f, indent=2)
