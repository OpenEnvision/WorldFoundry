import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


class Video_Dataset:
    def __init__(self, cur_full_info_path):
        self.json_path = cur_full_info_path
        self.annotations = self.load_annotations()

    def load_annotations(self):
        with open(self.json_path) as f:
            return json.load(f)

    def extract_frames(self, video_path, extract_frames_persecond, resize_fx, resize_fy):
        """Extract frames from a video file"""
        frames = []
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return frames

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps / extract_frames_persecond)

        curr_frame = 1
        end_frame = total_frames - 1

        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break

            frame = cv2.resize(frame, None, fx=resize_fx, fy=resize_fx)
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip

        # Get the last frame
        video.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
        success, frame = video.read()
        if success:
            frame = cv2.resize(frame, None, fx=resize_fx, fy=resize_fx)
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()
        return frames

    def process_video(self, videos_dict, extract_frames_persecond=2, resize_fx=1, resize_fy=1):
        """Process videos from a dictionary of model:path pairs"""
        base64Frames = {model: [] for model in videos_dict}

        # Process each video in the dictionary
        for model_name, video_path in videos_dict.items():
            frames = self.extract_frames(video_path, extract_frames_persecond, resize_fx, resize_fy)
            base64Frames[model_name] = frames

        return base64Frames

    def process_video2gridview(self, videos_dict, extract_frames_persecond=8):
        """Process videos for grid view"""
        base64Frames = {model: [] for model in videos_dict}

        def process_video(model_name, video_path):
            frames = []
            video = cv2.VideoCapture(video_path)

            if not video.isOpened():
                print(f"Error: Cannot open video file {video_path}")
                return

            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            frames_to_skip = int(fps / extract_frames_persecond)
            curr_frame = 0

            while curr_frame < total_frames:
                video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                curr_frame += frames_to_skip
                success, frame = video.read()
                if not success:
                    break
                frames.append(frame)
                if len(frames) == extract_frames_persecond:
                    height, width, _ = frames[0].shape
                    grid_image = np.zeros((height, extract_frames_persecond * width, 3))

                    for j in range(extract_frames_persecond):
                        grid_image[0:height, j * width : (j + 1) * width] = frames[j]

                    _, buffer = cv2.imencode(".jpg", grid_image)
                    base64Frames[model_name].append(base64.b64encode(buffer).decode("utf-8"))
                    frames = []

            video.release()

        with ThreadPoolExecutor() as executor:
            futures = []
            for model_name, video_path in videos_dict.items():
                futures.append(executor.submit(process_video, model_name, video_path))

            for future in futures:
                future.result()

        return base64Frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get item from dataset

        Returns:
            dict: Contains frames, grid_frames and prompt for the video
        """
        annotation = self.annotations[idx]

        # Process videos using the videos dictionary
        frames = self.process_video(annotation["videos"], 2)
        grid_frames = self.process_video2gridview(annotation["videos"], 8)

        return {"frames": frames, "grid_frames": grid_frames, "prompt": annotation["prompt_en"]}


def get_prompt_from_filename(path: str):
    """
    1. prompt-0.suffix -> prompt
    2. prompt.suffix -> prompt
    """
    prompt = Path(path).stem
    number_ending = r"-\d+$"  # checks ending with -<number>
    if re.search(number_ending, prompt):
        return re.sub(number_ending, "", prompt)
    return prompt


def save_json(data, path, indent=4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path):
    """
    Load a JSON file from the given file path.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - data (dict or list): The data loaded from the JSON file, which could be a dictionary or a list.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)
