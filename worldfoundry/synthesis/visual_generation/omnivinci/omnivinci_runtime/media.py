# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import PIL.Image

from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.configuration_vila import VILAConfig
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.constants import MEDIA_TOKENS


class File:
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    """An image in the processor conversation."""


class Video(File):
    """A video in the processor conversation."""


def get_overlap(first: list[float], second: list[float]) -> tuple[float, float] | None:
    overlap_start = max(first[0], second[0])
    overlap_end = min(first[1], second[1])
    return (overlap_start, overlap_end) if overlap_start < overlap_end else None


def _load_speech(speech_path: str, config: VILAConfig) -> tuple[np.ndarray, dict[str, Any]]:
    import whisper

    if ".mp4" in speech_path:
        from decord import AudioReader, cpu

        speech = AudioReader(speech_path, ctx=cpu(0), sample_rate=config.audio_sampling_rate, mono=True)
        original_samples = speech.shape[1]
        end_sample = min(config.audio_chunk_length * config.audio_sampling_rate, original_samples)
        speech = speech[:end_sample].asnumpy()[0]
    else:
        import librosa

        speech, _ = librosa.load(speech_path, sr=config.audio_sampling_rate)
        original_samples = speech.shape[0]
        end_sample = min(config.audio_chunk_length * config.audio_sampling_rate, original_samples)
        speech = speech[:end_sample]
    speech = speech.astype(np.float32)
    audio_n_samples = int(
        np.ceil(speech.shape[0] / (config.audio_sampling_rate * 30)) * (config.audio_sampling_rate * 30)
    )
    speech = whisper.pad_or_trim(speech, length=audio_n_samples)
    return speech, {
        "new_audio_chunk_length": int(audio_n_samples // config.audio_sampling_rate),
        "new_audio_n_samples": audio_n_samples,
        "ori_audio_duration": original_samples / config.audio_sampling_rate,
        "audio_start_sec": 0.0,
        "audio_end_sample_sec": end_sample / config.audio_sampling_rate,
    }


def _load_video(video_path: str, config: VILAConfig) -> tuple[list[PIL.Image.Image], np.ndarray | None, dict[str, Any]]:
    """Extract the released uniform frame indices and aligned audio segments."""
    vidcap = cv2.VideoCapture(video_path)
    try:
        audio_info = None
        audio_features = None
        if config.load_audio_in_video:
            try:
                audio_features, audio_info = _load_speech(video_path, config)
            except Exception:
                audio_features = None
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        while frame_count > 0:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if vidcap.grab():
                break
            frame_count -= 1
        else:
            raise ValueError(f"Video '{video_path}' has no frames.")
        indices = np.round(np.linspace(0, frame_count - 1, config.num_video_frames)).astype(int)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_duration = frame_count / fps
        interleaved = config.load_audio_in_video and config.interleaved_vis_aud_in_video and audio_features is not None
        if interleaved:
            segment_duration = config.interleaved_video_segment_duration
            segment_vis_indices_list = []
            segment_aud_indices_list = []
            segment_counts = np.ceil(video_duration / segment_duration).astype(int)
            stft_frames_per_second = config.audio_sampling_rate // config.audio_hop_length
            frame_index = 0
            audio_start_index = 0
            for segment in range(segment_counts):
                end_frame = min((segment + 1) * segment_duration * fps, frame_count)
                segment_indices = []
                while frame_index < len(indices) and indices[frame_index] < end_frame:
                    segment_indices.append(indices[frame_index])
                    frame_index += 1
                segment_vis_indices_list.append(segment_indices)
                clip_start = segment * segment_duration
                overlap = get_overlap(
                    [clip_start, min(clip_start + segment_duration, video_duration)],
                    [audio_info["audio_start_sec"], audio_info["audio_end_sample_sec"]],
                )
                if overlap is not None:
                    audio_end_index = round((overlap[1] - audio_info["audio_start_sec"]) * stft_frames_per_second)
                    segment_aud_indices_list.append([audio_start_index, audio_end_index])
                    audio_start_index = audio_end_index
                else:
                    segment_aud_indices_list.append([])
        frames = {}
        frame_times = {}
        for index in indices:
            if index in frames:
                continue
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = vidcap.read()
            if not success:
                continue
            frames[index] = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_times[index] = index / fps
        output_frames = [frames[index] for index in indices if index in frames]
        video_info = {}
        if interleaved:
            processed_frame_index = 0
            segment_positions = []
            for segment_indices in segment_vis_indices_list:
                segment_positions.append([])
                for index in segment_indices:
                    if index in frames:
                        segment_positions[-1].append(processed_frame_index)
                        processed_frame_index += 1
            video_info["segment_vis_indices_list"] = segment_positions
            video_info["segment_aud_indices_list"] = segment_aud_indices_list
            video_info["expected_frame_count"] = len(indices)
        if audio_info is not None:
            audio_info["video_path"] = video_path
        video_info.update(
            {
                "video_path": video_path,
                "has_audio": audio_features is not None,
                "video_duration": video_duration,
                "audio_info": audio_info,
                "video_frame_times": [frame_times[index] for index in indices if index in frame_times],
            }
        )
        return output_frames, audio_features, video_info
    finally:
        vidcap.release()


def extract_media(messages: list[dict[str, Any]], config: VILAConfig) -> dict[str, list[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        parts = message["value"] if isinstance(message["value"], list) else [message["value"]]
        for part in parts:
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, Image):
                media["image"].append(PIL.Image.open(part.path))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                frames, audio_features, video_info = _load_video(part.path, config)
                media["video"].append(frames)
                media["video_info"].append(video_info)
                if audio_features is not None:
                    media["sound"].append(audio_features)
                    media["audio_info"].append(video_info["audio_info"])
                    text += MEDIA_TOKENS["sound"]
                text += MEDIA_TOKENS["video"]
            else:
                raise ValueError(f"Unsupported OmniVinci prompt part: {type(part)}")
        message["value"] = text
    return media
