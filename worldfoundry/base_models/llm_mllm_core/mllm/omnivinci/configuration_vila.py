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

from typing import Any, ClassVar

from transformers import PretrainedConfig


class VILAConfig(PretrainedConfig):
    model_type = "vila"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        llm_cfg: dict[str, Any] | None = None,
        vision_tower_cfg: dict[str, Any] | None = None,
        mm_projector_cfg: dict[str, Any] | None = None,
        speech_tower_cfg: dict[str, Any] | str | None = None,
        sound_tower_cfg: dict[str, Any] | str | None = None,
        speech_mm_projector_cfg: dict[str, Any] | str | None = None,
        sound_mm_projector_cfg: dict[str, Any] | str | None = None,
        architectures: list[str] | None = None,
        resume_path: str | None = None,
        hidden_size: int | None = None,
        mm_hidden_size: int | None = None,
        image_aspect_ratio: str | None = None,
        num_video_frames: int | None = None,
        fps: float | None = None,
        mm_vision_select_layer: int | None = None,
        mm_vision_select_feature: str | None = None,
        mm_use_im_start_end: bool = False,
        mm_use_im_patch_token: bool = False,
        mm_projector_lr: float | None = None,
        vision_tower_lr: float | None = None,
        vision_resolution: int | None = None,
        interpolate_mode: str | None = None,
        s2: bool | None = None,
        dynamic_s2: bool | None = None,
        s2_scales: str | list[int] | None = None,
        s2_max_split_size: int | None = None,
        s2_resize_output_to_scale_idx: int = 0,
        min_tiles: int | None = 1,
        max_tiles: int | None = 12,
        num_time_tokens: int | None = None,
        time_token_format: str | None = None,
        image_encoder: str | dict[str, Any] = '{"_target_": "BasicImageEncoder"}',
        video_encoder: str | dict[str, Any] = '{"_target_": "TSPVideoEncoder"}',
        sound_encoder: str | dict[str, Any] = '{"_target_": "BasicSoundEncoder"}',
        speech_encoder: str | dict[str, Any] = '{"_target_": "BasicSpeechEncoder"}',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.speech_tower_cfg = speech_tower_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.speech_mm_projector_cfg = speech_mm_projector_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.resume_path = resume_path

        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.fps = fps
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.vision_tower_lr = vision_tower_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.dynamic_s2 = dynamic_s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size
        self.s2_resize_output_to_scale_idx = s2_resize_output_to_scale_idx
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.num_time_tokens = num_time_tokens
        self.time_token_format = time_token_format

        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.sound_encoder = sound_encoder
        self.speech_encoder = speech_encoder
        self.audio_sampling_rate = 16000
        self.audio_chunk_length = 120
        self.interleaved_vis_aud_in_video = True
        self.interleaved_video_segment_duration = 30
        self.audio_hop_length = 60

        super().__init__(**kwargs)
