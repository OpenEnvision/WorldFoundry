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

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen2TokenizerFast, SiglipImageProcessor
from transformers.feature_extraction_utils import BatchFeature

from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.configuration_vila import VILAConfig
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.constants import DEFAULT_IMAGE_TOKEN
from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.mm_utils import process_image, process_images

from .media import Image, Video, extract_media
from .tokenizer_utils import tokenize_conversation


class VILAProcessor:
    """OmniVinci's local image/video/text processor and temporal media metadata."""

    def __init__(
        self, image_processor: SiglipImageProcessor, tokenizer: Qwen2TokenizerFast, config: VILAConfig
    ) -> None:
        self.config = config
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer("<|endoftext|>").input_ids[0]

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, *, local_files_only: bool = True) -> VILAProcessor:
        checkpoint = Path(checkpoint_path)
        image_processor = SiglipImageProcessor.from_pretrained(
            checkpoint / "vision_tower", local_files_only=local_files_only
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(checkpoint / "llm", local_files_only=local_files_only)
        config = VILAConfig.from_pretrained(checkpoint, local_files_only=local_files_only)
        return cls(image_processor, tokenizer, config)

    def apply_chat_template(
        self, conversation: list[dict], *, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> list[dict]:
        messages = []
        roles = {"user": "human", "assistant": "gpt", "system": "system"}
        for message in conversation:
            value = []
            for content in message["content"]:
                if content["type"] == "image":
                    value.append(Image(content["image"]))
                elif content["type"] == "video":
                    value.append(Video(content["video"]))
                elif content["type"] == "text":
                    value.append(content["text"])
                else:
                    raise ValueError(f"Unsupported OmniVinci content type: {content['type']}")
            messages.append({"from": roles[message["role"]], "value": value})
        return messages

    def __call__(self, conversations: list[list[dict[str, Any]]]) -> BatchFeature:
        input_ids_list = []
        media = defaultdict(list)
        media_config = defaultdict(dict)
        for conversation in conversations:
            features = self.__single_call__(conversation)
            input_ids_list.append(features.input_ids)
            for name in features.media:
                media[name] += features.media[name]
            for name in features.media_config:
                media_config[name].update(features.media_config[name])
        max_length = max(ids.shape[1] for ids in input_ids_list)
        input_ids = torch.cat(
            [
                torch.cat((torch.full_like(ids, self.pad_token_id)[:, : max_length - ids.shape[1]], ids), dim=1)
                for ids in input_ids_list
            ],
            dim=0,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask[input_ids == self.pad_token_id] = False
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )

    def __single_call__(self, conversation: list[dict[str, Any]]) -> BatchFeature:
        conversation = copy.deepcopy(conversation)
        media = extract_media(conversation, self.config)
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    self.config.image_processor = self.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(media["image"][0], self.config, None, enable_dynamic_res=True).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if isinstance(self.config.s2_scales, str):
                            self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(media["image"], self.image_processor, self.config).half()
                media[name] = list(images)
            elif name == "video":
                media[name] = [
                    process_images(images, self.image_processor, self.config).half() for images in media[name]
                ]
            elif name == "sound":
                media[name] = list(media[name])
            elif name in ["video_info", "audio_info"]:
                media[name] = [media[name]]
            else:
                raise ValueError(f"Unsupported OmniVinci media type: {name}")
        inputs = tokenize_conversation(
            conversation,
            self.tokenizer,
            mm_use_bos_eos_tokens=self.config.mm_use_bos_eos_tokens,
            unified_audio_encoder=self.config.unified_audio_encoder,
            add_generation_prompt=True,
        )
        return BatchFeature(data={"input_ids": inputs.unsqueeze(0), "media": media, "media_config": media_config})
