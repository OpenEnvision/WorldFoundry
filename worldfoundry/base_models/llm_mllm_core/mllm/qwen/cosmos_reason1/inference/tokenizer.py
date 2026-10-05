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

from typing import Optional

try:
    from qwen_vl_utils import extract_vision_info, process_vision_info
except ImportError as exc:
    _QWEN_VL_UTILS_IMPORT_ERROR = exc

    def _missing_qwen_vl_utils(*args, **kwargs):
        raise ImportError(
            "qwen_vl_utils is required for Qwen-VL image or video prompts. "
            "Install the model runtime dependencies before using visual inputs."
        ) from _QWEN_VL_UTILS_IMPORT_ERROR

    extract_vision_info = _missing_qwen_vl_utils
    process_vision_info = _missing_qwen_vl_utils

from transformers.models.auto.processing_auto import AutoProcessor


def build_tokenizer(
    tokenizer_type: str,
    cache_dir: Optional[str] = None,
):
    return Processor(tokenizer_type, cache_dir)


def flatten_content_list(messages):
    new_messages = []
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            text_list = [item["text"] for item in message["content"]]
            message["content"] = " ".join(text_list)
        new_messages.append(message)
    return new_messages


class Processor:
    # This is a wrapper around the AutoProcessor class to add some helper functions
    def __init__(self, name="Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=None):
        self.name = name

        if name not in [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen2.5-0.5B",
        ]:
            raise ValueError(f"Error loading processor {name}, please check if the tokenizer is available")
        if "VL" not in name:
            self.is_vision_tokenizer = False
        else:
            self.is_vision_tokenizer = True

        self.processor = AutoProcessor.from_pretrained(cache_dir or name)

        if hasattr(self.processor, "image_token"):
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        else:
            self.image_token_id = None
        if hasattr(self.processor, "video_token"):
            self.video_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        else:
            self.video_token_id = None

        if hasattr(self.processor, "tokenizer"):
            self.eos_id = self.processor.tokenizer.eos_token_id
            self.pad_id = self.processor.tokenizer.pad_token_id
        else:
            self.eos_id = self.processor.eos_token_id
            self.pad_id = self.processor.pad_token_id

    def apply_chat_template(
        self, messages, add_generation_prompt=False, return_tensors="pt", tokenize=True, add_vision_id=False
    ):
        assert tokenize, "tokenize must be True"
        if self.name.startswith("Qwen/Qwen2"):
            # Use manual workaround for add_vision_id bug
            if not self.is_vision_tokenizer:
                messages = flatten_content_list(messages)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                add_vision_id=add_vision_id,
            )
            if self.is_vision_tokenizer:
                image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
                vision_infos = extract_vision_info(messages)
            else:
                image_inputs, video_inputs, vision_infos = None, None, []

            # add fps ourselves, as videos have been presampled according to the specified token length
            fps_list = []
            for vision_info in vision_infos:
                if "video" in vision_info:
                    fps_list.append(vision_info["fps"])

            # process inputs
            if self.is_vision_tokenizer:
                processor_kwargs = dict(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=False,
                    return_tensors=return_tensors,
                )
                # Recent Transformers releases reject an empty fps list for
                # text-only prompts. Preserve the upstream list behavior when
                # videos are present and omit the optional argument otherwise.
                if fps_list:
                    processor_kwargs["fps"] = fps_list
                inputs = self.processor(**processor_kwargs)
            else:
                inputs = self.processor(
                    text=[text],
                    padding=False,
                    return_tensors=return_tensors,
                )

            # save raw text
            inputs["text"] = text

            # Convert the processor's batch-of-one output into one feature record.
            inputs["input_ids"] = inputs["input_ids"][0]  # N_dialogue, N_token -> N_token
            inputs["attention_mask"] = inputs["attention_mask"][0]  # N_dialogue, N_token -> N_token
            # inputs["image_grid_thw"]: N_img, 3
            # inputs["video_grid_thw"]: N_video, 3
        else:
            raise ValueError(f"apply_chat_template is not implemented for tokenizer_type {self.name}")

        return inputs

    def encode(self, *args, **kwargs):
        return self.processor.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.decode(*args, **kwargs)
