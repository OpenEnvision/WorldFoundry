# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
from transformers import Qwen2TokenizerFast

from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.constants import MEDIA_TOKENS, MM_BOS_EOS_TOKENS


def tokenize_conversation(
    messages: list[dict[str, str]],
    tokenizer: Qwen2TokenizerFast,
    mm_use_bos_eos_tokens: bool,
    unified_audio_encoder: bool,
    add_generation_prompt: bool = False,
) -> torch.Tensor:
    """Apply the released Qwen chat template and modality BOS/EOS tokens."""
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    conversation = [{"role": roles[message["from"]], "content": message["value"].strip()} for message in messages]
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=add_generation_prompt, tokenize=False)
    if mm_use_bos_eos_tokens:
        for name, (bos, eos) in MM_BOS_EOS_TOKENS.items():
            if unified_audio_encoder and name == "speech":
                bos, eos = MM_BOS_EOS_TOKENS["sound"]
            media_token = MEDIA_TOKENS[name]
            if media_token in text:
                parts = text.split(media_token)
                parts[0] = parts[0] + bos
                parts[-1] = eos + parts[-1]
                text = media_token.join(parts)
    return tokenizer(text, return_tensors="pt").input_ids[0]
