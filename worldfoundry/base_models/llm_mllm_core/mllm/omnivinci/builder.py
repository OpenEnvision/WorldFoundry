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

from pathlib import Path

from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2TokenizerFast

from .configuration_vila import VILAConfig
from .constants import MEDIA_TOKENS


def build_llm_and_tokenizer(
    checkpoint_path: Path, config: VILAConfig, device_map: str | dict
) -> tuple[Qwen2ForCausalLM, Qwen2TokenizerFast]:
    """Load the released decoder and media token IDs from local checkpoint data."""
    llm_path = checkpoint_path / "llm"
    llm_config = Qwen2Config.from_pretrained(llm_path, local_files_only=True)
    llm_config._attn_implementation = None
    llm_config.model_max_length = None
    llm = Qwen2ForCausalLM.from_pretrained(
        llm_path,
        config=llm_config,
        torch_dtype=config.model_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        llm_path, padding_side="right", use_fast=True, legacy=False, local_files_only=True
    )
    if config.chat_template is not None:
        template = (checkpoint_path / f"{config.chat_template}.jinja").read_text()
        tokenizer.chat_template = template.replace("    ", "").replace("\n", "")
    tokenizer.media_tokens = dict(MEDIA_TOKENS)
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)
    tokenizer.padding_side = "left"
    config.hidden_size = llm.config.hidden_size
    return llm, tokenizer
