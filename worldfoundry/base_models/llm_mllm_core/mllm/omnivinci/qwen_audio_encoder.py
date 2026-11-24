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


import torch
from torch import nn
from transformers import PretrainedConfig, Qwen2AudioEncoder


class Qwen2AudioTower(nn.Module):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__()
        self.audio_tower = Qwen2AudioEncoder.from_pretrained(
            model_name_or_path, attn_implementation="flash_attention_2", local_files_only=True
        )
        self.audio_chunk_unit_length = 3000

    def forward(self, sounds: list[torch.Tensor | dict[str, torch.Tensor]]) -> tuple[torch.Tensor, list[int]]:
        if type(sounds) is list:
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features") or (type(sound) is dict and "input_features" in sound):
                    sound = sound["input_features"]

                sound_feature = self.forward_audio_tower_batch(sound)
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            if len(sound_features) > 0:
                sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")

        return sound_features, audio_output_lengths

    def forward_audio_tower_batch(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Process long audio input by splitting into fixed-size chunks (30 seconds),
        padding if needed, batching them together, and processing through the audio tower.

        Args:
            inp: Tensor of shape (batch_size, n_mels, seq_len)

        Returns:
            Tensor of shape (batch_size, num_chunks * chunk_seq_len, hidden_size)
        """
        batch_size, n_mels, seq_len = inp.shape
        chunk_length = self.audio_chunk_unit_length
        num_chunks = (seq_len + chunk_length - 1) // chunk_length  # Ceiling division

        padded_chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)

            # Extract and pad chunk if necessary
            chunk = inp[:, :, start_idx:end_idx]
            if chunk.shape[2] < chunk_length:
                pad_len = chunk_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len), mode="constant", value=0)

            padded_chunks.append(chunk)

        # Stack chunks along batch dimension
        all_chunks = torch.cat(padded_chunks, dim=0).reshape(batch_size * num_chunks, n_mels, chunk_length)

        # Forward pass through the audio tower
        chunk_outputs = self.audio_tower(all_chunks)
        hidden_states = chunk_outputs.last_hidden_state

        # Reshape back to (batch_size, num_chunks * seq_len', hidden_size)
        _, chunk_seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, num_chunks * chunk_seq_len, hidden_size)

        return hidden_states
