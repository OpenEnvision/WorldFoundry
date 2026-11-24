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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GenerationConfig, PreTrainedModel, WhisperFeatureExtractor

from .base_projector import MultimodalProjector, MultimodalProjectorConfig
from .builder import build_llm_and_tokenizer
from .configuration_vila import VILAConfig
from .constants import IGNORE_INDEX
from .media_encoder import BasicImageEncoder, BasicSoundEncoder, TSPVideoEncoder
from .qwen_audio_encoder import Qwen2AudioTower
from .siglip_encoder import SiglipVisionTowerDynamicS2
from .sound_base_projector import SoundMultimodalProjector, SoundMultimodalProjectorConfig


class VILAForCausalLM(PreTrainedModel):
    config_class = VILAConfig
    main_input_name = "input_embeds"
    _no_split_modules: ClassVar[list[str]] = ["Qwen2DecoderLayer", "SiglipEncoderLayer"]

    def __init__(self, config: VILAConfig, device_map: str | dict = "auto") -> None:
        super().__init__(config)
        checkpoint = Path(config._name_or_path)
        projector_config = MultimodalProjectorConfig.from_pretrained(checkpoint / "mm_projector", local_files_only=True)
        self.mm_projector = MultimodalProjector.from_pretrained(
            checkpoint / "mm_projector", config, config=projector_config, local_files_only=True
        )
        self.vision_tower = SiglipVisionTowerDynamicS2(str(checkpoint / "vision_tower"), config)
        self.sound_tower = Qwen2AudioTower(str(checkpoint / "sound_tower"), config)
        sound_config = SoundMultimodalProjectorConfig.from_pretrained(
            checkpoint / "sound_mm_projector", local_files_only=True
        )
        self.sound_mm_projector = SoundMultimodalProjector.from_pretrained(
            checkpoint / "sound_mm_projector",
            config,
            config=sound_config,
            torch_dtype=config.model_dtype,
            local_files_only=True,
        )
        if device_map in ["auto", "cuda"]:
            self.mm_projector = self.mm_projector.cuda()
            self.vision_tower = self.vision_tower.cuda()
            self.sound_tower = self.sound_tower.cuda()
            self.sound_mm_projector = self.sound_mm_projector.cuda()
        self.llm, self.tokenizer = build_llm_and_tokenizer(checkpoint, config, device_map)
        self.llm_model_embed_tokens = self.llm.model.embed_tokens
        video_config = {key: value for key, value in config.video_encoder.items() if key != "_target_"}
        sound_config = {key: value for key, value in config.sound_encoder.items() if key != "_target_"}
        self.encoders = {
            "image": BasicImageEncoder(self),
            "video": TSPVideoEncoder(self, **video_config),
            "sound": BasicSoundEncoder(self, **sound_config),
        }
        self.generation_config = GenerationConfig.from_model_config(config)
        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str | dict = "auto",
        local_files_only: bool = True,
    ) -> VILAForCausalLM:
        """Build explicit in-tree classes and load the released component key layout."""
        config = VILAConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=local_files_only)
        config._name_or_path = str(pretrained_model_name_or_path)
        config.model_dtype = torch_dtype
        return cls._from_config(config, torch_dtype=torch_dtype, device_map=device_map)

    def encode_video(self, videos: list[torch.Tensor]) -> torch.Tensor:
        images = torch.cat(videos, dim=0)
        image_features = self.vision_tower(images)
        image_features, block_sizes = self.merge_features_for_dynamic_s2(image_features, [None] * len(images))
        image_features = [
            self.split_chessboard(features, block_size[0], block_size[1])
            for features, block_size in zip(image_features, block_sizes, strict=False)
        ]
        image_features = torch.cat([rearrange(features, "b c h w -> b (h w) c") for features in image_features], dim=0)
        image_features = self.mm_projector(image_features.to(self.device, self.dtype))
        image_features = list(image_features.split([1] * len(image_features), dim=0))
        image_features = [self.merge_chessboard(features, 1, 1) for features in image_features]
        image_features = [rearrange(features, "1 c h w -> (h w) c") for features in image_features]
        return torch.stack(image_features, dim=0)

    def encode_sound(self, sounds: list[torch.Tensor]) -> list[torch.Tensor]:
        features, lengths = self.sound_tower(sounds)
        features = self.sound_mm_projector(features)
        result = []
        start = 0
        for length in lengths:
            result.append(features[start : start + length])
            start += length
        return result

    def merge_features_for_dynamic_s2(
        self, image_features: torch.Tensor, block_sizes: list[tuple[int, int] | None]
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        scales = self.vision_tower.scales
        resize_output_to_scale_idx = self.vision_tower.resize_output_to_scale_idx

        image_features_each_image = []
        new_block_sizes = []
        block_cnt = 0
        for block_size_each_image in block_sizes:
            if block_size_each_image is None:
                cur_features = image_features[block_cnt : block_cnt + 1]
                cur_features = rearrange(cur_features, "1 (h w) c -> 1 c h w", h=int(cur_features.shape[1] ** 0.5))
                cur_features = cur_features.repeat(1, len(scales), 1, 1)
                image_features_each_image.append(cur_features)
                new_block_sizes.append((1, 1))
                block_cnt += 1
            else:
                cur_features_each_scale = []
                for scale in scales[:-1]:
                    num_blocks_this_scale = (scale // scales[0]) ** 2
                    cur_features_each_scale.append(
                        self.merge_chessboard(
                            image_features[block_cnt : block_cnt + num_blocks_this_scale],
                            num_split_h=scale // scales[0],
                            num_split_w=scale // scales[0],
                        )
                    )  # 1 * C * H * W
                    block_cnt += num_blocks_this_scale
                num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]
                cur_features_each_scale.append(
                    self.merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_last_scale],
                        num_split_h=block_size_each_image[0],
                        num_split_w=block_size_each_image[1],
                    )
                )  # 1 * C * H * W
                block_cnt += num_blocks_last_scale

                # resize and concat features from different scales
                output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[-2:]
                cur_features = torch.cat(
                    [
                        F.interpolate(cur_features_each_scale[i].to(torch.float32), size=output_size, mode="area").to(
                            cur_features_each_scale[i].dtype
                        )
                        for i in range(len(cur_features_each_scale))
                    ],
                    dim=1,
                )

                image_features_each_image.append(cur_features)

                if resize_output_to_scale_idx == len(scales) - 1 or resize_output_to_scale_idx == -1:
                    new_block_sizes.append(block_size_each_image)
                else:
                    new_block_sizes.append(
                        (
                            scales[resize_output_to_scale_idx] // scales[0],
                            scales[resize_output_to_scale_idx] // scales[0],
                        )
                    )

        assert block_cnt == len(image_features)

        return image_features_each_image, new_block_sizes

    @staticmethod
    def split_chessboard(x: torch.Tensor, num_split_h: int, num_split_w: int) -> torch.Tensor:
        """Stack row-major spatial tiles on the batch axis."""
        B, C, H, W = x.shape
        assert H % num_split_h == 0 and W % num_split_w == 0
        h, w = H // num_split_h, W // num_split_w
        x_split = torch.cat(
            [x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w] for i in range(num_split_h) for j in range(num_split_w)],
            dim=0,
        )
        return x_split

    @staticmethod
    def merge_chessboard(x: torch.Tensor, num_split_h: int, num_split_w: int) -> torch.Tensor:
        """Merge row-major patch blocks into NCHW feature maps."""
        B = x.shape[0]
        if x.dim() == 3:
            N = x.shape[1]
            x = rearrange(x, "b (h w) c -> b c h w", h=int(N**0.5), w=int(N**0.5))

        assert B % (num_split_h * num_split_w) == 0
        b = B // (num_split_h * num_split_w)

        x_merge = torch.cat(
            [
                torch.cat(
                    [x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b] for j in range(num_split_w)], dim=-1
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

        return x_merge

    def encode_images(
        self, images: torch.Tensor, block_sizes: list[tuple[int, int] | None] | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        if block_sizes is None:
            block_sizes = [None] * len(images)

        if getattr(self.config, "dynamic_s2", False):
            image_features = self.vision_tower(images)

            image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

            image_features = [
                self.split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes, strict=False)
            ]  # list of B * C * H * W tensors
            image_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0
            )  # B * N * C

            image_features = self.mm_projector(image_features)
            image_features = list(
                image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
            )
            image_features = [
                self.merge_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes, strict=False)
            ]  # list of 1 * C * H * W tensors
            image_features = [rearrange(x, "1 c h w -> (h w) c") for x in image_features]  # list of N * C tensors
            if all(feature.shape[0] == image_features[0].shape[0] for feature in image_features):
                image_features = torch.stack(image_features, dim=0)
        else:
            image_features = self.vision_tower(images)

            image_features = self.mm_projector(image_features)
        return image_features

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: dict[str, list[torch.Tensor]],
        media_config: dict[str, dict[str, Any]],
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        media = copy.deepcopy(media) if media is not None else defaultdict(list)
        media_config = copy.deepcopy(media_config) if media_config is not None else defaultdict(dict)

        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        # Extract text and media embeddings
        text_embeds = self.llm_model_embed_tokens(input_ids)

        mm_info = {}
        if "video_info" in media:
            video_info = media["video_info"]
            del media["video_info"]
            mm_info["video_info"] = video_info
        else:
            video_info = None

        if "audio_info" in media:
            audio_info = media["audio_info"]
            del media["audio_info"]
            mm_info["audio_info"] = audio_info
        else:
            audio_info = None

        media_embeds = self.__embed_media_tokens(media, media_config, mm_info)

        # Based on segment_aud_indices_list and segment_vis_indices_list, get interleaved vis-aud embeddings for video
        video_sound_embeds_idx = 0
        text_embeds = text_embeds.to(self.dtype)

        if video_info is not None and self.config.load_audio_in_video and self.config.interleaved_vis_aud_in_video:
            sep_embed = self.encoders["video"].embed_tokens("\n").to(text_embeds.dtype)
            assert self.encoders["video"].end_tokens is None, "end_tokens must be None for interleaved vis-aud in video"
            new_video_embeds = deque()
            video_embeds_idx = 0
            for k in range(len(video_info)):
                if video_info[k] is None:
                    continue
                for i in range(len(video_info[k])):
                    has_audio = video_info[k][i]["has_audio"]
                    if not has_audio:
                        new_video_embeds.append(media_embeds["video"][video_embeds_idx])
                        video_embeds_idx += 1
                        continue

                    # Check bounds for sound embeddings
                    if video_sound_embeds_idx >= len(media_embeds["sound"]):
                        raise ValueError(
                            f"Sound embeddings index {video_sound_embeds_idx} out of bounds for video_info[{k}][{i}]"
                        )

                    segment_aud_indices_list = video_info[k][i]["segment_aud_indices_list"]
                    segment_vis_indices_list = video_info[k][i]["segment_vis_indices_list"]

                    vis_fea_len_per_frame = (
                        media_embeds["video"][video_embeds_idx].shape[0] / video_info[k][i]["expected_frame_count"]
                    )
                    aud_fea_len_per_stft_frame = (
                        media_embeds["sound"][video_sound_embeds_idx].shape[0]
                        / audio_info[k][i]["new_audio_n_stft_frames"]
                    )
                    vis_end = 0
                    aud_end = 0
                    _new_video_embed = []
                    for j in range(len(segment_vis_indices_list)):
                        _vis_aud_fea = []
                        if len(segment_vis_indices_list[j]) > 0:
                            _new_frames = [
                                int(np.ceil((_frame + 1) * vis_fea_len_per_frame))
                                for _frame in segment_vis_indices_list[j]
                            ]
                            _vis_fea_end = _new_frames[-1]
                            # Ensure we don't exceed the available features
                            _vis_fea_end = min(_vis_fea_end, media_embeds["video"][video_embeds_idx].shape[0])
                            if (
                                j == len(segment_vis_indices_list) - 1
                                and i == len(video_info) - 1
                                and k == len(video_info[i]) - 1
                                and _vis_fea_end != media_embeds["video"][video_embeds_idx].shape[0]
                            ):
                                print(
                                    f"Warning: The number of last interleaved video features does not match the video feature length. Expected: {media_embeds['video'][video_embeds_idx].shape[0]}, Got: {_vis_fea_end}"
                                )
                                _vis_fea_end = media_embeds["video"][video_embeds_idx].shape[0]
                            _vis_fea = media_embeds["video"][video_embeds_idx][vis_end:_vis_fea_end]
                            vis_end = _vis_fea_end
                            _vis_aud_fea.append(_vis_fea)
                        _vis_aud_fea.append(sep_embed)
                        if len(segment_aud_indices_list[j]) > 0:
                            _new_audio_indices = [
                                int(np.ceil(_fea * aud_fea_len_per_stft_frame)) for _fea in segment_aud_indices_list[j]
                            ]
                            _aud_fea_end = _new_audio_indices[-1]
                            # Ensure we don't exceed the available features
                            _aud_fea_end = min(_aud_fea_end, media_embeds["sound"][video_sound_embeds_idx].shape[0])
                            _aud_fea = media_embeds["sound"][video_sound_embeds_idx][aud_end:_aud_fea_end]
                            _vis_aud_fea.append(_aud_fea)
                            aud_end = _aud_fea_end
                        _vis_aud_fea.append(sep_embed)
                        _new_video_embed.append(torch.cat(_vis_aud_fea, dim=0))
                    video_sound_embeds_idx += 1
                    new_video_embeds.append(torch.cat(_new_video_embed, dim=0))
                    video_embeds_idx += 1

            assert len(new_video_embeds) == len(media_embeds["video"]), (
                "The number of new video embeddings does not match the number of original video embeddings."
            )
            media_embeds["video"] = new_video_embeds
        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]
        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        sound_embeds_idx = 0
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    name = media_tokens[input_ids[k][pos].item()]
                    if input_ids[k][pos].item() == self.tokenizer.media_token_ids["sound"]:
                        if self.config.interleaved_vis_aud_in_video and sound_embeds_idx < video_sound_embeds_idx:
                            media_embeds[name].popleft()
                            sound_embeds_idx += 1
                            pos += 1
                            continue
                        sound_embeds_idx += 1

                    end = pos + 1
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]

                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed

        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed! Still {len(media_embeds[name])} left.")

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)

    def __batchify_sequence(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                labels[k] = labels[k].to(device)
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id
        return generation_config

    def __embed_media_tokens(
        self,
        media: dict[str, list[torch.Tensor]],
        media_config: dict[str, dict[str, Any]],
        mm_info: dict[str, Any],
    ) -> dict[str, list[torch.Tensor]]:
        embeds = defaultdict(deque)

        if self.config.unified_audio_encoder:
            assert len(media["speech"]) == 0

        for name in media:
            if name == "speech":
                continue
            _encoder = self.encoders[name]

            if name == "sound":
                import whisper

                all_audio_chunk_lengths = []
                for _sample_idx in range(len(media[name])):
                    for _mm_idx in range(len(mm_info["audio_info"][_sample_idx])):
                        _new_audio_chunk_length = mm_info["audio_info"][_sample_idx][_mm_idx]["new_audio_chunk_length"]
                        all_audio_chunk_lengths.append(_new_audio_chunk_length)
                cur_batch_max_audio_duration = max(all_audio_chunk_lengths)
                cur_batch_max_audio_samples = cur_batch_max_audio_duration * self.config.audio_sampling_rate

                whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    self.config._name_or_path,
                    chunk_length=cur_batch_max_audio_duration,
                    sampling_rate=self.config.audio_sampling_rate,
                    hop_length=self.config.audio_hop_length,
                    local_files_only=True,
                )

                new_media = []
                assert len(all_audio_chunk_lengths) == len(media[name]), (
                    "The number of audio chunk lengths does not match the number of audio samples."
                )

                _media = media.get(name, [])
                aud_idx = 0
                for _batch_idx in range(len(mm_info["audio_info"])):
                    _audio_info = mm_info["audio_info"][_batch_idx]
                    if _audio_info is not None:
                        for _mm_idx in range(len(_audio_info)):
                            _audio = _media[aud_idx]
                            if type(_audio) is torch.Tensor:
                                device = _audio.device
                                dtype = _audio.dtype
                                _audio = _audio.cpu().float()
                            else:
                                device = self.device
                                dtype = self.dtype
                            _audio = whisper.pad_or_trim(_audio, length=cur_batch_max_audio_samples)
                            aud_idx += 1
                            stft_features = whisper_feature_extractor(
                                _audio,
                                sampling_rate=self.config.audio_sampling_rate,
                                return_attention_mask=True,
                                padding="max_length",
                                return_tensors="pt",
                            ).to(device, dtype)

                            new_media.append(stft_features)
                            _audio_info[_mm_idx]["new_audio_chunk_length"] = cur_batch_max_audio_duration
                            _audio_info[_mm_idx]["new_audio_n_samples"] = cur_batch_max_audio_samples
                            _audio_info[_mm_idx]["audio_end_sample_sec"] = (
                                _audio_info[_mm_idx]["audio_start_sec"] + cur_batch_max_audio_duration
                            )
                            _audio_info[_mm_idx]["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]
                media[name] = new_media

            if len(media[name]) > 0:
                embeds[name] = deque(_encoder(media[name], media_config[name], mm_info))
        return embeds
