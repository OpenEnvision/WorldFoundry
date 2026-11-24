"""Inference-only multimodal backbone used by MolmoBot.

The released MolmoBot action policies need the per-layer VLM hidden states, not
text generation.  Keeping that distinction here avoids materializing a
``[batch, sequence, vocabulary]`` logits tensor on every action request.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import field
from typing import ClassVar

import torch

from .. import tokenizer
from ..config import D
from ..model_config import BaseModelConfig
from ..preprocessing.data_formatter import DataFormatter
from ..preprocessing.multimodal_collator import MMCollator
from ..preprocessing.multimodal_preprocessor import ExamplePreprocessor
from ..preprocessing.video_preprocessor import MultiModalVideoPreprocessorConfig
from ..tokenizer import get_special_token_ids
from ..torch_utils import BufferCache
from .llm import Llm, LlmConfig
from .model import ModelBase, OLMoOutput
from .vision_backbone import MolmoVisionBackbone, MolmoVisionBackboneConfig


@dataclasses.dataclass
class VideoOlmoConfig(BaseModelConfig):
    """Checkpoint-compatible configuration for the MolmoBot VLM backbone."""

    _model_name: ClassVar[str] = "video_olmo"

    data_formatter: DataFormatter = field(default_factory=DataFormatter)
    llm: LlmConfig = field(default_factory=LlmConfig)
    vision_backbone: MolmoVisionBackboneConfig | None = field(default_factory=MolmoVisionBackboneConfig)
    mm_preprocessor: MultiModalVideoPreprocessorConfig = field(default_factory=MultiModalVideoPreprocessorConfig)
    bi_directional_attn: str | None = None
    shared_low_high_embedding: bool = True

    # These values occur in released YAML files. Context-parallel training and
    # debug-only forward variants are intentionally not part of this runtime.
    debug: str | None = None
    cp_enabled: bool = False
    apply_cp_to_vision_backbone: bool = False

    @classmethod
    def get_default_model_name(cls) -> str:
        return "video_olmo"

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "llm" not in config:
            raise ValueError(
                "Legacy monolithic Molmo checkpoints are not supported by the inference-only MolmoBot integration."
            )
        if "image_as_video" in config:
            if config.image_as_video is None:
                del config["image_as_video"]
            else:
                raise ValueError("image_as_video checkpoints are not MolmoBot action policies.")
        config.llm = LlmConfig.update_legacy_settings(config.llm)
        if config.vision_backbone is not None:
            config.vision_backbone = MolmoVisionBackboneConfig.update_legacy_settings(config.vision_backbone)
        config.data_formatter = DataFormatter.update_legacy_settings(config.data_formatter)
        config.mm_preprocessor = MultiModalVideoPreprocessorConfig.update_legacy_settings(config.mm_preprocessor)
        return config

    def build_tokenizer(self):
        return self.llm.build_tokenizer()

    def build_preprocessor(
        self,
        for_inference: bool,
        is_training: bool = False,
        text_seq_len: int | None = None,
        max_seq_len: int | None = None,
        include_image: bool = False,
        **kwargs,
    ) -> ExamplePreprocessor:
        # ``max_text_len`` was used by one downstream wrapper. Keep the alias at
        # the API edge without carrying the training preprocessor implementation.
        if text_seq_len is None:
            text_seq_len = kwargs.pop("max_text_len", None)
        if kwargs:
            raise TypeError(f"Unexpected preprocessor options: {sorted(kwargs)}")
        if is_training or not for_inference:
            raise ValueError("MolmoBot's in-tree preprocessing supports inference only.")
        if self.vision_backbone is None:
            raise ValueError("MolmoBot requires a vision backbone.")
        return ExamplePreprocessor(
            self.data_formatter,
            self.mm_preprocessor.build(
                self.build_tokenizer(),
                self.vision_backbone.build_preprocessor(),
                text_seq_len,
                max_seq_len,
            ),
            for_inference=True,
            is_training=False,
            include_image=include_image,
        )

    def build_collator(
        self,
        output_shapes,
        pad_mode: str | None,
        include_metadata: bool = True,
    ) -> MMCollator:
        if self.cp_enabled:
            raise ValueError("Released MolmoBot inference checkpoints do not use context-parallel collation.")
        return MMCollator(
            get_special_token_ids(self.build_tokenizer()),
            output_shapes,
            include_metadata=include_metadata,
            pad=pad_mode,
            cp_enabled=False,
        )

    def build_model(self, device=None) -> VideoOlmo:
        return VideoOlmo(self, device)

    @property
    def max_sequence_length(self) -> int:
        return self.llm.max_sequence_length


class VideoOlmo(ModelBase):
    """Molmo multimodal encoder specialized for action-policy inference."""

    _IMAGE_TOKEN_NAMES = (
        tokenizer.IMAGE_PATCH_TOKEN,
        tokenizer.IM_COL_TOKEN,
        tokenizer.IM_START_TOKEN,
        tokenizer.LOW_RES_IMAGE_START_TOKEN,
        tokenizer.FRAME_START_TOKEN,
        tokenizer.IM_END_TOKEN,
        tokenizer.FRAME_END_TOKEN,
        tokenizer.IMAGE_LOW_RES_TOKEN,
    )

    def __init__(self, config: VideoOlmoConfig, device=None):
        super().__init__()
        if config.cp_enabled or config.apply_cp_to_vision_backbone:
            raise ValueError(
                "Context parallelism in the released MolmoBot code is a training path and is "
                "not supported by this inference-only integration."
            )
        if config.debug is not None:
            raise ValueError(f"Debug-only MolmoBot mode is not supported: {config.debug!r}")

        self.config = config
        self.__cache = BufferCache()
        self.transformer: Llm = config.llm.build(self.__cache, device)
        self.vision_backbone: MolmoVisionBackbone | None = None
        if config.vision_backbone is not None:
            self.vision_backbone = config.vision_backbone.build(config.llm, device)

        self.special_ids = tokenizer.get_special_token_ids(config.build_tokenizer())
        self._image_token_ids = torch.tensor(
            [self.special_ids[name] for name in self._IMAGE_TOKEN_NAMES],
            dtype=torch.long,
            device="cpu",
        )
        self._low_res_image_start = self.special_ids[tokenizer.LOW_RES_IMAGE_START_TOKEN]
        self._frame_start = self.special_ids[tokenizer.FRAME_START_TOKEN]
        self._image_low_res_id = self.special_ids[tokenizer.IMAGE_LOW_RES_TOKEN]
        self._image_high_res_id = self.special_ids[tokenizer.IMAGE_PATCH_TOKEN]

        if config.bi_directional_attn == "within_image":
            if config.mm_preprocessor.image is not None:  # noqa: SIM102 - Statement form kept for readability.
                if not config.mm_preprocessor.image.use_single_crop_start_token:
                    raise ValueError("within_image attention requires use_single_crop_start_token.")
            if not config.mm_preprocessor.use_frame_special_tokens:
                raise ValueError("within_image attention requires frame special tokens.")

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _build_attention_bias(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attention_bias: torch.Tensor | None,
        response_mask: torch.Tensor | None,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != -1
        elif attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"MolmoBot attention_mask must have shape {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}."
            )
        attention_mask = attention_mask.to(dtype=torch.bool)

        causal = self.__cache.get("causal_mask")
        if causal is None or causal.device != input_ids.device or causal.shape[-1] < sequence_length:
            causal = torch.ones(
                sequence_length,
                sequence_length,
                device=input_ids.device,
                dtype=torch.bool,
            ).tril_()[None]
            self.__cache["causal_mask"] = causal
        else:
            causal = causal[:, :sequence_length, :sequence_length]

        bidirectional = None
        mode = self.config.bi_directional_attn
        image_token_ids = self._image_token_ids.to(input_ids.device)
        is_image_token = torch.any(input_ids[:, :, None] == image_token_ids[None, None, :], dim=-1)
        if mode == "image_tokens":
            bidirectional = is_image_token[:, :, None] & is_image_token[:, None, :]
        elif mode == "within_image":
            is_frame_start = (input_ids == self._frame_start) | (input_ids == self._low_res_image_start)
            frame_id = torch.cumsum(is_frame_start, dim=-1)
            same_or_earlier_frame = frame_id[:, None] <= frame_id[:, :, None]
            bidirectional = is_image_token[:, :, None] & is_image_token[:, None, :] & same_or_earlier_frame
        elif mode == "image_to_question":
            if response_mask is None:
                raise ValueError("image_to_question attention requires response_mask.")
            bidirectional = is_image_token[:, :, None] & ~response_mask.to(dtype=torch.bool)[:, None, :]
        elif mode is not None:
            raise ValueError(f"Unsupported bi_directional_attn mode: {mode!r}")

        allowed = causal if bidirectional is None else causal | bidirectional
        allowed = attention_mask[:, None, :] & allowed
        allowed = allowed[:, None, :, :]

        minimum = torch.finfo(x_dtype).min
        if attention_bias is None:
            bias = torch.zeros((), dtype=x_dtype, device=input_ids.device).expand_as(allowed)
            bias = torch.where(allowed, bias, minimum)
        else:
            bias = torch.where(allowed, attention_bias.to(dtype=x_dtype), minimum)
        if bias.shape != (batch_size, 1, sequence_length, sequence_length):
            try:
                bias = torch.broadcast_to(bias, (batch_size, 1, sequence_length, sequence_length))
            except RuntimeError as error:
                raise ValueError(
                    f"attention_bias is not broadcastable to {(batch_size, 1, sequence_length, sequence_length)}."
                ) from error
        return attention_mask, bias

    def _add_visual_features(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_masks: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
        low_res_token_pooling: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        list[list[torch.Tensor]] | None,
        list[torch.Tensor | None] | None,
        list[torch.Tensor] | None,
    ]:
        if self.vision_backbone is None:
            raise ValueError("Images were provided, but this checkpoint has no vision backbone.")
        if token_pooling is None:
            raise ValueError("MolmoBot images require token_pooling.")

        deep_features: list[list[torch.Tensor]] | None = None
        deep_masks: list[torch.Tensor | None] | None = None
        image_positions: list[torch.Tensor] | None = None

        if low_res_token_pooling is None:
            vision_output = self.vision_backbone(images, image_masks, token_pooling)
            if isinstance(vision_output, list):
                image_features = vision_output[0]
                deep_features = [[features] for features in vision_output[1:]]
                deep_masks = [None]
            else:
                image_features = vision_output
            image_patch_positions = input_ids.reshape(-1) == self._image_high_res_id
            x = x.clone()
            x.reshape(-1, x.shape[-1])[image_patch_positions] += image_features
            if deep_features is not None:
                image_positions = [image_patch_positions]
            return x, deep_features, deep_masks, image_positions

        all_image_features = self.vision_backbone(
            images,
            image_masks,
            [low_res_token_pooling, token_pooling],
        )
        if len(all_image_features) != 2:
            raise RuntimeError("Expected low- and high-resolution vision features.")
        first_features = all_image_features[0][0]
        if isinstance(first_features, list):
            deep_features = [[None, None] for _ in range(len(first_features) - 1)]  # type: ignore[list-item]
            deep_masks = [None, None]
            image_positions = [None, None]  # type: ignore[list-item]

        for feature_index, token_id in enumerate((self._image_low_res_id, self._image_high_res_id)):
            image_features, valid_mask = all_image_features[feature_index]
            if isinstance(image_features, list):
                assert deep_features is not None
                for layer_index, features in enumerate(image_features[1:]):
                    deep_features[layer_index][feature_index] = features
                image_features = image_features[0]
            positions = input_ids.reshape(-1) == token_id
            if deep_features is not None:
                assert image_positions is not None and deep_masks is not None
                image_positions[feature_index] = positions
                deep_masks[feature_index] = valid_mask
            x = x.clone()
            x.reshape(-1, x.shape[-1])[positions] += image_features.reshape(-1, image_features.shape[-1])[
                valid_mask.reshape(-1)
            ]
        return x, deep_features, deep_masks, image_positions

    @staticmethod
    def _apply_deepstack(
        x: torch.Tensor,
        layer_features: list[torch.Tensor],
        image_positions: list[torch.Tensor],
        deep_masks: list[torch.Tensor | None],
    ) -> torch.Tensor:
        for features, positions, valid_mask in zip(layer_features, image_positions, deep_masks, strict=False):
            added = features if valid_mask is None else features.reshape(-1, features.shape[-1])[valid_mask.reshape(-1)]
            x = x.clone()
            x.reshape(-1, x.shape[-1])[positions] += added
        return x

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_masks: torch.Tensor | None = None,
        token_pooling: torch.Tensor | None = None,
        low_res_token_pooling: torch.Tensor | None = None,
        collect_layer_hidden_states: bool = True,
        **unsupported,
    ) -> OLMoOutput:
        """Encode text and images and return pre-norm hidden states per LLM layer."""
        active_unsupported = {
            key: value for key, value in unsupported.items() if value is not None and value is not False
        }
        if active_unsupported:
            raise ValueError(
                "Unsupported text-generation/training arguments in MolmoBot action inference: "
                f"{sorted(active_unsupported)}"
            )
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 2, got shape {tuple(input_ids.shape)}.")
        if images is not None and input_embeddings is not None:
            raise ValueError("Cannot provide both images and input_embeddings.")

        raw_input_ids = input_ids
        valid_tokens = raw_input_ids != -1
        safe_input_ids = torch.where(valid_tokens, raw_input_ids, 0)
        if position_ids is None:
            base_mask = valid_tokens if attention_mask is None else attention_mask.to(torch.bool)
            position_ids = torch.clamp(torch.cumsum(base_mask.to(torch.int32), dim=-1) - 1, min=0)

        if input_embeddings is None:
            embedding_ids = safe_input_ids
            if self.config.shared_low_high_embedding:
                embedding_ids = torch.where(
                    embedding_ids == self._image_low_res_id,
                    self._image_high_res_id,
                    embedding_ids,
                )
            x = self.transformer.wte(embedding_ids)
        else:
            x = input_embeddings

        attention_mask, llm_attention_bias = self._build_attention_bias(
            raw_input_ids,
            attention_mask,
            attention_bias,
            response_mask,
            x.dtype,
        )

        deep_features = None
        deep_masks = None
        image_positions = None
        if images is not None:
            x, deep_features, deep_masks, image_positions = self._add_visual_features(
                x,
                safe_input_ids,
                images,
                image_masks,
                token_pooling,
                low_res_token_pooling,
            )

        if not self.config.llm.rope:
            positions = torch.arange(safe_input_ids.shape[1], dtype=torch.long, device=x.device)[None]
            x = x + self.transformer.wpe(positions)
        x = self.transformer.emb_drop(x)
        if self.config.llm.normalize_input_embeds:
            x = x * (self.config.llm.d_model**0.5)

        layer_states: list[torch.Tensor] | None = [] if collect_layer_hidden_states else None
        for layer_index, block in enumerate(self.transformer.blocks):
            x, _ = block(
                x,
                attention_bias=llm_attention_bias,
                position_ids=position_ids,
                drop_mask=response_mask,
                layer_past=None,
                use_cache=False,
            )
            if deep_features is not None and layer_index < len(deep_features):
                assert image_positions is not None and deep_masks is not None
                x = self._apply_deepstack(
                    x,
                    deep_features[layer_index],
                    image_positions,
                    deep_masks,
                )
            if layer_states is not None:
                layer_states.append(x)

        internal: dict[str, Sequence[torch.Tensor]] | None = None
        if layer_states is not None:
            internal = {"layer_hidden_states": tuple(layer_states)}
        return OLMoOutput(logits=None, internal=internal)


__all__ = ["VideoOlmo", "VideoOlmoConfig"]
