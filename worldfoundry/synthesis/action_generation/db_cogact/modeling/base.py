# Inference-only DB-CogACT source retained in-tree.
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from worldfoundry.core.io.paths import resolve_local_hf_model_path

from ..preprocessing.constants import IMAGE_TOKEN_INDEX
from .projector import build_vision_projector
from .vision import build_vision_tower


class DexboticConfig(PretrainedConfig):
    model_type = "dexbotic"
    llm_config: str | PretrainedConfig
    mm_projector_type: str | None = "mlp2x_gelu"
    mm_vision_tower: str | None = None
    chat_template: str | None = "dexbotic"
    init_llm_weights: bool | None = False


@dataclass
class CausalLMOutputDexbotic(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class DexboticPretrainedModel(PreTrainedModel):
    config: DexboticConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _skip_keys_device_placement = "past_key_values"

    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


class DexboticVLMModel(DexboticPretrainedModel):
    def __init__(self, config: DexboticConfig):
        super().__init__(config)
        if config.init_llm_weights:
            if not isinstance(config.llm_config, (str, Path)):
                raise ValueError("init_llm_weights requires a local language-model checkpoint path")
            local_llm = resolve_local_hf_model_path(
                config.llm_config,
                required_files=("config.json",),
            )
            self.llm = AutoModel.from_pretrained(
                local_llm,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.config.init_llm_weights = False
        else:
            if isinstance(config.llm_config, str):
                local_llm = Path(config.llm_config).expanduser()
                if local_llm.exists():
                    llm_config = AutoConfig.from_pretrained(
                        local_llm,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                elif "qwen2.5" in config.llm_config.lower() or "qwen2" in config.llm_config.lower():
                    from transformers import Qwen2Config

                    payload = config.to_dict()
                    payload.pop("model_type", None)
                    payload.pop("llm_config", None)
                    llm_config = Qwen2Config(**payload)
                    attention_implementation = getattr(config, "_attn_implementation", None)
                    if attention_implementation:
                        llm_config._attn_implementation = attention_implementation
                else:
                    raise FileNotFoundError(
                        "DB-CogACT's nested language architecture must be available in-tree; "
                        f"unsupported remote-only config reference {config.llm_config!r}"
                    )
            elif isinstance(config.llm_config, PretrainedConfig):
                llm_config = config.llm_config
            self.llm = AutoModel.from_config(llm_config)
        self._merge_llm()
        if getattr(config, "mm_vision_tower", None) is not None:
            self.mm_vision_tower = self._build_mm_vision_module(config.mm_vision_tower)
        else:
            self.mm_vision_tower = self._build_mm_vision_module(config)
        self.mm_projector = self._build_mm_projector_module(config)

        self.post_init()

    def initialize_model(self, extra_config: dict):
        for key, value in extra_config.items():
            setattr(self.config, key, value)
        if getattr(self.config, "mm_vision_tower", None) is not None:
            self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        else:
            self.mm_vision_tower = self._build_mm_vision_module(self.config)
        self.mm_projector = self._build_mm_projector_module(self.config)

    def _merge_llm(self):
        # merge llm config with self.config, only add missing keys
        llm_config_dict = {
            k: v for k, v in self.llm.config.__dict__.items() if not k.startswith("_") and not hasattr(self.config, k)
        }
        for key, value in llm_config_dict.items():
            setattr(self.config, key, value)
        self.llm.resize_token_embeddings(self.config.vocab_size)

    def _build_mm_projector_module(self, config) -> nn.Module:
        if getattr(self, "mm_projector", None) is not None:
            return self.mm_projector
        self.mm_projector = build_vision_projector(config)

        return self.mm_projector

    def _build_mm_vision_module(self, config) -> nn.Module:
        if getattr(self, "mm_vision_tower", None) is not None:
            return self.mm_vision_tower
        if getattr(config, "vision_config", None) is not None and getattr(config, "processor_config", None) is not None:
            # FIXME: processor should be moved to top level config
            self.mm_vision_tower = build_vision_tower(
                config.vision_config, processor_config=config.processor_config, select_layer=None
            )
        else:
            self.mm_vision_tower = build_vision_tower(config)
        self.config.mm_hidden_size = self.mm_vision_tower.hidden_size

        return self.mm_vision_tower

    def _load_pretrain_projector(self, pretrain_mm_mlp_adapter) -> None:
        if pretrain_mm_mlp_adapter is not None:
            print(f"=> loading pretrain_mm_mlp_adapter from {pretrain_mm_mlp_adapter} ...")
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(mm_projector_weights, Mapping) or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in mm_projector_weights.items()
            ):
                raise TypeError("pretrained DB-CogACT projector must be a string-to-tensor state dictionary")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"), strict=True)

    @property
    def mm_projector_module(self) -> nn.Module:
        return self.mm_projector

    @property
    def mm_projector_prefix(self) -> str:
        return "mm_projector"

    @property
    def mm_vision_module(self) -> nn.Module:
        return self.mm_vision_tower

    @property
    def mm_vision_prefix(self) -> str:
        return "mm_vision"

    @property
    def backbone(self) -> PreTrainedModel:
        return self.llm

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def _extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        def encode_image(image: torch.Tensor) -> torch.Tensor:
            image_features = self.mm_vision_module(image)
            image_features = self.mm_projector_module(image_features)
            return image_features

        if images.ndim == 5:
            # [B n_image, C, H, W] -> [B*n_image, C, H, W]
            concat_images = torch.cat(list(images), dim=0)
            concat_image_features = encode_image(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(concat_image_features, split_sizes, dim=0)  # {[n_image n_token C] * B}
            # {[n_image*n_token C] * B}
            image_features = [x.flatten(0, 1) for x in image_features]
            image_features = torch.stack(image_features, dim=0)  # [B, n_image*n_token, C]
        else:
            image_features = encode_image(images)

        execution_device = getattr(self, "_worldfoundry_execution_device", self.device)
        image_features = image_features.to(execution_device)
        return image_features

    def _prepare_inputs_for_multimodal(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: torch.Tensor | None,
        cache_position: torch.Tensor | None,
        images: torch.Tensor | None,
    ) -> tuple:
        """Insert vision embeddings into an inference prompt."""
        vision = self.mm_vision_module
        if vision is None or images is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, cache_position
        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, cache_position

        image_features = self._extract_vision_features(images)
        original_attention_mask = attention_mask
        original_position_ids = position_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        unpadded_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask, strict=False)]
        new_input_embeds = []
        cur_image_idx = 0
        for ids in unpadded_ids:
            embeds, cur_image_idx = self._insert_multimodal_embeds(image_features, ids, cur_image_idx)
            new_input_embeds.append(embeds)

        max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if max_length is not None:
            new_input_embeds = [embeds[:max_length] for embeds in new_input_embeds]
        padded, attention_mask, position_ids = self._pad_multimodal_embeds(
            new_input_embeds, attention_mask, position_ids
        )
        inputs_embeds = torch.stack(padded, dim=0)
        attention_mask = (
            None if original_attention_mask is None else attention_mask.to(dtype=original_attention_mask.dtype)
        )
        position_ids = None if original_position_ids is None else position_ids
        if original_attention_mask is None or cache_position is None:
            cache_position = None
        else:
            cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, cache_position

    def _insert_multimodal_embeds(self, image_features, input_ids, image_index):
        image_positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        if not image_positions:
            embeds = self.backbone.embed_tokens(input_ids)
            return embeds, image_index + 1

        boundaries = [-1, *image_positions, input_ids.shape[0]]
        token_segments = [input_ids[boundaries[i] + 1 : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        lengths = [segment.shape[0] for segment in token_segments]
        flat_tokens = torch.cat(token_segments)
        embedded_segments = torch.split(self.backbone.embed_tokens(flat_tokens), lengths, dim=0)
        merged = []
        for index, segment in enumerate(embedded_segments):
            merged.append(segment)
            if index < len(image_positions):
                merged.append(image_features[image_index])
                image_index += 1
        return torch.cat(merged), image_index

    def _pad_multimodal_embeds(self, input_embeds, attention_mask, position_ids):
        max_length = max(embeds.shape[0] for embeds in input_embeds)
        batch_size = len(input_embeds)
        padded = []
        padded_attention = torch.zeros(
            (batch_size, max_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        padded_positions = torch.zeros((batch_size, max_length), dtype=position_ids.dtype, device=position_ids.device)
        left_padding = getattr(self.config, "tokenizer_padding_side", "right") == "left"
        for batch_index, embeds in enumerate(input_embeds):
            length = embeds.shape[0]
            padding = torch.zeros((max_length - length, embeds.shape[1]), dtype=embeds.dtype, device=embeds.device)
            if left_padding:
                padded.append(torch.cat((padding, embeds), dim=0))
                padded_attention[batch_index, -length:] = True
                padded_positions[batch_index, -length:] = torch.arange(
                    length, dtype=position_ids.dtype, device=position_ids.device
                )
            else:
                padded.append(torch.cat((embeds, padding), dim=0))
                padded_attention[batch_index, :length] = True
                padded_positions[batch_index, :length] = torch.arange(
                    length, dtype=position_ids.dtype, device=position_ids.device
                )
        return padded, padded_attention, padded_positions


class DexboticForCausalLM(DexboticPretrainedModel, GenerationMixin):
    config_class = DexboticConfig
    _tied_weights_keys: ClassVar[dict] = {}

    def __init__(self, config: DexboticConfig):
        super().__init__(config)
        config.model_type = self.config_class.model_type

        self._real_init(config)

    def _real_init(self, config: DexboticConfig):
        self.model = DexboticVLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return getattr(self, "lm_head", None)

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self, "lm_head"):
            self.lm_head = new_embeddings

    def process_images(self, images):
        vision_tower = self.model.mm_vision_module
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "pad")
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    @staticmethod
    def expand2square(pil_img, background_color):
        from PIL import Image

        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if images is not None:
            _inputs["images"] = images
        return _inputs


class ActionOutputForCausalLM(ABC):
    @abstractmethod
    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs): ...  # noqa: B006 - Keeps the upstream default object and signature.

    def _denorm(self, actions, action_norms) -> np.ndarray:
        """Denormalize the actions
        Args:
            actions (np.array): Normalized actions with shape [T, D]
            action_norms (dict): Dictionary of normalization parameters
        """
        actions = np.clip(actions, -1, 1)
        min, max = np.array(action_norms["min"]), np.array(action_norms["max"])
        min = min.reshape(1, -1)
        max = max.reshape(1, -1)
        actions = min + (actions + 1) * 0.5 * (max - min)
        return actions
