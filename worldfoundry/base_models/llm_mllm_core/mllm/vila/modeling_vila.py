# ruff: noqa: F405 - Names come from the upstream star import.
import copy
import logging
import os
import os.path
from collections import defaultdict, deque
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (
    AutoConfig,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .base_projector import MultimodalProjector, MultimodalProjectorConfig
from .builder import build_llm_and_tokenizer
from .configuration_vila import VILAConfig
from .constants import *  # noqa: F403 - Upstream star import kept as written.
from .media import extract_media
from .media_encoder import BasicImageEncoder, BasicVideoEncoder
from .mm_utils import process_image, process_images
from .siglip_encoder import SiglipVisionTower, SiglipVisionTowerDynamicS2, SiglipVisionTowerS2
from .tokenizer_utils import tokenize_conversation
from .utils import get_model_config


# quick hack for remote code
def get_pg_manager():
    return None


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None
    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(model_type_or_path), f"Resume mm projector path {model_type_or_path} does not exist!"
        return MultimodalProjector.from_pretrained(model_type_or_path, config)
    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config)
        return mm_projector


def build_vision_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(model_name_or_path), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = vision_tower_arch if vision_tower_arch is not None else model_name_or_path

    use_s2 = getattr(config, "s2", False)
    use_dynamic_s2 = getattr(config, "dynamic_s2", False)

    if "siglip" in vision_tower_name:
        if use_dynamic_s2:
            vision_tower = SiglipVisionTowerDynamicS2(model_name_or_path, config)
        elif use_s2:
            vision_tower = SiglipVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise NotImplementedError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = (
        vision_tower.config.hidden_size if not (use_s2 or use_dynamic_s2) else vision_tower.hidden_size
    )
    return vision_tower


class VILAPretrainedModel(PreTrainedModel):
    config_class = VILAConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _no_split_modules: ClassVar[list] = ["Qwen2DecoderLayer", "SiglipEncoderLayer"]

    def __init__(self, config: VILAConfig, *args, **kwargs):
        super().__init__(config)
        self.config = config
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        # loading on auto by default
        device_map = kwargs.get("device_map", "auto")
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        if device_map in ["auto", "cuda"] and torch.cuda.is_available():
            self.mm_projector = self.mm_projector.cuda()
            self.vision_tower = self.vision_tower.cuda()
        # set device_map auto can autoamtically shard llm to different devices
        self.llm, self.tokenizer = self.init_llm(llm_cfg, config, device_map=device_map)

        # NOTE(ligeng): hard code to set padding_side to left
        self.tokenizer.padding_side = "left"
        # TODO(ligeng): need to add other decoders from config
        self.encoders = {"image": BasicImageEncoder(self), "video": BasicVideoEncoder(self)}

        self.post_config()
        self.is_loaded = True

        assert self.llm is not None or self.vision_tower is not None or self.mm_projector is not None, (
            "At least one of the components must be instantiated."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs,
    ):
        config = VILAConfig.from_pretrained(pretrained_model_name_or_path)
        return cls._from_config(config, **kwargs)

    def init_llm(self, llm_config, config, *args, **kwargs):
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_config, config, *args, **kwargs)
        # hard coded for NVILA
        # variables for XGrammar
        # print("DEBUG", len(self.tokenizer.added_tokens_encoder.keys()), self.tokenizer.added_tokens_encoder.keys())
        NUM_EXTRA_TOKENS = len(self.tokenizer.added_tokens_encoder.keys())

        self.pad_token_list = (
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.tokenize("<|endoftext|>")[0],  # for qwen
        )

        # TODO: SENTINEL_TOKEN is not added, need to check with Zhijian
        self.vocab_size = self.tokenizer.vocab_size + NUM_EXTRA_TOKENS
        # XGrammar tokenizer and grammar compiler
        # lazy init only when specified json output during inference
        self.grammar_compiler = None
        self.llm.resize_token_embeddings(len(self.tokenizer))
        return self.llm, self.tokenizer

    def post_config(self):
        ######################################################################
        # TODO: need to check dtype with jason
        self.llm = self.llm.to(torch.float16)
        self.mm_projector = self.mm_projector.to(torch.float16)
        self.vision_tower = self.vision_tower.to(torch.float16)
        ######################################################################
        self.training = self.llm.training
        if self.training:
            self.train()
        else:
            self.eval()
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector


class VILAForCausalLM(VILAPretrainedModel):
    def __init__(self, config: VILAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
        scales = self.get_vision_tower().scales
        resize_output_to_scale_idx = self.get_vision_tower().resize_output_to_scale_idx

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
                # cur_features = rearrange(cur_features, "1 c h w -> (h w) c")

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

    def encode_images(self, images, block_sizes: tuple[int, ...] | None = None):
        if block_sizes is None:
            block_sizes = [None] * len(images)
        if getattr(self.config, "dynamic_s2", False):
            image_features = self.get_vision_tower()(images)
            image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

            image_features = [
                self.split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes, strict=False)
            ]  # list of B * C * H * W tensors
            image_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0
            )  # B * N * C
            image_features = self.get_mm_projector()(image_features)
            image_features = list(
                image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
            )
            image_features = [
                self.merge_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes, strict=False)
            ]  # list of 1 * C * H * W tensors
            image_features = [rearrange(x, "1 c h w -> (h w) c") for x in image_features]  # list of N * C tensors
            if all([feature.shape[0] == image_features[0].shape[0] for feature in image_features]):  # noqa: C419 - Evaluates every element before testing, as upstream.
                image_features = torch.stack(image_features, dim=0)
        else:
            image_features = self.get_vision_tower()(images)
            image_features = self.get_mm_projector()(image_features)
        return image_features

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: dict[str, list[torch.Tensor]],
        media_config: dict[str, dict[str, Any]],
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE(ligeng): deep copy to avoid modifying the original media and media_config
        media = copy.deepcopy(media)
        media_config = copy.deepcopy(media_config)

        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            for name in media:
                self.encoders[name].end_tokens = None

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        if media is not None:  # noqa: SIM108 - Branches carry per-case comments.
            media_embeds = self.__embed_media_tokens(media, media_config)
        else:
            # no media was provided, so we just return an empty dict
            media_embeds = {}

        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0

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
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                elif input_ids[k][pos].item() in self.pad_token_list:
                    # skip pad tokens
                    end = pos + 1
                    pos = end
                    continue
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

        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed! Still {len(media_embeds[name])} left.")

        inputs, labels = self.__truncate_sequence(inputs, labels)

        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self,
        media: dict[str, list[torch.Tensor]],
        media_config: dict[str, dict[str, Any]],
    ) -> dict[str, list[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            embeds[name] = deque(self.encoders[name](media[name], media_config[name]))
        return embeds

    def __truncate_sequence(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if any(len(input) > self.tokenizer.model_max_length for input in inputs):
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

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
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: dict[str, list[torch.Tensor]] | None = None,
        images: torch.FloatTensor | None = None,
        media_config: list | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        if images is not None:
            if media is not None:
                raise ValueError("Both 'media' and 'images' are provided. Please provide only one.")
            print("The 'images' argument is deprecated. Please use 'media' instead.")
            media = {"image": images}

        if media_config is None:
            media_config = defaultdict(dict)

        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask = self._embed(input_ids, media, media_config, labels, attention_mask)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )

        return outputs

    # TODO(ligeng): check how qwen implements this function
    # @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.FloatTensor | None = None,
        media: dict[str, list[torch.Tensor]] | None = None,
        media_config: dict[str, dict[str, Any]] | None = None,
        attention_mask: torch.LongTensor | None = None,
        return_output_ids_only: bool = False,
        **generation_kwargs,
    ) -> torch.LongTensor:
        """
        input_tokens: <image> describe the image
        media:        [Tensor(1, 3, 384, 384), ]
        ----------->
        input_tokens:      36000      001 002 003 004
        input_emds:     <media emd>   001 002 003 004
        """
        # NOTE: hard code to move to GPU
        # input_ids = input_ids.cuda()
        # media = {k: [v.cuda() if v is not None for v in media[k]] for k in media}
        # if attention_mask is not None:
        #     attention_mask = attention_mask.cuda()
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)
        output_ids = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

        if return_output_ids_only:
            return_value = output_ids
        else:
            # by default, return the input_ids and output_ids concatenated to keep consistency with the community VLMs like qwen
            generation_config = generation_kwargs.get("generation_config")
            if generation_config is not None:
                num_generations = generation_config.num_return_sequences
                repeat_input_ids = input_ids.repeat_interleave(num_generations, dim=0)
                return_value = torch.cat([repeat_input_ids, output_ids], dim=-1)
            else:
                return_value = torch.cat([input_ids, output_ids], dim=-1)

        return return_value

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: str | list,
        generation_config: GenerationConfig | None = None,
        response_format=None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        xgr_logits_processor = None

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(media["image"][0], self.config, None, enable_dynamic_res=True).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(media["image"], self.vision_tower.image_processor, self.config).half()
                media[name] = list(images)
            elif name == "video":
                if self.config.image_aspect_ratio == "dynamic" and self.config.video_max_tiles > 1:
                    media[name] = [
                        process_images(
                            images,
                            self.vision_tower.image_processor,
                            self.config,
                            enable_dynamic_res=True,
                            max_tiles=self.config.video_max_tiles,
                        ).half()
                        for images in media[name]
                    ]
                elif self.config.image_aspect_ratio == "dynamic_s2" and self.config.video_max_tiles > 1:
                    self.config.image_processor = self.vision_tower.image_processor
                    if type(self.config.s2_scales) is str:
                        self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                    media[name] = [
                        torch.cat(
                            [
                                process_image(
                                    image,
                                    self.config,
                                    None,
                                    enable_dynamic_s2=True,
                                    max_tiles=self.config.video_max_tiles,
                                )[0].half()
                                for image in images
                            ]
                        )
                        for images in media[name]
                    ]
                else:
                    media[name] = [
                        process_images(images, self.vision_tower.image_processor, self.config).half()
                        for images in media[name]
                    ]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).unsqueeze(0)
        input_ids = input_ids.to(_module_device(self.llm))

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

        # print("input_ids", input_ids.shape)
        # print(input_ids)
        # print(self.tokenizer.batch_decode(input_ids))
        # print("media", {k: len(v) for k, v in media.items()})
        # print("media_config", media_config)
        # print("generation_config", generation_config)
        # Generate the response
        try:
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,  # structured generation
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            logging.warning("Generation failed with sampling, retrying with greedy decoding.")
            generation_config.do_sample = False
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,
            )

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return response

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
