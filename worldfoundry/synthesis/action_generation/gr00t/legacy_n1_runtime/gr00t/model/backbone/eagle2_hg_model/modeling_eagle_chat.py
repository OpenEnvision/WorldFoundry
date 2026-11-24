from typing import ClassVar

from torch import nn
from transformers import LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.utils import logging

logger = logging.get_logger(__name__)
from .configuration_eagle_chat import Eagle2ChatConfig  # noqa


class Eagle2ChatModel(PreTrainedModel):
    config_class = Eagle2ChatConfig
    main_input_name = "pixel_values"
    _no_split_modules: ClassVar[list[str]] = ["LlamaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_cache_class = False
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = False

    def __init__(self, config: Eagle2ChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size

        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))

        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio

        logger.info(f"num_image_token: {self.num_image_token}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == "siglip_vision_model":
                self.vision_model = SiglipVisionModel(config.vision_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size

        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.img_context_token_id = None
        self.system_message = "You are a helpful assistant."  # Default system message

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """ """

        if self.select_layer == -1:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
            # if there is vit_embeds.last_hidden_state, use it.
            if hasattr(vit_embeds, "last_hidden_state"):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        if isinstance(self.vision_model, SiglipVisionModel):
            pass
        else:
            vit_embeds = vit_embeds[:, 1:, :]  # torch.Size([B, 1024, 1024])

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.downsample_ratio
        )  # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0], -1, vit_embeds.shape[-1]
        )  # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])
        vit_embeds = self.mlp1(vit_embeds)  # .to(pixel_values.device)

        return vit_embeds

    def get_prompt(self, system_prompt, messages, sep) -> str:
        """Get the prompt for generation."""

        ret = "" if system_prompt == "" else system_prompt + sep + "\n"
        for role, message in messages:
            if message:
                ret += role + "\n" + message + sep + "\n"
            else:
                ret += role + "\n"
        return ret

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
