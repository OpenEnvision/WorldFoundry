# Inference-only DB-CogACT source retained in-tree.
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _vision_config(vision_tower: str) -> CLIPVisionConfig:
    local = Path(str(vision_tower)).expanduser()
    if local.exists():
        return CLIPVisionConfig.from_pretrained(local, local_files_only=True)
    if str(vision_tower).rstrip("/").lower() == "openai/clip-vit-large-patch14-336":
        # The released DB-CogACT checkpoint contains all CLIP parameters but
        # stores only the upstream model id.  Retain the public architecture
        # locally so checkpoint loading never performs a Hub request.
        return CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
        )
    raise FileNotFoundError(
        f"DB-CogACT vision config must be in-tree or a supported released architecture, got {vision_tower!r}"
    )


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False
        # DB-CogACT checkpoints carry the complete CLIP state dict under the
        # outer model.  Always construct the nested module from local config;
        # Accelerate/Transformers will place it on meta when appropriate and
        # the outer loader fills its parameters.  Checking ``torch.empty(...)
        # .is_meta`` is not a valid way to detect Accelerate's init context.
        self._meta_initialized = True

        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.cfg_only = _vision_config(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel(self.cfg_only)

    def load_model(self):
        if self.is_loaded:
            return
        local = Path(str(self.vision_tower_name)).expanduser()
        if local.exists() and (local / "preprocessor_config.json").is_file():
            self.image_processor = CLIPImageProcessor.from_pretrained(local, local_files_only=True)
        else:
            image_size = int(self.config.image_size)
            self.image_processor = CLIPImageProcessor(
                do_resize=True,
                size={"shortest_edge": image_size},
                do_center_crop=True,
                crop_size={"height": image_size, "width": image_size},
                do_rescale=True,
                rescale_factor=1 / 255,
                do_normalize=True,
                image_mean=_OPENAI_CLIP_MEAN,
                image_std=_OPENAI_CLIP_STD,
                do_convert_rgb=True,
            )

        # Weights are loaded by the outer DB-CogACT checkpoint. This method
        # finalizes only preprocessing and frozen inference state.
        self.vision_tower.requires_grad_(False)

        if hasattr(self, "cfg_only"):
            del self.cfg_only
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        image_features = image_features[:, 1:]

        return image_features

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return getattr(self, "_worldfoundry_execution_device", self.vision_tower.device)

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
