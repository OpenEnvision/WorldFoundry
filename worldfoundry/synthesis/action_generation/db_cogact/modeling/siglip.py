# Inference-only DB-CogACT source retained in-tree.
import torch
import torch.nn as nn
from transformers import PretrainedConfig, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from worldfoundry.core.io.paths import resolve_local_hf_model_path


def _local_assets(location: str, *required_files: str):
    return resolve_local_hf_model_path(location, required_files=required_files)


class SiglipVisionTower(nn.Module):
    def __init__(
        self,
        vision_tower_config,
        processor_config=None,
        delay_load=False,
        select_layer=-2,
    ):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_config = vision_tower_config
        if processor_config is not None:
            self.processor_config = processor_config
        else:
            assert isinstance(vision_tower_config, str), (
                f"vision_tower_config should be str or `PretrainedConfig, but got {type(self.vision_tower_config)}"
            )
            self.processor_config = vision_tower_config

        self.select_layer = select_layer

        if not delay_load:
            self.load_model()
        else:
            if isinstance(vision_tower_config, str):
                local_vision = _local_assets(self.vision_tower_config, "config.json")
                self.cfg_only = SiglipVisionConfig.from_pretrained(
                    local_vision,
                    local_files_only=True,
                )
            elif isinstance(vision_tower_config, PretrainedConfig):
                self.cfg_only = vision_tower_config
            else:
                raise ValueError(f"vision_tower_config should be str or dict, but got {type(self.vision_tower_config)}")

    def load_model(self):
        if self.is_loaded:
            return
        if isinstance(self.processor_config, str):
            local_processor = _local_assets(
                self.processor_config,
                "preprocessor_config.json",
            )
            self.image_processor = SiglipImageProcessor.from_pretrained(
                local_processor,
                local_files_only=True,
            )
        elif isinstance(self.processor_config, SiglipImageProcessor):
            self.image_processor = self.processor_config
        elif isinstance(self.processor_config, dict):
            self.image_processor = SiglipImageProcessor(**self.processor_config)
        else:
            raise TypeError("processor_config must be a local path, processor, or config mapping")
        self.image_processor.crop_size = self.image_processor.size
        if isinstance(self.vision_tower_config, str):
            local_vision = _local_assets(self.vision_tower_config, "config.json")
            self.vision_tower = SiglipVisionModel.from_pretrained(
                local_vision,
                local_files_only=True,
            )
        elif isinstance(self.vision_tower_config, PretrainedConfig):
            self.vision_tower = SiglipVisionModel(self.vision_tower_config)
        else:
            raise ValueError(
                f"vision_tower_config should be str or `PretrainedConfig, but got {type(self.vision_tower_config)}"
            )

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_layer is None:
            return image_forward_outs.last_hidden_state
        else:
            return image_forward_outs.hidden_states[self.select_layer]

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
        return self.vision_tower.device

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
