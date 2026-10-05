from __future__ import annotations

from .models.wan_video_dit import WanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import WanS2VModel
from worldfoundry.base_models.diffusion_model.models.encoders.wan import WanImageEncoder, WanTextEncoder
from .models.wan_video_vae import WanVideoVAE, WanVideoVAE38
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants import WanMotionControllerModel
from .models.wan_video_vace import VaceWanModel
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants import WanS2VAudioEncoder
from worldfoundry.core.model_loading import load_model_loader_registry


_CONFIG_RESOURCE = "models/runtime/configs/worldcam/model_loader.yaml"

_MODEL_CLASSES = {
    "WanModel": WanModel,
    "WanS2VModel": WanS2VModel,
    "WanTextEncoder": WanTextEncoder,
    "WanImageEncoder": WanImageEncoder,
    "WanVideoVAE": WanVideoVAE,
    "WanVideoVAE38": WanVideoVAE38,
    "WanMotionControllerModel": WanMotionControllerModel,
    "VaceWanModel": VaceWanModel,
    "WanS2VAudioEncoder": WanS2VAudioEncoder,
}


_registry = load_model_loader_registry(_CONFIG_RESOURCE, _MODEL_CLASSES)
model_loader_configs = _registry.model_loader_configs
huggingface_model_loader_configs = _registry.huggingface_model_loader_configs
patch_model_loader_configs = _registry.patch_model_loader_configs

__all__ = [
    "huggingface_model_loader_configs",
    "model_loader_configs",
    "patch_model_loader_configs",
]
