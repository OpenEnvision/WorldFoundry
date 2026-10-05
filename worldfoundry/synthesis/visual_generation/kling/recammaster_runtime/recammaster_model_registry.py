from .models.wan_model import WanModel
from .models.wan_video_image_encoder import WanImageEncoder
from .models.wan_video_text_encoder import WanTextEncoder
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.extension_modules.wanx_vae.wanx_vae import (
    WanVAE as WanVideoVAE,
)
from worldfoundry.core.model_loading import load_model_loader_registry


_MODEL_CLASSES = {
    "WanModel": WanModel,
    "WanTextEncoder": WanTextEncoder,
    "WanImageEncoder": WanImageEncoder,
    "WanVideoVAE": WanVideoVAE,
}


_registry = load_model_loader_registry(
    "models/runtime/configs/kling/recammaster/model_loader.yaml",
    _MODEL_CLASSES,
)
model_loader_configs = _registry.model_loader_configs
huggingface_model_loader_configs = _registry.huggingface_model_loader_configs
patch_model_loader_configs = _registry.patch_model_loader_configs

__all__ = [
    "huggingface_model_loader_configs",
    "model_loader_configs",
    "patch_model_loader_configs",
]
