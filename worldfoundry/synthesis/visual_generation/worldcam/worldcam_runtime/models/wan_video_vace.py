from worldfoundry.base_models.diffusion_model.models.networks.wan.vace_core import (
    build_vace_wan_classes,
)

from worldfoundry.core.model_loading import hash_state_dict_keys
from .wan_video_dit import DiTBlock

VaceWanAttentionBlock, VaceWanModel, VaceWanModelDictConverter = build_vace_wan_classes(
    DiTBlock,
    hash_state_dict_keys,
    module_name=__name__,
)

__all__ = ["VaceWanAttentionBlock", "VaceWanModel", "VaceWanModelDictConverter"]
