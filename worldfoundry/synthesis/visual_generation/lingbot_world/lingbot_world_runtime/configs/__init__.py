import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .shared_config import _load_yaml_config
from .wan_i2v_A14B import i2v_A14B

_registry_cfg = _load_yaml_config("registry.yaml")

WAN_CONFIGS = {
    "i2v-A14B": i2v_A14B,
}

SIZE_CONFIGS = {key: tuple(value) for key, value in _registry_cfg["size_configs"].items()}
MAX_AREA_CONFIGS = {key: height * width for key, (height, width) in SIZE_CONFIGS.items()}
SUPPORTED_SIZES = {key: tuple(value) for key, value in _registry_cfg["supported_sizes"].items()}
