# Inference-only Being-H0.5 runtime retained in-tree.
# Copyright 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from enum import Enum
from importlib import import_module

from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config


class _LazyArchRegistry(Mapping):
    def __init__(self, specs):
        self._specs = specs
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = tuple(_load_symbol(*spec) for spec in self._specs[key])
        return self._cache[key]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)


class _LazySymbolRegistry(Mapping):
    def __init__(self, specs):
        self._specs = specs
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = _load_symbol(*self._specs[key])
        return self._cache[key]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)


def _load_symbol(module_name, attr_name):
    return getattr(import_module(module_name), attr_name)


# ==============================================================================
# Model Architecture Registry
# ==============================================================================

LLM_MODEL_ARCH = _LazyArchRegistry(
    {
        "Qwen2ForCausalLM": (
            ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2_navit", "Qwen2Config"),
            ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2_navit", "Qwen2ForCausalLM"),
            ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen2", "Qwen2Tokenizer"),
        ),
        "Qwen3ForCausalLM": (
            ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen3_navit", "Qwen3Config"),
            ("worldfoundry.base_models.llm_mllm_core.mllm.qwen.beingh.qwen3_navit", "Qwen3ForCausalLM"),
            ("transformers", "AutoTokenizer"),
        ),
    }
)

VIT_MODEL_ARCH = _LazyArchRegistry(
    {
        "InternVisionModel": (
            ("worldfoundry.base_models.llm_mllm_core.mllm.intern_vit.configuration", "InternVisionConfig"),
            ("worldfoundry.base_models.llm_mllm_core.mllm.intern_vit.model", "InternVisionModel"),
        ),
    }
)

CONNECTOR_ARCH = _LazySymbolRegistry(
    {
        "internvl_connector": (
            "worldfoundry.synthesis.action_generation.being_h05.modeling.layers",
            "InternVLConnector",
        ),
    }
)


# ==============================================================================
# Special Tokens
# ==============================================================================

# Basic tokens
BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"

# Vision tokens
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<|vision_start|>"  #'<img>'
IMG_END_TOKEN = "<|vision_end|>"  #'</img>'
IMAGE_TOKEN = "<|image_pad|>"
VIDOE_TOKEN = "<|video_pad|>"

# Spatial tokens
QUAD_START_TOKEN = "<|quad_start|>"  #'<quad>'
QUAD_END_TOKEN = "<|quad_end|>"  #'</quad>'
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"

# Action and state tokens
ACTION_TOKEN = "<|action_pad|>"
STATE_TOKEN = "<state_pad>"

# Prop tokens
PROP_START_TOKEN = "<prop>"
PROP_END_TOKEN = "</prop>"
PROP_CONTEXT_TOKEN = "<PROP_CONTEXT>"

# Absolute transform tokens - Used to mark absolute transform sequences
ABS_START_TOKEN = "<abs>"
ABS_END_TOKEN = "</abs>"
ABS_TRANS_TOKEN = "<ABS_TRANS>"  # Special token placeholder for absolute transform

# Latent vision tokens - Used to mark latent vision sequences
LAT_START_TOKEN = "<lat>"
LAT_END_TOKEN = "</lat>"
LAT_VIS_TOKEN = "<LAT_VIS>"  # Special token placeholder for latent vision


# ==============================================================================
# Constants
# ==============================================================================

BLOCK_SIZE = 130
IGNORE_INDEX = -100

# Image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


# ==============================================================================
# Embodiment Configuration
# ==============================================================================


class EmbodimentTag(Enum):
    """Enumeration of supported robot embodiments."""

    LIBERO_FRANKA = "libero_franka_gripper"
    LIBERO = "libero"
    ROBOCASA = "robocasa"

    NEW_EMBODIMENT = "new_embodiment"


_DATA_CONFIG = load_vla_va_wam_runtime_config("being-h05")
_PREPROCESSING_CONFIG = _DATA_CONFIG["preprocessing"]
TARGET_STATE_ROTATION_TYPE = str(_PREPROCESSING_CONFIG["target_state_rotation_type"])
TARGET_ACTION_ROTATION_TYPE = str(_PREPROCESSING_CONFIG["target_action_rotation_type"])

# Rotation dimension mapping
_ROTATION_DIM_MAP = {
    "rotation_6d": 6,
    "axis_angle": 3,
    "quaterion": 4,
}


def _get_rotation_dim(rotation_type: str) -> int:
    """Get rotation dimension based on rotation type."""
    if rotation_type in _ROTATION_DIM_MAP:
        return _ROTATION_DIM_MAP[rotation_type]
    elif "euler_angles" in rotation_type:
        return 3
    else:
        raise ValueError(f"Unknown rotation type: {rotation_type}")


TARGET_STATE_ROTATION_DIM = _get_rotation_dim(TARGET_STATE_ROTATION_TYPE)
TARGET_ACTION_ROTATION_DIM = _get_rotation_dim(TARGET_ACTION_ROTATION_TYPE)
