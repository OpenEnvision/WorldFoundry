"""
configuration_prismatic.py

HuggingFace-style configuration definition for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`.
Default configuration specifies `siglip-224px+7b`.
"""

from typing import Any

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

# === Utilities for Mapping Prismatic names to HF names ===
_ARCHITECTURE_DATA = load_vla_va_wam_runtime_config("openvla-architecture")
VISION_BACKBONE_TO_RESOLUTION: dict[str, list[int]] = {
    str(name): [int(item) for item in payload["resolutions"]]
    for name, payload in _ARCHITECTURE_DATA["vision_backbones"].items()
}
VISION_BACKBONE_TO_TIMM_ID: dict[str, list[str]] = {
    str(name): [str(item) for item in payload["timm_ids"]]
    for name, payload in _ARCHITECTURE_DATA["vision_backbones"].items()
}
TIMM_OVERRIDE_ACT_LAYER: dict[str, list[str | None]] = {
    str(name): [None if item is None else str(item) for item in payload["override_act_layers"]]
    for name, payload in _ARCHITECTURE_DATA["vision_backbones"].items()
}
LLM_BACKBONE_TO_HF_PATH = {
    str(name): str(payload["hf_path"]) for name, payload in _ARCHITECTURE_DATA["language_backbones"].items()
}
LLM_BACKBONE_TO_HF_METACLASS = {
    str(name): str(payload["hf_metaclass"]) for name, payload in _ARCHITECTURE_DATA["language_backbones"].items()
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)


class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: bool | None = None,
        image_resize_strategy: str = "letterbox",
        text_config: dict[str, Any] | None = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }")

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }")

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)
