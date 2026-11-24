"""Lightweight VQAScore model registry.

Model implementations pull in different optional SDKs and model-specific
packages. Keep package import cheap and import the selected implementation only
when a caller asks for that model.
"""

from __future__ import annotations

from importlib import import_module
from typing import NamedTuple

from ...constants import HF_CACHE_DIR


class _ModelSpec(NamedTuple):
    module: str
    model_dict: str
    model_class: str
    names: tuple[str, ...]


_MODEL_SPECS: tuple[_ModelSpec, ...] = (
    _ModelSpec(
        "clip_t5_model",
        "CLIP_T5_MODELS",
        "CLIPT5Model",
        ("clip-flant5-xxl", "clip-flant5-xl", "clip-flant5-xxl-no-system", "clip-flant5-xxl-no-system-no-user"),
    ),
    _ModelSpec(
        "llava_model",
        "LLAVA_MODELS",
        "LLaVAModel",
        ("llava-v1.5-13b", "llava-v1.5-7b", "sharegpt4v-7b", "sharegpt4v-13b"),
    ),
    _ModelSpec("llava16_model", "LLAVA16_MODELS", "LLaVA16Model", ("llava-v1.6-13b",)),
    _ModelSpec(
        "instructblip_model",
        "InstructBLIP_MODELS",
        "InstructBLIPModel",
        ("instructblip-flant5-xxl", "instructblip-flant5-xl"),
    ),
    _ModelSpec("gpt4v_model", "GPT4V_MODELS", "GPT4VModel", ("gpt-4-turbo", "gpt-4o", "gpt-4.1")),
    _ModelSpec(
        "llavaov_model",
        "LLAVA_OV_MODELS",
        "LLaVAOneVisionModel",
        ("llava-onevision-qwen2-7b-si", "llava-onevision-qwen2-7b-ov"),
    ),
    _ModelSpec("mplug_model", "MPLUG_OWL3_MODELS", "mPLUGOwl3Model", ("mplug-owl3-7b",)),
    _ModelSpec(
        "paligemma_model",
        "PALIGEMMA_MODELS",
        "PaliGemmaModel",
        ("paligemma-3b-mix-224", "paligemma-3b-mix-448", "paligemma-3b-mix-896"),
    ),
    _ModelSpec(
        "internvl_model",
        "INTERNVL2_MODELS",
        "InternVL2Model",
        (
            "internvl2-1b",
            "internvl2-2b",
            "internvl2-4b",
            "internvl2-8b",
            "internvl2-26b",
            "internvl2-40b",
            "internvl2-llama3-76b",
            "internvl2.5-1b",
            "internvl2.5-2b",
            "internvl2.5-4b",
            "internvl2.5-8b",
            "internvl2.5-26b",
            "internvl2.5-38b",
            "internvl2.5-78b",
            "internvl3-8b",
            "internvl3-14b",
            "internvl3-78b",
        ),
    ),
    _ModelSpec(
        "internvideo_model",
        "INTERNVIDEO2_MODELS",
        "InternVideo2Model",
        ("internvideo2-chat-8b", "internvideo2-chat-8b-hd", "internvideo2-chat-8b-internlm"),
    ),
    _ModelSpec("internlm_model", "INTERNLMXCOMPOSER25_MODELS", "InternLMXComposer25Model", ("internlmxcomposer25-7b",)),
    _ModelSpec(
        "llama32_model",
        "LLAMA_32_VISION_MODELS",
        "LLaMA32VisionModel",
        (
            "llama-3.2-1b",
            "llama-3.2-3b",
            "llama-3.2-1b-instruct",
            "llama-3.2-3b-instruct",
            "llama-guard-3-1b",
            "llama-3.2-11b-vision",
            "llama-3.2-11b-vision-instruct",
            "llama-3.2-90b-vision",
            "llama-3.2-90b-vision-instruct",
            "llama-guard-3-11b-vision",
        ),
    ),
    _ModelSpec(
        "molmo_model",
        "MOLMO_MODELS",
        "MOLMOVisionModel",
        ("molmo-72b-0924", "molmo-7b-d-0924", "molmo-7b-o-0924", "molmoe-1b-0924"),
    ),
    _ModelSpec(
        "gemini_model",
        "GEMINI_MODELS",
        "GeminiModel",
        ("gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"),
    ),
    _ModelSpec(
        "qwen2vl_model",
        "QWEN2_VL_MODELS",
        "Qwen2VLModel",
        (
            "qwen2-vl-2b",
            "qwen2-vl-7b",
            "qwen2-vl-72b",
            "qwen2.5-vl-3b",
            "qwen2.5-vl-7b",
            "qwen2.5-vl-32b",
            "qwen2.5-vl-72b",
        ),
    ),
    _ModelSpec("llavavideo_model", "LLAVA_VIDEO_MODELS", "LLaVAVideoModel", ("llava-video-7b", "llava-video-72B")),
    _ModelSpec("tarsier_model", "TARSIER_MODELS", "TarsierModel", ("tarsier-recap-7b", "tarsier2-7b")),
    _ModelSpec(
        "perceptionlm_model",
        "PERCEPTION_LM_MODELS",
        "PerceptionLMModel",
        ("perception-lm-1b", "perception-lm-3b", "perception-lm-8b"),
    ),
)

_MODEL_BY_NAME = {name: spec for spec in _MODEL_SPECS for name in spec.names}
_ATTR_TO_SPEC = {spec.model_dict: spec for spec in _MODEL_SPECS} | {spec.model_class: spec for spec in _MODEL_SPECS}


def _load_spec(spec: _ModelSpec):
    return import_module(f"{__name__}.{spec.module}")


def list_all_vqascore_models():
    return [name for spec in _MODEL_SPECS for name in spec.names]


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    try:
        spec = _MODEL_BY_NAME[model_name]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported VQAScore model: {model_name}") from exc
    module = _load_spec(spec)
    model_class = getattr(module, spec.model_class)
    return model_class(model_name, device=device, cache_dir=cache_dir, **kwargs)


def __getattr__(name: str):
    spec = _ATTR_TO_SPEC.get(name)
    if spec is None:
        raise AttributeError(name)
    return getattr(_load_spec(spec), name)


__all__ = [
    "get_vqascore_model",
    "list_all_vqascore_models",
    *sorted(_ATTR_TO_SPEC),
]
