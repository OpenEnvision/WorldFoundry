"""Native in-process LingBot-VLA v1 inference."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

from worldfoundry.synthesis.action_generation._lingbot_vla_runtime import (
    action_values,
    find_training_config,
    populate_images,
    read_yaml,
    resolve_norm_stats,
    resolve_robot_config,
    unwrap_observation,
)
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    completed_action_result,
    option_bool,
    option_int,
    runtime_options_cache_key,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], "_LingBotV1Runtime"] = {}
_CONFIG_ROOT = resolve_data_path("models", "runtime", "configs", "lingbot_vla", "v1")
_ASSET_ROOT = resolve_data_path("models", "runtime", "assets", "lingbot_vla", "v1")
_MODEL_CONFIG = load_vla_va_wam_runtime_config("lingbot-vla")
_BASE_ASSET_PATTERNS = tuple(str(item) for item in _MODEL_CONFIG["base_asset_patterns"])
_CHECKPOINT_PATTERNS = tuple(str(item) for item in _MODEL_CONFIG["checkpoint_patterns"])


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _merge_qwen_config(policy_config: Any, qwen_config: Any) -> Any:
    payload = qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else dict(qwen_config)
    for key in {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }:
        if key in payload:
            setattr(policy_config, key, payload[key])
    if "vision_config" not in payload:
        raise ValueError("Qwen2.5-VL config has no vision_config")
    policy_config.vision_config = qwen_config.vision_config
    return policy_config


class _LingBotV1Runtime:
    def __init__(self, location: str, device: str, options: Mapping[str, Any]) -> None:
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, PretrainedConfig

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.model_loading.checkpoints import load_safetensors_into_model_streaming
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.core.execution.inference import compile_module_if_enabled, install_worldfoundry_inference_infra
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot
        from worldfoundry.core.utils.tensors.torch import freeze_params, set_random_seed

        from .preprocessing.features import FeatureTransform
        from .modeling.policy import LingbotVlaPolicy

        install_worldfoundry_inference_infra()
        self.options = dict(options)
        if not option_bool(options.get("local_files_only"), True):
            raise ValueError("LingBot-VLA is local-only; stage checkpoints with hfd first")
        if option_bool(options.get("trust_remote_code"), False):
            raise ValueError("LingBot-VLA uses only its in-tree model implementation")
        self.device = resolve_inference_device(device)
        self.dtype = resolve_inference_dtype(self.device, options.get("torch_dtype", "auto"))
        set_random_seed(option_int(options.get("seed"), 42))

        self.checkpoint = materialize_hf_snapshot(
            location,
            revision=str(options["revision"]) if options.get("revision") else None,
            cache_dir=options.get("cache_dir"),
            allow_patterns=_CHECKPOINT_PATTERNS,
            required_files=("config.json",),
            local_files_only=True,
            token=options.get("hf_token") or os.environ.get("HF_TOKEN"),
        )
        training_path = find_training_config(
            self.checkpoint,
            options.get("training_config_path"),
            _CONFIG_ROOT / "inference_config.yaml",
        )
        training = read_yaml(training_path)
        if not isinstance(training.get("model"), Mapping) or not isinstance(training.get("data"), Mapping):
            raise ValueError(f"invalid LingBot-VLA training config: {training_path}")

        base_location = str(
            options.get("base_model_path")
            or os.environ.get("QWEN25_PATH")
            or training["model"].get("tokenizer_path")
            or "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        self.base_model = materialize_hf_snapshot(
            base_location,
            revision=str(options["base_revision"]) if options.get("base_revision") else None,
            cache_dir=options.get("cache_dir"),
            allow_patterns=_BASE_ASSET_PATTERNS,
            required_files=("config.json",),
            local_files_only=True,
            token=options.get("hf_token") or os.environ.get("HF_TOKEN"),
        )

        config = PretrainedConfig.from_json_file(str(self.checkpoint / "config.json"))
        combined = {**dict(training["model"]), **dict(training.get("train") or {})}
        config.__dict__.update({key: value for key, value in combined.items() if not hasattr(config, key)})
        config.attention_implementation = str(options.get("attention_implementation") or "eager")
        self.backbone_attention_implementation = resolve_transformers_attention_implementation(
            str(
                options.get("backbone_attention_implementation")
                or options.get("vision_attention_implementation")
                or "flash_attention_2"
            ),
            self.device,
        )
        if self.backbone_attention_implementation == "sdpa":
            # This checkpoint-compatible Qwen implementation has eager and
            # FA2 layers only, unlike the current Transformers implementation.
            logging.getLogger(__name__).warning(
                "LingBot-VLA v1 Qwen layers do not implement SDPA; using eager attention."
            )
            self.backbone_attention_implementation = "eager"
        config.vit_attn_implementation = self.backbone_attention_implementation
        config.tokenizer_path = str(self.base_model)
        config = _merge_qwen_config(
            config,
            AutoConfig.from_pretrained(
                str(self.base_model),
                local_files_only=True,
                trust_remote_code=False,
            ),
        )
        config.torch_dtype = str(self.dtype).removeprefix("torch.")
        if hasattr(config, "qwen_expert_config"):
            config.qwen_expert_config.torch_dtype = config.torch_dtype
        if training["model"].get("vocab_size"):
            config.vocab_size = training["model"]["vocab_size"]
        config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(
            str(self.base_model),
            padding_side="right",
            local_files_only=True,
            trust_remote_code=False,
        )
        data_config = SimpleNamespace(**dict(training["data"]))
        data_config.max_state_dim = config.max_state_dim
        data_config.max_action_dim = config.max_action_dim
        data_config.resize_imgs_with_padding = config.resize_imgs_with_padding
        data_config.tokenizer_max_length = config.tokenizer_max_length

        # Avoid allocating and initializing a full FP32 copy before loading.
        # Checkpoint tensors materialize directly at the requested dtype/device.
        with init_empty_weights(include_buffers=False):
            policy = LingbotVlaPolicy(config, tokenizer_path=str(self.base_model))
        self.weight_report = load_safetensors_into_model_streaming(
            policy,
            self.checkpoint,
            strict=option_bool(options.get("strict_weights"), True),
            device=self.device,
            dtype=self.dtype,
        )
        meta_parameters = [name for name, value in policy.named_parameters() if value.is_meta]
        meta_buffers = [name for name, value in policy.named_buffers() if value.is_meta]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                "LingBot-VLA checkpoint left tensors on the meta device: "
                f"parameters={meta_parameters}, buffers={meta_buffers}"
            )
        policy = policy.to(device=self.device, dtype=self.dtype).eval()
        freeze_params(policy)
        compile_enabled = option_bool(options.get("use_compile"), False)
        policy.model.qwenvl_with_expert = compile_module_if_enabled(
            policy.model.qwenvl_with_expert,
            enabled=compile_enabled,
            label="lingbot_vla_v1_qwenvl_with_expert",
            mode=str(options.get("compile_mode")) if options.get("compile_mode") else None,
            dynamic=option_bool(options.get("compile_dynamic"), False),
        )

        self.robot_name = str(options.get("robot_name") or options.get("robo_name") or "robotwin")
        self.robot_config = resolve_robot_config(_ASSET_ROOT, self.robot_name, options)
        self.norm_stats = resolve_norm_stats(_ASSET_ROOT, "robotwin_50.json", data_config, options)
        self.transform = FeatureTransform(
            str(self.robot_config),
            data_config,
            self.processor.tokenizer,
            self.processor.image_processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(self.norm_stats),
        )
        policy.feature_transform = self.transform
        self.policy = policy
        self.config = config
        self.data_config = data_config
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(option_int(options.get("seed"), 42))

    def infer(self, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        raw = unwrap_observation(observation, instruction)
        image_keys = list(self.transform.org_features["images"])
        populate_images(
            raw,
            image,
            image_keys,
            replicate_single=option_bool(self.options.get("replicate_single_image"), False),
        )
        image_size = int(getattr(self.data_config, "img_size", 224))
        for key in image_keys:
            array = np.asarray(raw[key])
            if array.dtype != np.uint8:
                if array.dtype.kind == "f" and array.max(initial=0) <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            resized = Image.fromarray(array).convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            raw[key] = np.transpose(np.asarray(resized), (2, 0, 1)) / 255.0

        for key, value in list(raw.items()):
            if isinstance(value, np.ndarray):
                raw[key] = torch.from_numpy(value)
        state_keys = list(self.transform.org_features["states"])
        missing_state = [key for key in state_keys if key not in raw]
        if missing_state:
            raise ValueError(f"LingBot-VLA is missing required state features: {missing_state}")
        for key in state_keys:
            if not isinstance(raw[key], torch.Tensor):
                raw[key] = torch.as_tensor(raw[key])
        action_keys = list(self.transform.org_features["actions"])
        state_dim = int(raw[state_keys[0]].shape[-1])
        for key in action_keys:
            raw.setdefault(key, torch.zeros(self.transform.chunk_size, state_dim))
        raw[f"{action_keys[0]}_is_pad"] = torch.zeros(raw[action_keys[0]].shape[0])

        transformed = self.transform.apply(raw)
        images = transformed["images"]
        image_masks = transformed["img_masks"]
        if images.ndim == 4:
            images = images.unsqueeze(0)
            image_masks = image_masks.unsqueeze(0)
        with torch.inference_mode():
            noise = torch.randn(
                (1, self.config.n_action_steps, self.config.max_action_dim),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            actions = self.policy.model.sample_actions(
                images.to(device=self.device, dtype=self.dtype),
                image_masks.to(device=self.device),
                transformed["lang_tokens"].unsqueeze(0).to(device=self.device),
                transformed["lang_masks"].unsqueeze(0).to(device=self.device),
                transformed["state"].unsqueeze(0).to(device=self.device, dtype=self.dtype),
                noise=noise,
                num_steps=option_int(self.options.get("num_denoising_steps"), 10),
            )
        transformed["actions"] = actions.squeeze(0).to(device="cpu", dtype=torch.float32)
        transformed["state"] = transformed["state"].float()
        output = self.transform.unapply(transformed)
        use_length = option_int(self.options.get("use_length"), -1)
        action_chunk: dict[str, Any] = {}
        for key in action_keys:
            if key not in output:
                continue
            value = output[key].float().cpu()
            length = value.shape[0] if use_length < 0 else use_length
            if length > value.shape[0]:
                raise ValueError(f"use_length={length} exceeds LingBot-VLA chunk length {value.shape[0]}")
            action_chunk[key] = value[:length].numpy()
        if not action_chunk:
            raise RuntimeError(f"LingBot-VLA produced no configured action keys; available={sorted(output)}")
        return action_chunk


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> _LingBotV1Runtime:
    key = (location, device, os.environ.get("QWEN25_PATH", ""), runtime_options_cache_key(options))
    if key not in _RUNTIME_CACHE:
        _RUNTIME_CACHE[key] = _LingBotV1Runtime(location, device, options)
    return _RUNTIME_CACHE[key]


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del action_context
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "robbyant/lingbot-vla-4b-posttrain-robotwin")
    runtime = _runtime_for(location, device, options)
    raw_actions = runtime.infer(instruction, image, observation)
    return completed_action_result(
        model_id="lingbot-vla",
        instruction=instruction,
        actions=action_values(raw_actions),
        raw_output=raw_actions,
        checkpoint_path=str(runtime.checkpoint),
        device=runtime.device,
        runtime="worldfoundry.lingbot_vla_v1.native_in_process",
        metadata={
            "official_entrypoint": "LingbotVlaPolicy.model.sample_actions",
            "robot_name": runtime.robot_name,
            "robot_config": str(runtime.robot_config),
            "norm_stats": str(runtime.norm_stats),
            "base_model": str(runtime.base_model),
            "torch_dtype": str(runtime.dtype),
            "backbone_attention_implementation": runtime.backbone_attention_implementation,
            "weight_report": runtime.weight_report,
            "upstream_revision": "4eb34b7693a0",
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
