"""Native in-process LingBot-VLA v2 inference with official batched semantics."""

from __future__ import annotations

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


_RUNTIME_CACHE: dict[tuple[Any, ...], "_LingBotV2Runtime"] = {}
_CONFIG_ROOT = resolve_data_path("models", "runtime", "configs", "lingbot_vla", "v2")
_ASSET_ROOT = resolve_data_path("models", "runtime", "assets", "lingbot_vla", "v2")
_MODEL_CONFIG = load_vla_va_wam_runtime_config("lingbot-vla-v2")
_BASE_ASSET_PATTERNS = tuple(str(item) for item in _MODEL_CONFIG["base_asset_patterns"])
_CHECKPOINT_PATTERNS = tuple(str(item) for item in _MODEL_CONFIG["checkpoint_patterns"])


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _merge_qwen3_config(config: Any, qwen_config: Any) -> Any:
    payload = qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else dict(qwen_config)
    text = payload.get("text_config") or {}
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
        if key in text:
            setattr(config, key, text[key])
        elif key in payload:
            setattr(config, key, payload[key])
    if "vision_config" not in payload:
        raise ValueError("Qwen3-VL config has no vision_config")
    config.vision_config = qwen_config.vision_config
    return config


class _LingBotV2Runtime:
    def __init__(self, location: str, device: str, options: Mapping[str, Any]) -> None:
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.model_loading.checkpoints import load_safetensors_into_model_streaming
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
        from worldfoundry.core.execution.inference import compile_module_if_enabled, install_worldfoundry_inference_infra
        from worldfoundry.core.io.assets.hf import materialize_hf_snapshot
        from worldfoundry.core.io.paths import resolve_local_hf_model_path
        from worldfoundry.core.utils.tensors.torch import freeze_params, set_random_seed

        from .preprocessing.features import FeatureTransform
        from .modeling.configuration import LingbotVLAV2Config
        from .modeling.policy import LingbotVlaV2Policy
        from .modeling.qwen3_vl import apply_lingbot_qwen3_vl_patch

        install_worldfoundry_inference_infra()
        self.options = dict(options)
        if not option_bool(options.get("local_files_only"), True):
            raise ValueError("LingBot-VLA v2 is local-only; stage checkpoints with hfd first")
        if option_bool(options.get("trust_remote_code"), False):
            raise ValueError("LingBot-VLA v2 uses only its in-tree model implementation")
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
            raise ValueError(f"invalid LingBot-VLA v2 training config: {training_path}")

        model_config = {**dict(training["model"]), **dict(training.get("train") or {})}
        base_location = str(
            options.get("base_model_path")
            or os.environ.get("QWEN3VL_PATH")
            or os.environ.get("QWEN3_PATH")
            or model_config.get("tokenizer_path")
            or "Qwen/Qwen3-VL-4B-Instruct"
        )
        if "qwen3" not in base_location.lower() or "vl" not in base_location.lower():
            raise ValueError(f"LingBot-VLA v2 requires a Qwen3-VL base model, got {base_location!r}")
        self.base_model = materialize_hf_snapshot(
            str(resolve_local_hf_model_path(base_location, required_files=("config.json",))),
            revision=str(options["base_revision"]) if options.get("base_revision") else None,
            cache_dir=options.get("cache_dir"),
            allow_patterns=_BASE_ASSET_PATTERNS,
            required_files=("config.json",),
            local_files_only=True,
            token=options.get("hf_token") or os.environ.get("HF_TOKEN"),
        )

        apply_lingbot_qwen3_vl_patch()
        config = LingbotVLAV2Config(**model_config)
        for key, value in model_config.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        config.attention_implementation = str(options.get("attention_implementation") or "eager")
        self.backbone_attention_implementation = resolve_transformers_attention_implementation(
            str(
                options.get("backbone_attention_implementation")
                or options.get("vision_attention_implementation")
                or "flash_attention_2"
            ),
            self.device,
        )
        config.vit_attn_implementation = self.backbone_attention_implementation
        config.tokenizer_path = str(self.base_model)
        config = _merge_qwen3_config(
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
        config.num_steps = option_int(options.get("num_denoising_steps"), int(getattr(config, "num_steps", 10)))
        config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(
            str(self.base_model),
            padding_side="right",
            local_files_only=True,
            trust_remote_code=False,
        )
        data_config = SimpleNamespace(**dict(training["data"]))
        # The public checkpoint is FP32 and >25 GB. Meta construction prevents
        # a second random FP32 model from existing during startup.
        with init_empty_weights(include_buffers=False):
            policy = LingbotVlaV2Policy(config, eval=True)
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
                "LingBot-VLA v2 checkpoint left tensors on the meta device: "
                f"parameters={meta_parameters}, buffers={meta_buffers}"
            )
        policy = policy.to(device=self.device, dtype=self.dtype).eval()
        freeze_params(policy)

        compile_enabled = option_bool(options.get("use_compile"), False)
        policy.model._use_compile_predict_velocity = compile_enabled
        policy.model._compiled_predict_velocity = None
        policy.model.qwenvl_with_expert = compile_module_if_enabled(
            policy.model.qwenvl_with_expert,
            enabled=compile_enabled,
            label="lingbot_vla_v2_qwenvl_with_expert",
            mode=str(options.get("compile_mode")) if options.get("compile_mode") else None,
            dynamic=option_bool(options.get("compile_dynamic"), False),
        )
        self.sample_actions = compile_module_if_enabled(
            policy.model.sample_actions,
            enabled=compile_enabled and option_bool(options.get("compile_sample_actions"), True),
            label="lingbot_vla_v2_sample_actions",
            mode=str(options.get("compile_mode")) if options.get("compile_mode") else None,
            fullgraph=False,
            dynamic=option_bool(options.get("compile_dynamic"), False),
        )

        self.robot_name = str(options.get("robot_name") or options.get("robo_name") or "robotwin")
        self.robot_config = resolve_robot_config(_ASSET_ROOT, self.robot_name, options)
        self.norm_stats = resolve_norm_stats(_ASSET_ROOT, f"{self.robot_name}.json", data_config, options)
        self.transform = FeatureTransform(
            str(self.robot_config),
            data_config,
            config,
            self.processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(self.norm_stats),
        )
        policy.feature_transform = self.transform
        self.policy = policy
        self.config = config
        self.data_config = data_config
        self.action_keys = list(self.transform.org_features["actions"])
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(option_int(options.get("seed"), 42))

    def _prepare(self, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch
        from torchvision.transforms.v2 import Resize

        raw = unwrap_observation(observation, instruction)
        image_keys = list(self.transform.org_features["images"])
        populate_images(
            raw,
            image,
            image_keys,
            replicate_single=option_bool(self.options.get("replicate_single_image"), False),
        )
        resize = Resize((int(getattr(self.data_config, "img_size", 256)),) * 2)
        for key in image_keys:
            value = torch.as_tensor(np.asarray(raw[key])).permute(2, 0, 1).contiguous().float()
            raw[key] = resize(value)
        for key, value in list(raw.items()):
            if isinstance(value, np.ndarray):
                raw[key] = torch.from_numpy(value)
        state_keys = list(self.transform.org_features["states"])
        missing_state = [key for key in state_keys if key not in raw]
        if missing_state:
            raise ValueError(f"LingBot-VLA v2 is missing required state features: {missing_state}")
        for key in state_keys:
            if not isinstance(raw[key], torch.Tensor):
                raw[key] = torch.as_tensor(raw[key])
        transformed = self.transform.apply(raw, policy_eval=True)
        transformed["state"] = transformed["state"].to(self.dtype)
        return transformed

    @staticmethod
    def _pad_and_stack(values: Sequence[Any]) -> Any:
        import torch
        from worldfoundry.core.utils.tensors.batch import stack_or_pad_tensors

        if not isinstance(values[0], torch.Tensor):
            return list(values)
        return stack_or_pad_tensors(values)

    def infer(self, instruction: str, image: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        batch_source = observation.get("batch")
        is_batch = isinstance(batch_source, Sequence) and not isinstance(batch_source, (str, bytes, bytearray))
        observations = list(batch_source) if is_batch else [observation]
        if not observations or not all(isinstance(item, Mapping) for item in observations):
            raise ValueError("LingBot-VLA v2 batch must be a non-empty sequence of observation mappings")
        transformed = [self._prepare(instruction, None if is_batch else image, item) for item in observations]
        batch: dict[str, Any] = {}
        for key in transformed[0]:
            batch[key] = self._pad_and_stack([item[key] for item in transformed])

        image_grid = batch.get("image_grid_thw")
        if image_grid is not None:
            image_grid = image_grid.to(device=self.device, dtype=torch.long)
        with torch.inference_mode():
            noise = torch.randn(
                (len(observations), self.config.n_action_steps, self.config.max_action_dim),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            actions = self.sample_actions(
                batch["images"].to(device=self.device, dtype=self.dtype),
                batch["img_masks"].to(device=self.device),
                batch["lang_tokens"].to(device=self.device),
                batch["lang_masks"].to(device=self.device),
                batch["state"].to(device=self.device, dtype=self.dtype),
                noise=noise,
                image_grid_thw=image_grid,
            ).to(device="cpu", dtype=torch.float32)

        chunks: dict[str, list[np.ndarray]] = {key: [] for key in self.action_keys}
        for item, action in zip(transformed, actions):
            current = dict(item)
            current["actions"] = action
            current["state"] = current["state"].float()
            output = self.transform.unapply(current)
            for key in self.action_keys:
                if key not in output:
                    raise RuntimeError(f"LingBot-VLA v2 output is missing configured action key {key!r}")
                value = output[key]
                chunks[key].append(value.float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))

        use_length = option_int(self.options.get("use_length"), -1)
        result: dict[str, Any] = {}
        for key, values in chunks.items():
            stacked = np.stack(values, axis=0)
            length = stacked.shape[1] if use_length < 0 else use_length
            if length > stacked.shape[1]:
                raise ValueError(f"use_length={length} exceeds LingBot-VLA v2 chunk length {stacked.shape[1]}")
            stacked = stacked[:, :length]
            result[key] = stacked if is_batch else stacked[0]
        return result


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> _LingBotV2Runtime:
    key = (
        location,
        device,
        os.environ.get("QWEN3VL_PATH", ""),
        os.environ.get("QWEN3_PATH", ""),
        runtime_options_cache_key(options),
    )
    if key not in _RUNTIME_CACHE:
        _RUNTIME_CACHE[key] = _LingBotV2Runtime(location, device, options)
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
    location = checkpoint_path or str(options.get("checkpoint_ref") or "robbyant/lingbot-vla-v2-6b")
    runtime = _runtime_for(location, device, options)
    raw_actions = runtime.infer(instruction, image, observation)
    return completed_action_result(
        model_id="lingbot-vla-v2",
        instruction=instruction,
        actions=action_values(raw_actions),
        raw_output=raw_actions,
        checkpoint_path=str(runtime.checkpoint),
        device=runtime.device,
        runtime="worldfoundry.lingbot_vla_v2.native_in_process",
        metadata={
            "official_entrypoint": "LingbotVlaV2Policy.model.sample_actions",
            "supports_batched_observations": True,
            "robot_name": runtime.robot_name,
            "robot_config": str(runtime.robot_config),
            "norm_stats": str(runtime.norm_stats),
            "base_model": str(runtime.base_model),
            "torch_dtype": str(runtime.dtype),
            "backbone_attention_implementation": runtime.backbone_attention_implementation,
            "weight_report": runtime.weight_report,
            "upstream_revision": "69729b4ef24c",
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
