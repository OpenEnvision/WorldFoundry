"""Direct in-process X-VLA action inference.

This runtime loads weights through local in-tree classes with
``trust_remote_code=False``.  The upstream repository, FastAPI server, and
robot/simulator clients are never consulted at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_local_hf_model_path, resolve_worldfoundry_path
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
)

_DATA_CONFIG = load_vla_va_wam_runtime_config("xvla")
_REQUIRED_CHECKPOINT_FILES = tuple(str(item) for item in _DATA_CONFIG["required_checkpoint_files"])
_CHECKPOINT_DOMAIN_IDS: dict[str, int | None] = {
    str(repo_id): (int(domain_id) if domain_id is not None else None)
    for repo_id, domain_id in dict(_DATA_CONFIG["checkpoint_domain_ids"]).items()
}


def _profiled_domain_id(checkpoint: str) -> int | None:
    """Return the official embodiment ID for an exact repo ID or local basename."""

    text = str(checkpoint).rstrip("/")
    basename = text.rsplit("/", 1)[-1]
    normalized_basename = basename.replace("--", "/")
    for repo_id, domain_id in _CHECKPOINT_DOMAIN_IDS.items():
        repo_basename = repo_id.rsplit("/", 1)[-1]
        if text == repo_id or basename == repo_basename or normalized_basename == repo_id:
            return domain_id
    return None


@dataclass(frozen=True)
class XVLARuntimeConfig:
    checkpoint_location: str
    device: str = str(_DATA_CONFIG["device"])
    torch_dtype: str = str(_DATA_CONFIG["torch_dtype"])
    cache_dir: str | None = None
    local_files_only: bool = bool(_DATA_CONFIG["local_files_only"])
    revision: str | None = None
    adapter_path: str | None = None
    attention_backend: str = str(_DATA_CONFIG["attention_backend"])
    domain_id: int = int(_DATA_CONFIG["domain_id"])
    denoising_steps: int = int(_DATA_CONFIG["denoising_steps"])
    seed: int = int(_DATA_CONFIG["seed"])


class XVLARuntime:
    """Persistent checkpoint-backed X-VLA policy runtime."""

    def __init__(self, config: XVLARuntimeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._checkpoint_location: str | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        from transformers import AutoImageProcessor, BartTokenizerFast

        from worldfoundry.core.attention import resolve_transformers_attention_implementation
        from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype

        from .configuration import XVLAConfig
        from .modeling import XVLA
        from .processing import XVLAProcessor

        device = resolve_inference_device(self.config.device)
        dtype = resolve_inference_dtype(device, self.config.torch_dtype)
        if not self.config.local_files_only:
            raise ValueError("X-VLA only supports staged local checkpoints; local_files_only must remain true")
        checkpoint_location = str(
            resolve_local_hf_model_path(
                self.config.checkpoint_location,
                required_files=_REQUIRED_CHECKPOINT_FILES,
            )
        )
        common: dict[str, Any] = {
            "cache_dir": self.config.cache_dir,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        common = {key: value for key, value in common.items() if value is not None}
        config = XVLAConfig.from_pretrained(checkpoint_location, **common)
        config._attn_implementation = resolve_transformers_attention_implementation(
            self.config.attention_backend,
            device,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint_location,
            # X-VLA checkpoints were released with the slow CLIP processor.
            # Transformers 5 otherwise silently switches to the fast variant,
            # which can change pixels and checkpoint parity.
            use_fast=False,
            **common,
        )
        # The released processor contract names BartTokenizer/BartTokenizerFast.
        # AutoTokenizer cannot infer that class from model_type="xvla" with
        # trust_remote_code disabled under Transformers 5 and loses pad/bos/eos.
        tokenizer = BartTokenizerFast.from_pretrained(
            checkpoint_location,
            **common,
        )
        processor = XVLAProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = XVLA.from_pretrained(
            checkpoint_location,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **common,
        )
        if self.config.adapter_path:
            from peft import PeftModel

            adapter_path = resolve_worldfoundry_path(self.config.adapter_path)
            if not adapter_path.is_dir():
                adapter_path = resolve_local_hf_model_path(self.config.adapter_path)
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                torch_dtype=dtype,
                is_trainable=False,
                local_files_only=True,
            )
        model = model.to(device=device, dtype=dtype).eval()
        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = dtype
        self._checkpoint_location = checkpoint_location
        return processor, model

    @staticmethod
    def _state(observation: Mapping[str, Any]) -> Any:
        state = first_present(observation, "proprio", "state", "robot_state", "eef_state")
        nested = observation.get("observation")
        if state is None and isinstance(nested, Mapping):
            state = first_present(nested, "proprio", "state", "robot_state", "eef_state")
        return state

    def predict_action(
        self,
        *,
        instruction: str,
        image: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch

        from worldfoundry.core.media.processing.image_utils import load_pil_image
        from worldfoundry.core.utils.tensors.torch import set_seed_everywhere

        processor, model = self._load()
        images = collect_images(
            observation,
            image,
            ("image0", "image1", "image2", "full_image", "wrist_image", "third_person_image"),
        )
        if not images:
            raise ValueError("X-VLA requires at least one RGB image")
        images = [load_pil_image(value, first_sequence_item=False) for value in images[:3]]
        inputs = processor(images=images, language_instruction=instruction)

        expected_proprio = int(model.action_space.dim_proprio)
        state = self._state(observation)
        if state is None:
            if model.use_proprio:
                raise ValueError(f"X-VLA requires a {expected_proprio}D proprio/state vector")
            state = np.zeros(expected_proprio, dtype=np.float32)
        proprio = torch.as_tensor(np.asarray(state), dtype=self._dtype)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or int(proprio.shape[0]) != 1:
            raise ValueError(f"X-VLA runtime expects one proprio vector, got {tuple(proprio.shape)}")
        if int(proprio.shape[-1]) != expected_proprio:
            raise ValueError(
                f"X-VLA {model.action_mode} expects {expected_proprio} proprio values, "
                f"got {proprio.shape[-1]}"
            )

        domain_id = int(self.config.domain_id)
        if domain_id < 0 or domain_id >= int(model.config.num_domains):
            raise ValueError(
                f"domain_id must be in [0, {model.config.num_domains}), got {domain_id}"
            )
        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if value.is_floating_point():
                value = value.to(device=self._device, dtype=self._dtype)
            else:
                value = value.to(device=self._device)
            model_inputs[key] = value
        model_inputs["proprio"] = proprio.to(device=self._device, dtype=self._dtype)
        model_inputs["domain_id"] = torch.tensor([domain_id], device=self._device, dtype=torch.long)

        set_seed_everywhere(int(self.config.seed))
        with torch.inference_mode():
            action = model.generate_actions(
                **model_inputs,
                steps=int(self.config.denoising_steps),
            )[0].float().cpu()
        return completed_action_result(
            model_id="xvla",
            instruction=instruction,
            actions=action.tolist(),
            checkpoint_path=self._checkpoint_location or self.config.checkpoint_location,
            device=str(self._device),
            runtime="worldfoundry.xvla.in_tree_runtime",
            metadata={
                "action_shape": list(action.shape),
                "action_mode": model.action_mode,
                "domain_id": domain_id,
                "denoising_steps": int(self.config.denoising_steps),
                "seed": int(self.config.seed),
                "image_views": len(images),
                "dtype": str(self._dtype),
            },
        )


_RUNTIME_CACHE: dict[tuple[str, str], XVLARuntime] = {}


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
    """Callable entrypoint used by the shared WorldFoundry policy adapter."""

    del action_context
    options = dict(runtime_options or {})
    checkpoint = (
        checkpoint_path
        or str(options.get("checkpoint_ref") or "")
        or str(options.get("repo_id") or "")
        or "2toINF/X-VLA-WidowX"
    )
    domain_value = options.get("domain_id")
    if domain_value in (None, ""):
        domain_value = _profiled_domain_id(checkpoint)
        if domain_value is None:
            raise ValueError(
                "X-VLA requires an explicit domain_id for a foundation or custom checkpoint; "
                "only exact released fine-tuned checkpoint names have an inferred domain"
            )
    config = XVLARuntimeConfig(
        checkpoint_location=checkpoint,
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "float32"),
        cache_dir=str(options["cache_dir"]) if options.get("cache_dir") else None,
        local_files_only=option_bool(options.get("local_files_only"), True),
        revision=str(options["revision"]) if options.get("revision") else None,
        adapter_path=str(options["adapter_path"]) if options.get("adapter_path") else None,
        attention_backend=str(options.get("attention_backend") or options.get("attn_implementation") or "auto"),
        domain_id=option_int(domain_value, 0),
        denoising_steps=option_int(options.get("denoising_steps") or options.get("steps"), 10),
        seed=option_int(options.get("seed"), 0),
    )
    cache_key = (config.checkpoint_location, runtime_options_cache_key(config.__dict__))
    runtime = _RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = XVLARuntime(config)
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime.predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


__all__ = ["XVLARuntime", "XVLARuntimeConfig", "predict_action"]
