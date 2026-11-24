"""Strict local-only TinyVLA checkpoint construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import LlavaPythiaConfig
from .constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from .model import LlavaPythiaForCausalLM


def _has_merged_weights(root: Path) -> bool:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((root / name).is_file() for name in names)


def _has_adapter_weights(root: Path) -> bool:
    return (root / "adapter_config.json").is_file() and any(
        (root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def validate_checkpoint(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"TinyVLA task checkpoint directory is missing: {root}")
    if _has_adapter_weights(root):
        if not (root / "non_lora_trainables.bin").is_file():
            raise FileNotFoundError(f"TinyVLA LoRA checkpoint requires non_lora_trainables.bin under {root}")
        return "lora"
    if _has_merged_weights(root):
        return "merged"
    raise FileNotFoundError(f"TinyVLA task checkpoint contains no merged or LoRA tensor weights: {root}")


def _config_root(checkpoint: Path, explicit: Path | None) -> Path:
    for candidate in (explicit, checkpoint, checkpoint.parent):
        if candidate is not None and (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"TinyVLA requires a local task config.json in {checkpoint} or its parent")


def _normalize_non_lora_keys(weights: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in weights.items()
        if "lora" not in key.lower()
    }
    if any(key.startswith("model.gpt_neox.") for key in normalized):
        normalized = {(key[6:] if key.startswith("model.") else key): value for key, value in normalized.items()}
    return normalized


def _load_lora_model(
    checkpoint: Path,
    base_model: Path,
    config: LlavaPythiaConfig,
    *,
    dtype: torch.dtype,
    attention_backend: str,
) -> LlavaPythiaForCausalLM:
    if not base_model.is_dir() or not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"TinyVLA local base VLM is incomplete: {base_model}")
    model = LlavaPythiaForCausalLM.from_pretrained(
        base_model,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=attention_backend,
        ignore_mismatched_sizes=True,
    )
    non_lora = torch.load(
        checkpoint / "non_lora_trainables.bin",
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(non_lora, dict)
        or not non_lora
        or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in non_lora.items())
    ):
        raise TypeError("TinyVLA non_lora_trainables.bin must contain a non-empty string-to-tensor mapping")
    non_lora = _normalize_non_lora_keys(non_lora)
    expected_action_keys = {key for key in model.state_dict() if key.startswith(("embed_out.", "proj_to_action."))}
    missing_action = sorted(expected_action_keys - set(non_lora))
    if missing_action:
        raise RuntimeError(
            "TinyVLA task checkpoint does not contain the complete action head; "
            f"first missing keys: {missing_action[:8]}"
        )
    incompatible = model.load_state_dict(non_lora, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"TinyVLA non-LoRA checkpoint has unexpected tensors: {incompatible.unexpected_keys[:8]}")

    from peft import PeftModel

    wrapped = PeftModel.from_pretrained(
        model,
        checkpoint,
        local_files_only=True,
        is_trainable=False,
    )
    merged = wrapped.merge_and_unload()
    if not isinstance(merged, LlavaPythiaForCausalLM):
        raise TypeError(f"unexpected TinyVLA merged model type: {type(merged)!r}")
    return merged


def load_local_policy(
    checkpoint: Path,
    *,
    base_model: Path | None,
    config_dir: Path | None,
    device: str,
    dtype: torch.dtype,
    attention_backend: str,
) -> tuple[Any, LlavaPythiaForCausalLM, Any, str]:
    """Load tokenizer, task policy, and image processor without network access."""

    from transformers import AutoTokenizer, CLIPImageProcessor, SiglipImageProcessor

    checkpoint_kind = validate_checkpoint(checkpoint)
    config_root = _config_root(checkpoint, config_dir)
    config = LlavaPythiaConfig.from_pretrained(
        config_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    required_fields = ("action_head_type", "action_dim", "state_dim", "chunk_size")
    missing = [field for field in required_fields if not hasattr(config, field)]
    if missing:
        raise ValueError(f"TinyVLA task config is missing architecture fields: {missing}")

    tokenizer_root = base_model if checkpoint_kind == "lora" else checkpoint
    if tokenizer_root is None:
        raise ValueError("TinyVLA LoRA inference requires a staged local base_model_path")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 1

    if checkpoint_kind == "lora":
        assert base_model is not None
        model = _load_lora_model(
            checkpoint,
            base_model,
            config,
            dtype=dtype,
            attention_backend=attention_backend,
        )
    else:
        model = LlavaPythiaForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            attn_implementation=attention_backend,
        )

    vision_name = str(config.vision_config["vision_tower"]["vision_model_name_or_path"]).lower()
    processor_type = (
        CLIPImageProcessor if "clip" in vision_name else SiglipImageProcessor if "siglip" in vision_name else None
    )
    if processor_type is None:
        raise ValueError(f"unsupported TinyVLA vision tower: {vision_name}")
    processor = None
    for root in (checkpoint, config_root, base_model):
        if root is None:
            continue
        try:
            processor = processor_type.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
            )
            break
        except (OSError, ValueError):
            continue
    if processor is None:
        raise FileNotFoundError("TinyVLA image preprocessor config was not found in local checkpoint assets")

    if bool(getattr(config, "mm_use_im_patch_token", True)):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if bool(getattr(config, "mm_use_im_start_end", False)):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return tokenizer, model, processor, checkpoint_kind


__all__ = ["load_local_policy", "validate_checkpoint"]
