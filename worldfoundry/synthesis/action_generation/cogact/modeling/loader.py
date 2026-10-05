"""Local, inference-only CogACT checkpoint loader."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.core.attention import resolve_transformers_attention_implementation
from worldfoundry.core.model_loading.checkpoints import assign_state_dict_strict
from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
from worldfoundry.core.io.paths import resolve_local_hf_model_path, resolve_worldfoundry_path

from .action import ActionModel
from .policy import CogACT


_DEFAULT_TOKENIZER_REF = "openvla/openvla-7b"
_MODEL_TYPES = {"DiT-S", "DiT-B", "DiT-L"}


def _checkpoint_paths(model_id_or_path: str | Path) -> tuple[Path, Path]:
    direct = resolve_worldfoundry_path(str(model_id_or_path)).expanduser()
    if direct.is_file():
        checkpoint = direct.resolve()
        run_dir = checkpoint.parents[1] if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    else:
        if direct.is_dir():
            run_dir = direct.resolve()
        else:
            run_dir = resolve_local_hf_model_path(
                model_id_or_path,
                required_files=("config.json", "dataset_statistics.json"),
            )
        checkpoint_dir = run_dir / "checkpoints"
        candidates = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"CogACT requires exactly one checkpoints/*.pt file in {run_dir}; found {len(candidates)}"
            )
        checkpoint = candidates[0].resolve()

    for required in (run_dir / "config.json", run_dir / "dataset_statistics.json"):
        if not required.is_file():
            raise FileNotFoundError(f"CogACT checkpoint asset is missing: {required}")
    if checkpoint.suffix != ".pt":
        raise ValueError(f"CogACT checkpoint must be a .pt file, got {checkpoint}")
    return checkpoint, run_dir


def _tokenizer_directory(run_dir: Path, tokenizer_ref: str | Path | None) -> Path:
    required = ("tokenizer_config.json",)
    if all((run_dir / item).is_file() for item in required):
        return run_dir

    refs = [tokenizer_ref, _DEFAULT_TOKENIZER_REF, "meta-llama/Llama-2-7b-hf"]
    errors: list[str] = []
    for ref in dict.fromkeys(str(item) for item in refs if item not in (None, "")):
        try:
            return resolve_local_hf_model_path(ref, required_files=required)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError(
        "CogACT requires local Llama-2 tokenizer assets (no model weights are needed). "
        f"Set tokenizer_ref to a local directory. Attempts: {errors}"
    )


def _map_vision_key(key: str) -> str:
    if key.startswith("dino_featurizer."):
        key = "vision_backbone.featurizer." + key.removeprefix("dino_featurizer.")
    elif key.startswith("siglip_featurizer."):
        key = "vision_backbone.fused_featurizer." + key.removeprefix("siglip_featurizer.")
    return key.replace(".ls1.gamma", ".ls1.scale_factor").replace(
        ".ls2.gamma", ".ls2.scale_factor"
    )


def _map_vlm_state_dict(groups: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    required = {"vision_backbone", "llm_backbone", "projector"}
    missing = sorted(required - set(groups))
    if missing:
        raise KeyError(f"CogACT checkpoint is missing VLM groups: {missing}")

    mapped = {_map_vision_key(key): value for key, value in groups["vision_backbone"].items()}
    for key, value in groups["llm_backbone"].items():
        if not key.startswith("llm."):
            raise KeyError(f"Unexpected CogACT LLM checkpoint key: {key}")
        mapped["language_model." + key.removeprefix("llm.")] = value
    projector_prefixes = {
        "projector.0.": "projector.fc1.",
        "projector.2.": "projector.fc2.",
        "projector.4.": "projector.fc3.",
    }
    for key, value in groups["projector"].items():
        target = next(
            (
                replacement + key[len(prefix) :]
                for prefix, replacement in projector_prefixes.items()
                if key.startswith(prefix)
            ),
            None,
        )
        if target is None:
            raise KeyError(f"Unexpected CogACT projector checkpoint key: {key}")
        mapped[target] = value
    return mapped


def _prismatic_config(attention_implementation: str) -> Any:
    from worldfoundry.synthesis.action_generation.openvla_oft.config import PrismaticConfig

    config = PrismaticConfig(
        vision_backbone_id="dinosiglip-vit-so-224px",
        llm_backbone_id="llama2-7b-pure",
        arch_specifier="no-align+fused-gelu-mlp",
        use_fused_vision_backbone=True,
        image_resize_strategy="resize-naive",
        text_config={
            "model_type": "llama",
            "pad_token_id": 32000,
            "torch_dtype": "bfloat16",
            "vocab_size": 32064,
        },
        llm_max_length=2048,
        pad_token_id=32000,
        pad_to_multiple_of=64,
    )
    config._attn_implementation = attention_implementation
    return config


def _processor(tokenizer_dir: Path) -> Any:
    from transformers import LlamaTokenizerFast

    from worldfoundry.synthesis.action_generation.openvla_oft.preprocessing import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        model_max_length=2048,
        padding_side="right",
    )
    probe = tokenizer("Test 123", add_special_tokens=True).input_ids
    if not probe or probe[0] != tokenizer.bos_token_id:
        raise RuntimeError("CogACT tokenizer must automatically prepend the Llama BOS token")
    if tokenizer.eos_token_id != CogACT.COGNITION_TOKEN_ID:
        raise RuntimeError(
            f"CogACT expects EOS/cognition token {CogACT.COGNITION_TOKEN_ID}, got {tokenizer.eos_token_id}"
        )
    if tokenizer.vocab_size <= CogACT.EMPTY_TOKEN_ID:
        raise RuntimeError("CogACT tokenizer vocabulary is missing the released empty-token id")

    image_processor = PrismaticImageProcessor(
        use_fused_vision_backbone=True,
        image_resize_strategy="resize-naive",
        input_sizes=[(3, 224, 224), (3, 224, 224)],
        interpolations=["bicubic", "bicubic"],
        means=[(0.485, 0.456, 0.406), (0.5, 0.5, 0.5)],
        stds=[(0.229, 0.224, 0.225), (0.5, 0.5, 0.5)],
    )
    return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _materialize_nonpersistent_buffers(vlm: Any) -> None:
    """Recreate buffers intentionally omitted from the released state dict."""

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    vlm.language_model.model.rotary_emb = LlamaRotaryEmbedding(
        vlm.config.text_config,
        device="cpu",
    )
    remaining = [name for name, tensor in vlm.named_buffers() if getattr(tensor, "is_meta", False)]
    if remaining:
        raise RuntimeError(f"CogACT VLM still has unmaterialized meta buffers: {remaining}")


def load_vla(
    model_id_or_path: str | Path,
    *,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
    tokenizer_ref: str | Path | None = None,
    device: str = "cuda",
    torch_dtype: str | torch.dtype = "auto",
    attn_implementation: str = "auto",
    action_model_type: str | None = None,
    future_action_window_size: int | None = None,
    past_action_window_size: int | None = None,
    use_ema: bool = False,
    compile_action_model: bool = False,
) -> CogACT:
    """Load CogACT entirely from local assets; never import or execute remote code."""

    del hf_token, cache_dir  # Retained for source-compatible callers; networking is intentionally unsupported.
    started = time.perf_counter()
    checkpoint_path, run_dir = _checkpoint_paths(model_id_or_path)
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    norm_stats = json.loads((run_dir / "dataset_statistics.json").read_text(encoding="utf-8"))

    selected_model_type = str(
        action_model_type or run_config.get("diffusion_model_type") or run_config.get("action_model_type") or "DiT-B"
    )
    if selected_model_type not in _MODEL_TYPES:
        raise ValueError(f"Unsupported CogACT action model {selected_model_type!r}; choices: {sorted(_MODEL_TYPES)}")
    future_window = int(
        run_config.get("future_action_window_size", 15)
        if future_action_window_size is None
        else future_action_window_size
    )
    past_window = int(
        run_config.get("past_action_window_size", 0)
        if past_action_window_size is None
        else past_action_window_size
    )

    resolved_device = resolve_inference_device(device)
    resolved_dtype = resolve_inference_dtype(resolved_device, torch_dtype)
    attention = resolve_transformers_attention_implementation(attn_implementation, resolved_device)
    processor = _processor(_tokenizer_directory(run_dir, tokenizer_ref))

    from worldfoundry.synthesis.action_generation.openvla_oft.modeling.model import (
        PrismaticForConditionalGeneration,
    )

    with torch.device("meta"):
        vlm = PrismaticForConditionalGeneration(_prismatic_config(attention))
        action_model = ActionModel(
            token_size=4096,
            model_type=selected_model_type,
            in_channels=7,
            future_action_window_size=future_window,
            past_action_window_size=past_window,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    groups = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
    if not isinstance(groups, Mapping):
        raise KeyError(f"CogACT checkpoint {checkpoint_path} has no model mapping")
    assign_state_dict_strict(
        vlm,
        _map_vlm_state_dict(groups),
        label=f"CogACT VLM checkpoint {checkpoint_path}",
    )
    action_group = "ema_diffusion" if use_ema and "ema_diffusion" in groups else "action_model"
    if action_group not in groups:
        raise KeyError(f"CogACT checkpoint {checkpoint_path} has no {action_group!r} group")
    assign_state_dict_strict(
        action_model,
        groups[action_group],
        label=f"CogACT {selected_model_type} checkpoint {checkpoint_path}",
    )

    _materialize_nonpersistent_buffers(vlm)
    vlm = vlm.to(device=resolved_device, dtype=resolved_dtype).eval()
    # The official sampler keeps the diffusion head in FP32; preserving that
    # avoids accumulation drift across 5-100 iterative denoising steps.
    action_model = action_model.to(device=resolved_device, dtype=torch.float32).eval()
    vlm.requires_grad_(False)
    action_model.requires_grad_(False)
    del checkpoint, groups
    gc.collect()

    return CogACT(
        vlm=vlm,
        processor=processor,
        action_model=action_model,
        norm_stats=norm_stats,
        device=resolved_device,
        vlm_dtype=resolved_dtype,
        attention_implementation=attention,
        checkpoint_path=str(checkpoint_path),
        load_seconds=time.perf_counter() - started,
        compile_action_model=compile_action_model,
    ).eval()


__all__ = ["load_vla"]
