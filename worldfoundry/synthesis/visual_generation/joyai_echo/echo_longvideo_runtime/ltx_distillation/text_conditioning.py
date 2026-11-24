"""Batch text conditioning and safe, portable cache serialization."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
from typing import TypeAlias

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .models.text_encoder_wrapper import create_language_only_text_encoder, create_text_embeddings_processor

TextCondition: TypeAlias = dict[str, torch.Tensor | None]
RawTextCondition: TypeAlias = tuple[tuple[torch.Tensor, ...], torch.Tensor]

_CACHE_SCHEMA_VERSION = "1"
_REQUIRED_TENSORS = {"video_context", "attention_mask"}


def _release_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def artifact_fingerprint(path: str | Path) -> str:
    """Create a cheap, path-independent fingerprint from artifact names and sizes."""
    root = Path(path).resolve()
    if root.is_file():
        records = [(root.name, root.stat().st_size)]
    elif root.is_dir():
        relevant_names = {
            "checkpoint.json",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.model",
            "tokenizer_config.json",
        }
        files = sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and (item.suffix in {".safetensors", ".pt"} or item.name in relevant_names)
        )
        records = [(item.relative_to(root).as_posix(), item.stat().st_size) for item in files]
    else:
        raise FileNotFoundError(f"model artifact not found: {root}")
    payload = "\n".join(f"{name}:{size}" for name, size in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def conditioning_cache_path(
    cache_root: str | Path,
    prompts_root: str | Path,
    prompt_file: str | Path,
    shot_index: int,
) -> Path:
    prompts_root = Path(prompts_root).resolve()
    prompt_file = Path(prompt_file).resolve()
    try:
        relative_prompt = prompt_file.relative_to(prompts_root)
    except ValueError:
        relative_prompt = Path(prompt_file.name)
    return Path(cache_root) / relative_prompt.with_suffix("") / f"shot_{shot_index:04d}.safetensors"


def save_text_conditioning(
    path: str | Path,
    condition: TextCondition,
    *,
    prompt: str,
    checkpoint_fingerprint: str,
    gemma_fingerprint: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: value.detach().cpu().contiguous() for key, value in condition.items() if isinstance(value, torch.Tensor)
    }
    missing = _REQUIRED_TENSORS - tensors.keys()
    if missing:
        raise ValueError(f"conditioning is missing required tensors: {sorted(missing)}")
    metadata = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "prompt_sha256": prompt_fingerprint(prompt),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "gemma_fingerprint": gemma_fingerprint,
        "has_audio_context": str("audio_context" in tensors).lower(),
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, destination)


def load_text_conditioning(
    path: str | Path,
    *,
    prompt: str,
    checkpoint_fingerprint: str,
    gemma_fingerprint: str,
) -> TextCondition:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"text conditioning cache not found: {source}")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        expected_metadata = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "prompt_sha256": prompt_fingerprint(prompt),
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "gemma_fingerprint": gemma_fingerprint,
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {expected!r})" for key, (actual, expected) in mismatches.items()
            )
            raise ValueError(f"stale or incompatible text conditioning cache {source}: {details}")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118 - safe_open handles are not iterable.

    missing = _REQUIRED_TENSORS - tensors.keys()
    if missing:
        raise ValueError(f"conditioning cache {source} is missing tensors: {sorted(missing)}")
    for key, tensor in tensors.items():
        if tensor.shape[0] != 1:
            raise ValueError(f"conditioning cache {source} has invalid {key} batch shape {tuple(tensor.shape)}")
    return {
        "video_context": tensors["video_context"],
        "audio_context": tensors.get("audio_context"),
        "attention_mask": tensors["attention_mask"],
    }


def encode_prompts_two_stage(
    prompts: list[str],
    *,
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
) -> list[TextCondition]:
    """Encode prompts in batches while keeping Gemma and Echo connectors disjoint."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prompts:
        return []

    text_encoder = create_language_only_text_encoder(
        checkpoint_path=checkpoint_path,
        gemma_path=gemma_path,
        device=device,
        dtype=dtype,
    )
    raw_conditions: list[RawTextCondition] = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(prompts), batch_size):
                prompt_batch = prompts[offset : offset + batch_size]
                hidden_states, attention_mask = text_encoder.encode_batch(prompt_batch)
                for batch_index in range(len(prompt_batch)):
                    raw_conditions.append(
                        (
                            tuple(hidden[batch_index : batch_index + 1].detach().cpu() for hidden in hidden_states),
                            attention_mask[batch_index : batch_index + 1].detach().cpu(),
                        )
                    )
                del hidden_states, attention_mask
    finally:
        del text_encoder
        _release_cuda(device)

    embeddings_processor = create_text_embeddings_processor(
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    conditions: list[TextCondition] = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(raw_conditions), batch_size):
                raw_batch = raw_conditions[offset : offset + batch_size]
                hidden_states = tuple(
                    torch.cat(
                        [condition[0][layer_index] for condition in raw_batch],
                        dim=0,
                    ).to(device)
                    for layer_index in range(len(raw_batch[0][0]))
                )
                attention_mask = torch.cat([condition[1] for condition in raw_batch], dim=0).to(device)
                output = embeddings_processor.process_hidden_states(
                    hidden_states,
                    attention_mask,
                    padding_side="left",
                )
                for batch_index in range(len(raw_batch)):
                    conditions.append(
                        {
                            "video_context": output.video_encoding[batch_index : batch_index + 1].detach().cpu(),
                            "audio_context": (
                                output.audio_encoding[batch_index : batch_index + 1].detach().cpu()
                                if output.audio_encoding is not None
                                else None
                            ),
                            "attention_mask": output.attention_mask[batch_index : batch_index + 1].detach().cpu(),
                        }
                    )
                del hidden_states, attention_mask, output
                raw_batch.clear()
    finally:
        del embeddings_processor, raw_conditions
        _release_cuda(device)
    return conditions
