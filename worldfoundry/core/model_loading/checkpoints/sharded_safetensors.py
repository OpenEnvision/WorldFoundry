"""Sharded safetensors loading helpers for Hugging Face-style checkpoints.

Large models ship as ``model.safetensors.index.json`` plus shards
(optionally ``.zst``-compressed).
:func:`load_sharded_safetensors_parallel_with_progress` merges shards
concurrently. :func:`load_safetensors_into_model_streaming` applies one
shard at a time so peak host memory stays bounded by the largest shard,
including meta-device assignment. :func:`safetensor_checkpoint_files`
resolves the unique shard set; :func:`unwrap_model` strips DDP wrappers.

This is not the URL dispatcher in :mod:`worldfoundry.core.model_loading.checkpoints.load`.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load as load_from_bytes
from safetensors.torch import load_file
from tqdm.auto import tqdm

from worldfoundry.core.distributed.runtime.logging import print_per_rank

# ──────────────────────────────────────────────────────────────────────────
# Shard I/O — optional .zst sibling; zstd must not deadlock on stderr
# ──────────────────────────────────────────────────────────────────────────


def _load_shard(shard_path: str, param_names: list[str], num_threads: int | None = None):
    """Load one shard, decompressing a sibling ``.zst`` through the system binary.

    Only the names listed in the index are returned so a shard that also
    carries unused tensors does not inflate the merged state dict.
    ``zstd -T{n}`` is attached (not a separate argv token) because the
    CLI treats a following ``4`` as a filename.
    """

    zstd_path = shard_path + ".zst"
    if os.path.exists(zstd_path):
        start_time = perf_counter()
        print_per_rank(f"Decompressing {zstd_path} with {num_threads} threads")
        cmd = ["zstd", "-d"]
        if num_threads:
            # zstd only accepts the attached spelling (-T4 / --threads=4); a
            # separated "-T", "4" pair is parsed as a file operand.
            cmd.append(f"-T{num_threads}")
        cmd.extend(["-c", zstd_path])

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"zstd binary not found while decompressing {zstd_path}; "
                "install zstd or decompress the shard manually"
            ) from exc
        # communicate() drains stdout and stderr concurrently; reading stdout
        # alone can deadlock once zstd fills the stderr pipe buffer.
        decompressed_data, stderr_data = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Decompression failed: {stderr_data.decode(errors='replace')}")
        print_per_rank(
            f"Decompressed {zstd_path} with {num_threads} threads, duration: {perf_counter() - start_time:.3f}s"
        )

        start_time = perf_counter()
        print_per_rank(f"Loading {shard_path} from zstd file")
        weights = load_from_bytes(decompressed_data)
        print_per_rank(
            f"Loaded {shard_path} from zstd file, duration: {perf_counter() - start_time:.3f}s"
        )
    else:
        weights = load_file(shard_path)

    return {name: weights[name] for name in param_names}


# ──────────────────────────────────────────────────────────────────────────
# Parallel merge — peak RAM is the sum of shards; prefer streaming for 7B+
# ──────────────────────────────────────────────────────────────────────────


def load_sharded_safetensors_parallel_with_progress(checkpoint_dir: str):
    """Load a safetensors checkpoint, reading independent shards concurrently.

    Args:
        checkpoint_dir: Directory containing either ``model.safetensors`` or a
            ``model.safetensors.index.json`` plus its shards. A sibling
            ``.zst`` file is decompressed through the system ``zstd`` command.

    Returns:
        Merged state dictionary for all parameters named by the index.

    Raises:
        RuntimeError: External zstd decompression fails.
        FileNotFoundError: Neither the index nor fallback model file exists.
    """
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_file_path = os.path.join(checkpoint_dir, "model.safetensors")
        return load_file(model_file_path)

    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    shard_map = {}
    for param_name, shard_file in index["weight_map"].items():
        shard_path = os.path.join(checkpoint_dir, shard_file)
        shard_map.setdefault(shard_path, []).append(param_name)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_load_shard, shard_path, param_names): shard_path
            for shard_path, param_names in shard_map.items()
        }
        pbar = tqdm(futures, desc="Loading shards", total=len(futures))
        for future in pbar:
            state_dict.update(future.result())

    return state_dict


# ──────────────────────────────────────────────────────────────────────────
# DDP unwrap + unique shard set — reject ambiguous index / glob matches
# ──────────────────────────────────────────────────────────────────────────


def unwrap_model(model):
    """Strip nested ``.module`` wrappers from one model or a list of models."""
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while hasattr(model_module, "module"):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def safetensor_checkpoint_files(checkpoint_dir: str | os.PathLike[str]) -> list[Path]:
    """Return the unique shards for a Hugging Face-style safetensors checkpoint."""

    root = Path(checkpoint_dir).expanduser().resolve()
    canonical_index = root / "model.safetensors.index.json"
    if canonical_index.is_file():
        index_files = [canonical_index]
    else:
        index_files = sorted(root.glob("*.safetensors.index.json"))
        if len(index_files) > 1:
            raise ValueError(f"ambiguous safetensors indexes in {root}: {index_files}")
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        files = sorted({root / str(name) for name in (index.get("weight_map") or {}).values()})
    else:
        # A Hugging Face policy snapshot may keep preprocessing statistics in
        # sibling ``*.safetensors`` files.  When the canonical unsharded model
        # file exists it is the complete checkpoint, not one shard among all
        # safetensors in the directory.
        model_file = root / "model.safetensors"
        if model_file.is_file():
            files = [model_file]
        else:
            canonical_shards = sorted(root.glob("model-*.safetensors"))
            if canonical_shards:
                files = canonical_shards
            else:
                candidates = sorted(root.glob("*.safetensors"))
                if len(candidates) > 1:
                    raise ValueError(
                        f"ambiguous noncanonical safetensors checkpoint in {root}: {candidates}"
                    )
                files = candidates
    if not files:
        raise FileNotFoundError(f"no *.safetensors checkpoint files found in {root}")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"safetensors index references missing shards: {missing}")
    return files


# ──────────────────────────────────────────────────────────────────────────
# Streaming assign — peak host memory bounded by the largest shard
# ──────────────────────────────────────────────────────────────────────────


def load_safetensors_into_model_streaming(
    model: Any,
    checkpoint_dir: str | os.PathLike[str],
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    assign: bool | None = None,
) -> dict[str, int]:
    """Load checkpoint shards one at a time and validate the complete key set.

    Unlike a merged state-dict loader, peak host memory is bounded by the
    largest shard. Meta-initialized models are materialized tensor-by-tensor
    directly on ``device``; floating checkpoint tensors may also be converted
    to ``dtype`` while streaming. Strict validation is performed after every
    shard has been applied, so the result is equivalent to one strict
    full-checkpoint load.
    """

    expected_state = model.state_dict()
    expected = set(expected_state)
    expected_shapes = {key: tuple(value.shape) for key, value in expected_state.items()}
    del expected_state
    if assign is None:
        # ``assign=True`` is required when the module was initialized on the
        # meta device. For ordinary modules preserve load_state_dict's copy
        # semantics so existing callers are unchanged.
        assign = any(getattr(value, "is_meta", False) for value in model.parameters())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    parameter_slots = dict(model.named_parameters(remove_duplicate=False)) if assign else {}
    buffer_slots = dict(model.named_buffers(remove_duplicate=False)) if assign else {}

    def convert_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Cast only floating tensors so integer index buffers keep their dtype."""

        target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
        if tensor.device == torch.device(device) and tensor.dtype == target_dtype:
            return tensor
        return tensor.to(device=device, dtype=target_dtype)

    def assign_tensor(key: str, tensor: Any) -> None:
        """Replace one meta (or live) slot in place; reject duplicate shard keys."""

        if key in loaded:
            raise RuntimeError(f"duplicate tensor across safetensors shards: {key}")
        old_parameter = parameter_slots.get(key)
        old_buffer = buffer_slots.get(key)
        if old_parameter is None and old_buffer is None:
            unexpected.add(key)
            return
        tensor = convert_tensor(tensor)
        parent_name, _, leaf_name = key.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if old_parameter is not None:
            replacement = torch.nn.Parameter(tensor, requires_grad=old_parameter.requires_grad)
            if hasattr(old_parameter, "_is_hf_initialized"):
                replacement._is_hf_initialized = old_parameter._is_hf_initialized
            parent._parameters[leaf_name] = replacement
        else:
            parent._buffers[leaf_name] = tensor
        loaded.add(key)

    def apply_tensors(tensors: dict[str, Any]) -> None:
        """Apply one shard via ``load_state_dict`` when the module is already materialized."""

        if not tensors:
            return
        duplicate = loaded.intersection(tensors)
        if duplicate:
            raise RuntimeError(f"duplicate tensors across safetensors shards: {sorted(duplicate)[:8]}")
        for key, value in tensors.items():
            if key in expected_shapes and tuple(value.shape) != expected_shapes[key]:
                mismatched.append((key, tuple(value.shape), expected_shapes[key]))
        incompatible = model.load_state_dict(tensors, strict=False, assign=assign)
        loaded.update(tensors)
        unexpected.update(incompatible.unexpected_keys)

    files = safetensor_checkpoint_files(checkpoint_dir)
    for path in files:
        if not assign:
            read_device = "cpu" if dtype is not None else device
            shard = load_file(str(path), device=str(read_device))
            if dtype is not None:
                shard = {key: convert_tensor(value) for key, value in shard.items()}
            apply_tensors(shard)
            del shard
            continue

        # Meta-initialized inference models can accept checkpoint tensors by
        # reference. Index parameter slots once, then replace them directly as
        # tensors stream from the mmap. Calling load_state_dict repeatedly
        # would re-walk the entire module tree for every small group.
        # Read from the mmap on CPU when conversion is requested. This avoids
        # ever allocating a full-precision copy of a large model on the target
        # accelerator before down-casting it.
        read_device = "cpu" if dtype is not None else str(device)
        with safe_open(str(path), framework="pt", device=read_device) as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key in expected_shapes and tuple(tensor.shape) != expected_shapes[key]:
                    mismatched.append((key, tuple(tensor.shape), expected_shapes[key]))
                assign_tensor(key, tensor)
    missing = expected.difference(loaded)
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            "safetensors checkpoint does not exactly match the model: "
            f"missing={sorted(missing)[:20]}, unexpected={sorted(unexpected)[:20]}, "
            f"shape_mismatches={mismatched[:20]}"
        )
    return {
        "files": len(files),
        "loaded_keys": len(loaded),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "shape_mismatches": len(mismatched),
    }


__all__ = [
    "load_safetensors_into_model_streaming",
    "load_sharded_safetensors_parallel_with_progress",
    "safetensor_checkpoint_files",
    "unwrap_model",
]
