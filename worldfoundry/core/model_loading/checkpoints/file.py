"""State-dict loading from safetensors, bins, folders, and remote URIs.

:func:`load_state_dict` is the entry: a path, list of paths, directory,
or ``*.safetensors.index.json`` becomes a flat mapping. Safetensors and
weights-only PyTorch loads are used; later files overwrite duplicate
keys. Hash helpers fingerprint key layouts for loader-registry matching
without reading tensor values.

Remote URIs go through :mod:`worldfoundry.core.io.filesystem.storage`. This is not
the URL dispatcher in :mod:`worldfoundry.core.model_loading.checkpoints.load`.
"""

import hashlib
import json
import os
import pickle
from typing import Any

import torch
from safetensors import safe_open

from worldfoundry.core.io.filesystem.storage import (
    is_dir_uri,
    is_file_uri,
    join_uri,
    list_uri,
    local_path_for_uri,
    parse_uri_scheme,
    read_text_uri,
)

# ──────────────────────────────────────────────────────────────────────────
# Full materialization — path / folder / index / safetensors / GGUF / pickle
# ──────────────────────────────────────────────────────────────────────────


def load_state_dict(file_path, torch_dtype=None, device="cpu", pin_memory=False, verbose=0):
    """Load checkpoint weights from one path, many paths, a folder, or an index.

    Args:
        file_path: Local/remote checkpoint URI, directory, safetensors index, or
            list of any of those. Later files overwrite duplicate keys.
        torch_dtype: Optional dtype conversion applied to tensor values.
        device: Device used while deserializing weights; CPU is the safe default.
        pin_memory: Pin CPU tensors after loading to accelerate a later GPU copy.
        verbose: Print start/finish messages when at least ``1``.

    Returns:
        Flat state-dict mapping parameter names to tensors or checkpoint values.

    Notes:
        File type is selected from the path: directories are scanned,
        ``*.safetensors.index.json`` follows shards, ``*.safetensors`` uses
        safetensors, and other suffixes use the safe PyTorch checkpoint loader.
    """
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype, device, pin_memory=pin_memory, verbose=verbose))
    else:
        file_path = str(file_path)
        if verbose >= 1:
            print(f"Loading file [started]: {file_path}")
        if is_dir_uri(file_path):
            state_dict = load_state_dict_from_folder(
                file_path, torch_dtype=torch_dtype, device=device, pin_memory=False, verbose=verbose
            )
        elif file_path.endswith(".safetensors.index.json"):
            state_dict = load_state_dict_from_safetensors_index(file_path, torch_dtype=torch_dtype, device=device)
        elif file_path.endswith(".safetensors"):
            state_dict = load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
        elif file_path.lower().endswith(".gguf"):
            state_dict = load_state_dict_from_gguf(
                file_path,
                torch_dtype=torch_dtype,
                device=device,
            )
        else:
            state_dict = load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)
        # If load state dict in CPU memory, `pin_memory=True` will make `model.to("cuda")` faster.
        if pin_memory:
            for i in state_dict:
                state_dict[i] = state_dict[i].pin_memory()
        if verbose >= 1:
            print(f"Loading file [done]: {file_path}")
    return state_dict


def load_state_dict_from_folder(file_path, torch_dtype=None, device="cpu", pin_memory=False, verbose=0):
    """Merge checkpoint files under a directory URI.

    Prefers ``*.safetensors.index.json`` shards when present; otherwise loads
    each ``.safetensors`` / ``.bin`` / ``.ckpt`` / ``.pth`` / ``.pt`` file.
    """
    file_path = str(file_path)
    if parse_uri_scheme(file_path) == "file":
        file_names = [join_uri(file_path, file_name) for file_name in os.listdir(file_path)]
    else:
        file_names = list_uri(file_path)
    index_files = sorted(file_name for file_name in file_names if file_name.endswith(".safetensors.index.json"))
    if index_files:
        state_dict = {}
        for file_path_ in index_files:
            state_dict.update(
                load_state_dict_from_safetensors_index(
                    file_path_,
                    torch_dtype=torch_dtype,
                    device=device,
                )
            )
        if pin_memory:
            for name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    state_dict[name] = tensor.pin_memory()
        return state_dict
    state_dict = {}
    for file_path_ in file_names:
        file_name = os.path.basename(file_path_)
        if "." not in file_name:
            continue
        if file_name.rsplit(".", 1)[-1].lower() not in {
            "safetensors",
            "bin",
            "ckpt",
            "pth",
            "pt",
            "gguf",
        }:
            continue
        state_dict.update(
            load_state_dict(
                file_path_,
                torch_dtype=torch_dtype,
                device=device,
                pin_memory=pin_memory,
                verbose=verbose,
            )
        )
    return state_dict


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    """Load one safetensors file, optionally casting tensors to ``torch_dtype``."""
    state_dict = {}
    with local_path_for_uri(file_path) as local_path:
        with safe_open(str(local_path), framework="pt", device=str(device)) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
                if torch_dtype is not None:
                    state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_gguf(file_path, torch_dtype=None, device="cpu"):
    """Load and dequantize one GGUF checkpoint into a flat state dictionary.

    GGUF is an optional checkpoint container, so its dependency is imported
    only when a ``.gguf`` file is selected. Quantized tensors are dequantized
    through the package's reference implementation before normal model key
    conversion and assignment. A later ``QuantizationMode.GGUF`` transform can
    repack eligible linears for compressed runtime storage; loading the file
    alone does not claim a low-bit GEMM.
    """

    try:
        import gguf
    except ImportError as error:
        raise RuntimeError(
            "loading GGUF checkpoints requires the optional 'gguf' package"
        ) from error

    state_dict = {}
    with local_path_for_uri(file_path) as local_path:
        reader = gguf.GGUFReader(str(local_path))
        for entry in reader.tensors:
            name = str(entry.name)
            if not name:
                raise ValueError(f"GGUF checkpoint {file_path!r} contains an empty tensor name")
            if name in state_dict:
                raise ValueError(
                    f"GGUF checkpoint {file_path!r} contains duplicate tensor {name!r}"
                )
            try:
                array = gguf.dequantize(entry.data, entry.tensor_type)
            except NotImplementedError:
                # Plain integer tensors are metadata/buffer payloads rather
                # than quantized floating weights. Preserve their dtype.
                array = entry.data
            tensor = torch.from_numpy(array.copy())
            if torch_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=torch_dtype)
            state_dict[name] = tensor.to(device=device)
    return state_dict


def load_state_dict_from_safetensors_index(file_path, torch_dtype=None, device="cpu"):
    """Load every unique shard referenced by a safetensors index.

    Args:
        file_path: Local or remote ``*.safetensors.index.json`` URI.
        torch_dtype: Optional dtype conversion for each tensor.
        device: Device used by the safetensors reader.

    Returns:
        Merged state dict containing all referenced shards.

    Raises:
        ValueError: The index has no mapping-valued ``weight_map``.
    """
    file_path = str(file_path)
    index = json.loads(read_text_uri(file_path))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Safetensors index {file_path!r} does not contain a weight_map")
    root = file_path.rsplit("/", 1)[0]
    state_dict = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = join_uri(root, shard_name)
        state_dict.update(load_state_dict_from_safetensors(shard_path, torch_dtype=torch_dtype, device=device))
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    """Load a PyTorch pickle checkpoint and unwrap a single conventional wrapper."""
    state_dict = load_torch_state_dict(file_path, map_location=device)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


# ──────────────────────────────────────────────────────────────────────────
# Safe torch.load — weights_only defaults on; never drop it on TypeError
# ──────────────────────────────────────────────────────────────────────────


def _torch_load(checkpoint_path: str | os.PathLike[str], **kwargs: Any) -> Any:
    """``torch.load`` after URI materialization; ``weights_only=True`` unless overridden.

    Failure: do not swallow ``TypeError`` from an old PyTorch that rejects
    ``weights_only`` — dropping that flag would silently run unrestricted
    pickle. The caller must omit the argument explicitly via
    :func:`load_torch_checkpoint` ``weights_only=None``.
    """
    with local_path_for_uri(checkpoint_path) as local_path:
        # Fail closed when the installed PyTorch does not support a requested
        # safety option.  Dropping ``weights_only`` here would silently turn a
        # tensor-only load into unrestricted pickle execution.
        kwargs.setdefault("weights_only", True)
        return torch.load(str(local_path), **kwargs)


def load_torch_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: Any = "cpu",
    weights_only: bool | None = True,
    allow_unsafe_pickle_fallback: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a PyTorch checkpoint with weights-only deserialization by default.

    Args:
        checkpoint_path: Local or remote checkpoint URI.
        map_location: Destination understood by ``torch.load``.
        weights_only: Safe deserialization mode. ``None`` omits the argument for
            compatibility with older PyTorch releases.
        allow_unsafe_pickle_fallback: Retry with unrestricted pickle only after
            a weights-only unpickling failure. Never enable for untrusted files.
        **kwargs: Additional ``torch.load`` options.

    Returns:
        Object produced by ``torch.load``.

    Warnings:
        Setting ``allow_unsafe_pickle_fallback=True`` can execute code embedded
        in a malicious checkpoint.
    """
    load_kwargs = {"map_location": map_location, **kwargs}
    if weights_only is not None:
        load_kwargs["weights_only"] = weights_only
    try:
        return _torch_load(checkpoint_path, **load_kwargs)
    except pickle.UnpicklingError as exc:
        if not (weights_only and allow_unsafe_pickle_fallback and "Weights only load failed" in str(exc)):
            raise
        fallback_kwargs = {"map_location": map_location, "weights_only": False, **kwargs}
        return _torch_load(checkpoint_path, **fallback_kwargs)


def load_torch_state_dict(checkpoint_path: str | os.PathLike[str], *, map_location: Any = "cpu") -> Any:
    """Load a PyTorch state-dict-shaped checkpoint in weights-only mode.

    Args:
        checkpoint_path: Local or remote checkpoint URI.
        map_location: Destination understood by ``torch.load``.

    Returns:
        Deserialized checkpoint object. The function does not unwrap outer
        ``state_dict``/``module`` keys; use ``load_state_dict`` for that policy.
    """
    return load_torch_checkpoint(checkpoint_path, map_location=map_location, weights_only=True)


# ──────────────────────────────────────────────────────────────────────────
# Key-layout fingerprints — identity hint for the loader registry, not a hash of values
# ──────────────────────────────────────────────────────────────────────────


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    """Serialize nested state-dict keys (and optional shapes) into one string."""
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_state_dict_keys(state_dict, with_shape=True):
    """Return a deterministic fingerprint of state-dict structure.

    Args:
        state_dict: Possibly nested mapping whose tensor keys identify a model
            checkpoint layout.
        with_shape: Include tensor dimensions as well as parameter names.

    Returns:
        MD5 hexadecimal digest used for loader-registry matching.

    Notes:
        This is an identity hint, not a content or security hash: tensor values
        are not read. Use ``hash_model_file`` when file content integrity is
        required.
    """
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


def split_state_dict_with_prefix(state_dict):
    """Split a state dict by the first dotted parameter-key segment.

    Args:
        state_dict: Mapping with string parameter names.

    Returns:
        List of sub-dictionaries in deterministic prefix order. Keys without a
        dot form their own prefix group.
    """

    prefix_dict = {}
    for key in sorted(key for key in state_dict if isinstance(key, str)):
        prefix = key if "." not in key else key.split(".", 1)[0]
        prefix_dict.setdefault(prefix, []).append(key)
    return [{key: state_dict[key] for key in keys} for keys in prefix_dict.values()]


def search_for_embeddings(state_dict):
    """Return all tensor leaves from a nested state dict."""

    embeddings = []
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            embeddings.append(value)
        elif isinstance(value, dict):
            embeddings.extend(search_for_embeddings(value))
    return embeddings


def search_parameter(param, state_dict, *, atol=1e-3):
    """Find the first state-dict key whose tensor numerically matches ``param``."""

    for name, candidate in state_dict.items():
        if not isinstance(candidate, torch.Tensor) or param.numel() != candidate.numel():
            continue
        if param.shape == candidate.shape:
            distance = torch.dist(param, candidate)
        else:
            distance = torch.dist(param.flatten(), candidate.flatten())
        if distance < atol:
            return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    """Print parameter-key matches between two state dicts for conversion scripts."""

    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape) >= 1 and source_state_dict[name].shape[0] % 3 == 0:
                length = source_state_dict[name].shape[0] // 3
                rename = [
                    search_parameter(source_state_dict[name][i * length : i * length + length], target_state_dict)
                    for i in range(3)
                ]
                if None not in rename:
                    print(f'"{name}": {rename},')
                    matched_keys.update(rename)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    """Recursively find files matching any suffix in ``extensions``."""

    suffixes = tuple(extensions)
    if is_dir_uri(folder):
        return list_uri(folder, recursive=True, suffix=suffixes)
    if is_file_uri(folder) and str(folder).endswith(suffixes):
        return [str(folder)]
    return []


def load_keys_dict(file_path):
    """Load parameter names and shapes from one path or a list of paths."""
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_keys_dict(file_path_))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_keys_dict_from_safetensors(file_path)
    if file_path.lower().endswith(".gguf"):
        return load_keys_dict_from_gguf(file_path)
    else:
        return load_keys_dict_from_bin(file_path)


def load_keys_dict_from_safetensors(file_path):
    """Read tensor shapes from a safetensors file without materializing values."""
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            keys_dict[k] = f.get_slice(k).get_shape()
    return keys_dict


def load_keys_dict_from_gguf(file_path):
    """Read GGUF tensor names and logical shapes without dequantizing values."""

    try:
        import gguf
    except ImportError as error:
        raise RuntimeError(
            "reading GGUF checkpoint keys requires the optional 'gguf' package"
        ) from error
    keys_dict = {}
    with local_path_for_uri(file_path) as local_path:
        reader = gguf.GGUFReader(str(local_path))
        for entry in reader.tensors:
            keys_dict[str(entry.name)] = [int(value) for value in reversed(entry.shape)]
    return keys_dict


def convert_state_dict_to_keys_dict(state_dict):
    """Replace tensors with shape lists, recursing into nested dictionaries."""
    keys_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            keys_dict[k] = list(v.shape)
        else:
            keys_dict[k] = convert_state_dict_to_keys_dict(v)
    return keys_dict


def load_keys_dict_from_bin(file_path):
    """Load a pickle checkpoint and convert it to a keys-and-shapes mapping."""
    state_dict = load_state_dict_from_bin(file_path)
    keys_dict = convert_state_dict_to_keys_dict(state_dict)
    return keys_dict


def convert_keys_dict_to_single_str(state_dict, with_shape=True):
    """Serialize a keys-and-shapes mapping into one deterministic string."""
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_model_file(path, with_shape=True):
    """Return an MD5 digest of checkpoint key names loaded from *path*."""
    keys_dict = load_keys_dict(path)
    keys_str = convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


__all__ = [
    "convert_keys_dict_to_single_str",
    "convert_state_dict_keys_to_single_str",
    "convert_state_dict_to_keys_dict",
    "hash_model_file",
    "hash_state_dict_keys",
    "load_keys_dict",
    "load_keys_dict_from_bin",
    "load_keys_dict_from_gguf",
    "load_keys_dict_from_safetensors",
    "load_state_dict",
    "load_state_dict_from_bin",
    "load_state_dict_from_folder",
    "load_state_dict_from_gguf",
    "load_state_dict_from_safetensors",
    "load_torch_checkpoint",
    "load_torch_state_dict",
    "build_rename_dict",
    "search_for_embeddings",
    "search_for_files",
    "search_parameter",
    "split_state_dict_with_prefix",
]
