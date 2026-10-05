"""Lazy checkpoint materialization: mmap safetensors, load tensors on first lookup.

Use :class:`DiskMap` when the full state dict does not fit in CPU RAM (or
when opening every shard at construct time would stall rank 0). Safetensors
stay memory-mapped; ``.pt`` binaries fall back to an in-memory compatibility
loader. After ``buffer_size`` elements the file handles are refreshed so a
long conversion pass does not pin every shard forever.

This is orthogonal to :mod:`worldfoundry.core.vram.device_map` (resident
GPUs) and layer-wise CPU offload (runtime swap). DiskMap only delays
*first* materialization.
"""

import json
import os
from collections.abc import Mapping

import torch
from safetensors import safe_open

from worldfoundry.core.model_loading import load_torch_state_dict
from worldfoundry.core.io.filesystem.storage import read_text_uri

# ──────────────────────────────────────────────────────────────────────────
# Index expansion — flatten sharded safetensors.json into unique shard paths
# ──────────────────────────────────────────────────────────────────────────


def _expand_safetensors_indexes(paths):
    """Expand sharded Safetensors indexes into unique shard paths."""

    expanded = []
    seen = set()
    for raw_path in paths:
        path = str(raw_path)
        if path.endswith(".safetensors.index.json"):
            index = json.loads(read_text_uri(path))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Safetensors index {path!r} does not contain a weight_map")
            root = path.rsplit("/", 1)[0]
            candidates = [f"{root}/{name}" for name in sorted(set(weight_map.values()))]
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


# ──────────────────────────────────────────────────────────────────────────
# Binary .pt adapter — DiskMap lookup assumes a safetensors-shaped reader
# ──────────────────────────────────────────────────────────────────────────


class SafetensorsCompatibleTensor:
    """Shape-only view so ``get_slice`` matches the safetensors reader contract.

    Binary checkpoints have no mmap slice API. Callers that only need
    ``get_shape()`` (key-hash / registry matching) must not force a second
    materialization of the full tensor.
    """

    def __init__(self, tensor):
        """Wrap an already-loaded tensor; the tensor itself is not copied."""
        self.tensor = tensor

    def get_shape(self):
        """Return the tensor shape as a list, matching safetensors ``get_slice``."""
        return list(self.tensor.shape)


class SafetensorsCompatibleBinaryLoader:
    """In-memory ``.pt`` / ``.bin`` reader with the safetensors ``safe_open`` surface.

    DiskMap's materialization loop only knows ``keys`` / ``get_tensor`` /
    ``get_slice``. Pickle checkpoints cannot mmap, so the whole state dict is
    loaded once and served from RAM. Prefer converting to safetensors when
    CPU RAM is the constraint this class exists to paper over.
    """

    def __init__(self, path, device):
        """Load the pickle checkpoint onto ``device``; warn that mmap is unavailable."""
        print(
            "Detected non-safetensors files, which may cause slower loading. It's recommended to convert it to a safetensors file."
        )
        self.state_dict = load_torch_state_dict(path, map_location=device)

    def keys(self):
        """Return checkpoint parameter names (same iteration contract as safetensors)."""
        return self.state_dict.keys()

    def get_tensor(self, name):
        """Return the stored tensor; :exc:`KeyError` if ``name`` is absent."""
        return self.state_dict[name]

    def get_slice(self, name):
        """Return a shape-queryable wrapper; the underlying tensor stays shared."""
        return SafetensorsCompatibleTensor(self.state_dict[name])


# ──────────────────────────────────────────────────────────────────────────
# Lazy mapping — mmap shards, convert dtype on lookup, refresh after buffer_size
# ──────────────────────────────────────────────────────────────────────────


class DiskMap(Mapping):
    """Lazy mapping from checkpoint parameter names to materialized tensors.

    Safetensors files remain memory-mapped and individual weights are loaded on
    lookup. PyTorch binary checkpoints use an in-memory compatibility loader.
    The mapping may apply a state-dict converter and periodically reopen files
    after ``buffer_size`` tensor elements have been materialized.
    """

    def __init__(self, path, device, torch_dtype=None, state_dict_converter=None, buffer_size=10**9):
        """Index one or more checkpoint files.

        Args:
            path: Checkpoint path or list of paths.
            device: Device on which fetched tensors are materialized.
            torch_dtype: Optional dtype conversion applied on lookup.
            state_dict_converter: Optional callable that remaps public model
                keys to keys stored in the checkpoint.
            buffer_size: Number of fetched tensor elements after which file
                handles are refreshed.
        """
        paths = path if isinstance(path, list) else [path]
        self.path = _expand_safetensors_indexes(paths)
        self.device = device
        self.torch_dtype = torch_dtype
        # Env wins so a long conversion job can shrink the pin window without
        # changing every call site. The constructor default is ~1e9 elements.
        if os.environ.get("WORLDFOUNDRY_DISK_MAP_BUFFER_SIZE") is not None:
            self.buffer_size = int(os.environ["WORLDFOUNDRY_DISK_MAP_BUFFER_SIZE"])
        else:
            self.buffer_size = buffer_size
        self.files = []
        self.flush_files()
        self.name_map = {}
        for file_id, file in enumerate(self.files):
            for name in file.keys():
                self.name_map[name] = file_id
        self.rename_dict = self.fetch_rename_dict(state_dict_converter)

    def flush_files(self):
        """Open or refresh checkpoint readers and reset the materialized-element count."""
        if len(self.files) == 0:
            for path in self.path:
                if path.endswith(".safetensors"):
                    self.files.append(safe_open(path, framework="pt", device=str(self.device)))
                else:
                    self.files.append(SafetensorsCompatibleBinaryLoader(path, device=self.device))
        else:
            for i, path in enumerate(self.path):
                if path.endswith(".safetensors"):
                    self.files[i] = safe_open(path, framework="pt", device=str(self.device))
        self.num_params = 0

    def __getitem__(self, name):
        """Materialize one checkpoint tensor, applying rename and dtype policy.

        ``rename_dict`` maps *public* model keys to *stored* checkpoint keys
        so converters do not rewrite every shard. CPU tensors are cloned: the
        mmap view must not be mutated by a later ``.to()``. After
        ``buffer_size`` elements the readers are reopened so a long
        conversion pass does not pin every shard for the process lifetime.

        Failure: :exc:`KeyError` when ``name`` is missing from both the
        rename map and the shard index.
        """
        if self.rename_dict is not None:
            name = self.rename_dict[name]
        file_id = self.name_map[name]
        param = self.files[file_id].get_tensor(name)
        if self.torch_dtype is not None and isinstance(param, torch.Tensor):
            param = param.to(self.torch_dtype)
        if isinstance(param, torch.Tensor) and param.device == "cpu":
            param = param.clone()
        if isinstance(param, torch.Tensor):
            self.num_params += param.numel()
        if self.num_params > self.buffer_size:
            self.flush_files()
        return param

    def fetch_rename_dict(self, state_dict_converter):
        """Build the optional model-key to checkpoint-key mapping."""
        if state_dict_converter is None:
            return None
        state_dict = {}
        for file in self.files:
            for name in file.keys():
                state_dict[name] = name
        state_dict = state_dict_converter(state_dict)
        return state_dict

    def __iter__(self):
        """Iterate public model keys when a converter is bound, else stored names."""
        if self.rename_dict is not None:
            return self.rename_dict.__iter__()
        else:
            return self.name_map.__iter__()

    def __len__(self):
        """Count public keys after conversion; otherwise unique stored names."""
        if self.rename_dict is not None:
            return len(self.rename_dict)
        return len(self.name_map)

    def __contains__(self, x):
        """Membership uses the same key space as :meth:`__iter__` / :meth:`__getitem__`."""
        if self.rename_dict is not None:
            return x in self.rename_dict
        else:
            return x in self.name_map


__all__ = ["DiskMap", "SafetensorsCompatibleBinaryLoader", "SafetensorsCompatibleTensor"]
