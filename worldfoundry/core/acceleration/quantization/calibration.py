"""Bounded channel observers and checkpoint-bound calibration artifacts.

Calibration is an explicit offline operation. Artifacts contain tensors and
primitive metadata only, load with ``weights_only=True``, and identify the exact
source weights. Observing a module never replaces its inference implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def projection_digest(module: nn.Module) -> str:
    values = {name: tensor_digest(value) for name, value in module.named_parameters(recurse=False)}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class ChannelObserver:
    """One channel maximum and call count per explicitly selected module."""

    def __init__(self, modules: Mapping[str, nn.Module], *, channel_dim: int = -1):
        if not modules:
            raise ValueError("calibration requires selected modules")
        self.modules = dict(modules)
        self.channel_dim = channel_dim
        self.maxima: dict[str, torch.Tensor] = {}
        self.calls = dict.fromkeys(modules, 0)
        self._handles = []
        self.digests = {name: projection_digest(module) for name, module in modules.items()}

    def __enter__(self):
        if self._handles:
            raise RuntimeError("calibration observer is already active")
        try:
            for name, module in self.modules.items():

                def observe(_module, inputs, *, name=name):
                    value = inputs[0].detach()
                    if isinstance(_module, nn.Linear):
                        from .linear import _autocast_linear_input

                        value = _autocast_linear_input(value)
                    if not value.is_floating_point() or value.ndim < 2 or not value.numel():
                        raise ValueError(f"invalid calibration input for {name}")
                    axis = self.channel_dim % value.ndim
                    maximum = value.float().abs().amax(dim=tuple(i for i in range(value.ndim) if i != axis))
                    if not bool(torch.isfinite(maximum).all()):
                        raise ValueError(f"nonfinite calibration input for {name}")
                    previous = self.maxima.get(name)
                    self.maxima[name] = maximum if previous is None else torch.maximum(previous, maximum)
                    self.calls[name] += 1

                self._handles.append(module.register_forward_pre_hook(observe))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def validate(self) -> None:
        if self._handles:
            raise RuntimeError("finish calibration before exporting")
        for name, module in self.modules.items():
            if self.calls[name] == 0:
                raise ValueError(f"calibration did not execute {name}")
            if projection_digest(module) != self.digests[name]:
                raise ValueError(f"source weights changed during calibration: {name}")


def save_calibration(path, *, kind: str, states: Mapping, metadata: Mapping) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"use a new calibration artifact path: {path}")
    if not states or not metadata:
        raise ValueError("calibration states and provenance metadata are required")
    # Validate provenance before serialization; no arbitrary Python objects.
    json.dumps(dict(metadata), allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save({"version": 1, "kind": kind, "states": dict(states), "metadata": dict(metadata)}, stream)


def load_calibration(path, *, kind: str) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or set(artifact) != {"version", "kind", "states", "metadata"}:
        raise ValueError("invalid calibration artifact schema")
    if type(artifact["version"]) is not int or artifact["version"] != 1 or artifact["kind"] != kind:
        raise ValueError("incompatible calibration artifact version or kind")
    if not isinstance(artifact["states"], dict) or not artifact["states"]:
        raise ValueError("calibration artifact must contain named module states")
    if any(not isinstance(name, str) or not name for name in artifact["states"]):
        raise ValueError("calibration module names must be nonempty strings")
    if not isinstance(artifact["metadata"], dict) or not artifact["metadata"]:
        raise ValueError("calibration provenance is required")
    json.dumps(artifact["metadata"], allow_nan=False)
    return artifact
