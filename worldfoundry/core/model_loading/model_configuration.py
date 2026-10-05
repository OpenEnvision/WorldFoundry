"""Native configuration helpers shared by checkpoint-shaped PyTorch modules.

Vendored runtimes ship native modules in a Diffusers-like directory
layout without inheriting Diffusers' ``ModelMixin``.
:class:`ConfigNamespace` is a dict with attribute access.
:class:`NativeConfigMixin` registers constructor kwargs, loads
``config.json``, and reconstructs weights via
:func:`worldfoundry.core.model_loading.model.load_model`.
:func:`register_to_config` captures ``__init__`` arguments.

This is not Hydra instantiation
(:mod:`worldfoundry.core.model_loading.factory`).
"""

from __future__ import annotations

import inspect
import json
from functools import wraps
from pathlib import Path

import torch

# ──────────────────────────────────────────────────────────────────────────
# Attribute-access dict — KeyError becomes AttributeError for config.key reads
# ──────────────────────────────────────────────────────────────────────────


class ConfigNamespace(dict):
    """Mapping with attribute access for immutable-looking model configuration."""

    def __getattr__(self, name: str):
        """Read a config key as an attribute; missing keys raise :exc:`AttributeError`."""
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value) -> None:
        """Write through to the mapping so ``config.x = y`` stays JSON-serializable."""
        self[name] = value


# ──────────────────────────────────────────────────────────────────────────
# Native module mixin — register kwargs, load config.json, reconstruct weights
# ──────────────────────────────────────────────────────────────────────────


class NativeConfigMixin:
    """Provide the subset of config registration required by native modules."""

    config: ConfigNamespace

    def register_to_config(self, **values) -> None:
        """Store keyword values on ``self.config``, creating it if needed."""
        config = getattr(self, "config", None)
        if config is None:
            config = ConfigNamespace()
            self.config = config
        config.update(values)

    @classmethod
    def from_config(cls, config, **overrides):
        """Construct a native module from mapping- or attribute-style config data."""

        if hasattr(config, "to_dict"):
            values = dict(config.to_dict())
        elif isinstance(config, dict):
            values = dict(config)
        else:
            values = dict(vars(config))
        values.update(overrides)
        parameters = inspect.signature(cls.__init__).parameters
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            values = {key: value for key, value in values.items() if key in parameters}
        return cls(**values)

    @classmethod
    def load_config(cls, pretrained_model_path, **_kwargs):
        """Load a Diffusers-style ``config.json`` for a native module.

        Several vendored runtimes publish native PyTorch modules in the
        standard Diffusers directory layout.  Keeping this small compatibility
        surface lets those runtimes use their upstream loading calls without
        making every native model inherit Diffusers' ``ModelMixin``.
        """

        path = Path(pretrained_model_path).expanduser()
        config_path = path / "config.json" if path.is_dir() else path
        if not config_path.is_file():
            raise FileNotFoundError(f"native model config not found: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"native model config must be a JSON object: {config_path}")
        return payload

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        *,
        torch_dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        strict: bool = True,
        **overrides,
    ):
        """Load native weights from a Diffusers-style checkpoint directory."""

        root = Path(pretrained_model_path).expanduser()
        subfolder = overrides.pop("subfolder", None)
        if subfolder:
            root = root / str(subfolder)
        if not root.is_dir():
            raise FileNotFoundError(f"native pretrained model directory not found: {root}")
        config = cls.load_config(root)
        config.update(overrides)
        parameters = inspect.signature(cls.__init__).parameters
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            config = {key: value for key, value in config.items() if key in parameters}

        candidates = [
            root / "diffusion_pytorch_model.safetensors.index.json",
            root / "model.safetensors.index.json",
            root / "diffusion_pytorch_model.safetensors",
            root / "model.safetensors",
            root / "pytorch_model.bin",
            root / "diffusion_pytorch_model.bin",
        ]
        checkpoint: Path | list[Path] | None = next((path for path in candidates if path.is_file()), None)
        if checkpoint is None:
            shards = sorted(root.glob("diffusion_pytorch_model-*.safetensors"))
            if not shards:
                shards = sorted(root.glob("model-*.safetensors"))
            checkpoint = shards or None
        if checkpoint is None:
            raise FileNotFoundError(f"native model weights not found under: {root}")

        from worldfoundry.core.model_loading.model import load_model

        return load_model(
            cls,
            checkpoint,
            config=config,
            torch_dtype=torch_dtype or torch.float32,
            device=device,
            strict=strict,
        )


# ──────────────────────────────────────────────────────────────────────────
# Constructor capture — bind args onto self.config before the real __init__
# ──────────────────────────────────────────────────────────────────────────


def register_to_config(initializer):
    """Capture constructor arguments without depending on an external model framework."""

    signature = inspect.signature(initializer)

    @wraps(initializer)
    def wrapped(self, *args, **kwargs):
        """Populate ``self.config`` then call the original initializer."""
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = {name: value for name, value in bound.arguments.items() if name != "self"}
        self.config = ConfigNamespace(values)
        return initializer(self, *args, **kwargs)

    return wrapped


__all__ = ["ConfigNamespace", "NativeConfigMixin", "register_to_config"]
