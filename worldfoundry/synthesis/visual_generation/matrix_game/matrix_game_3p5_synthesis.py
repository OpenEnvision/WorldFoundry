"""Synthesis facades for the two independently registered Matrix-Game 3.5 models."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from worldfoundry.core.io import load_serialized, resolve_data_path
from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .matrix_game_3p5_runtime import MatrixGame35Runtime

RUNTIME_DEFAULTS_PATH = "models/runtime/configs/matrix_game_3p5/runtime_defaults.yaml"
_FRAMEWORK_OPTIONS = frozenset(
    {
        "acquisition_root",
        "adapter",
        "adapter_target",
        "hf_models_root",
        "manifest_path",
        "model_adapter",
        "pipeline_binding",
        "pipeline_target",
        "profile_id",
        "profile_path",
        "repo_id",
        "repo_root",
        "runtime_profile",
        "variant_id",
    }
)


def _runtime_defaults(model_id: str) -> dict[str, Any]:
    payload = load_serialized(resolve_data_path(*Path(RUNTIME_DEFAULTS_PATH).parts))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("defaults"), Mapping):
        raise ValueError(f"Invalid Matrix-Game 3.5 runtime defaults: {RUNTIME_DEFAULTS_PATH}")
    value = payload["defaults"].get(model_id)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing runtime defaults for {model_id!r}")
    return dict(value)


def _normalize_runtime_options(options: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "base_model_path": "wan_dir",
        "ckpt": "checkpoint_path",
        "ckpt_dir": "checkpoint_path",
        "checkpoint_dir": "checkpoint_path",
        "da3_path": "da3_dir",
        "da3_model_path": "da3_dir",
        "guidance_scale": "cfg_scale",
        "model_path": "checkpoint_path",
        "num_inference_steps": "steps",
        "pretrained_model_path": "checkpoint_path",
        "tokenizer_path": "tokenizer_dir",
        "wan_base_dir": "wan_dir",
        "wan_model_path": "wan_dir",
    }
    for source, destination in aliases.items():
        if source in options:
            options[destination] = options[source]
        options.pop(source, None)
    conda_dir = options.pop("conda_dir", None) or options.pop("python_env_dir", None)
    if conda_dir is not None and options.get("python_executable") in (None, ""):
        options["python_executable"] = str(Path(conda_dir).expanduser() / "bin" / "python")
    for key in _FRAMEWORK_OPTIONS:
        options.pop(key, None)

    parameters = inspect.signature(MatrixGame35Runtime).parameters
    unknown = sorted(set(options) - set(parameters))
    if unknown:
        raise TypeError(f"Unsupported Matrix-Game 3.5 runtime options: {', '.join(unknown)}")
    return options


class MatrixGame35Synthesis(BaseSynthesis):
    """Lazy, subprocess-backed synthesis facade with an immutable model ID."""

    MODEL_ID: ClassVar[str]

    def __init__(
        self,
        *,
        runtime_options: Mapping[str, Any] | None = None,
        runtime: MatrixGame35Runtime | None = None,
        lazy: bool = True,
    ) -> None:
        super().__init__()
        if not getattr(self, "MODEL_ID", None):
            raise TypeError("Instantiate a model-specific MatrixGame35Synthesis subclass")
        self.runtime_options = dict(runtime_options or {})
        self._runtime = runtime
        if not lazy:
            self._ensure_runtime()

    def _ensure_runtime(self) -> MatrixGame35Runtime:
        if self._runtime is None:
            self._runtime = MatrixGame35Runtime(**self.runtime_options)
        return self._runtime

    @property
    def runtime(self) -> MatrixGame35Runtime:
        return self._ensure_runtime()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        lazy: bool = True,
        generator_overrides: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "MatrixGame35Synthesis":
        del args
        options = _runtime_defaults(cls.MODEL_ID)
        if isinstance(pretrained_model_path, Mapping):
            options.update(pretrained_model_path)
        elif pretrained_model_path is not None and str(pretrained_model_path).strip():
            options["checkpoint_path"] = pretrained_model_path
        options.update(generator_overrides or {})
        options.update(kwargs)

        requested_model = str(options.pop("model_id", cls.MODEL_ID) or cls.MODEL_ID)
        if requested_model != cls.MODEL_ID:
            raise ValueError(f"{cls.__name__} is bound to {cls.MODEL_ID!r}, got {requested_model!r}")
        options["model_id"] = cls.MODEL_ID
        if device is not None:
            options["device"] = device
        return cls(runtime_options=_normalize_runtime_options(options), lazy=lazy)

    def preflight(self) -> dict[str, Any]:
        return self._ensure_runtime().preflight()

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("return_dict", None)
        return self._ensure_runtime().predict(*args, **kwargs)

    def close(self) -> None:
        self._runtime = None


class MatrixGame35FirstPersonSynthesis(MatrixGame35Synthesis):
    MODEL_ID = "matrix-game-3.5-first-person"


class MatrixGame35ThirdPersonSynthesis(MatrixGame35Synthesis):
    MODEL_ID = "matrix-game-3.5-third-person"


__all__ = [
    "MatrixGame35FirstPersonSynthesis",
    "MatrixGame35Synthesis",
    "MatrixGame35ThirdPersonSynthesis",
]
