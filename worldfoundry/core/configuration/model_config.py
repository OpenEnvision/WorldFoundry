"""Model-agnostic configuration dataclasses shared by diffusion runtimes.

Released transformers serialize an architecture blob plus a handful of
runtime fields. :class:`ModelConfig` / :class:`ArchConfig` are the
strict update surface: :meth:`ModelConfig.update_model_arch` rejects
unknown architecture keys, and :meth:`ModelConfig.update_model_config`
refuses to replace ``arch_config`` wholesale.

:class:`DiTArchConfig` / :class:`DiTConfig` add the fields every
diffusion transformer shares (hidden size, heads, FSDP/compile
predicates, LoRA maps). :func:`build_kwargs_from_config` keeps only
keys a callable accepts so extra YAML keys do not explode ``__init__``.

Prefer the alias :data:`DiffusionModelConfig` in new code;
``ModelConfig`` remains for existing family subclasses.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# ABI guards — keep extra YAML keys from exploding a constructor
# ──────────────────────────────────────────────────────────────────────────


def require_config_value(config: dict[str, Any], key: str, expected: Any) -> None:
    """Require one serialized configuration value to match its supported ABI."""

    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def build_kwargs_from_config(config: dict[str, Any], target: Callable[..., Any]) -> dict[str, Any]:
    """Keep only configuration keys accepted by a callable."""

    parameters = inspect.signature(target).parameters
    return {key: value for key, value in config.items() if key in parameters}


# ──────────────────────────────────────────────────────────────────────────
# Architecture vs runtime — arch_config is never replaced wholesale
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ArchConfig:
    """Architecture fields loaded from a transformer configuration."""

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Base model configuration with strict architecture updates."""

    arch_config: ArchConfig = field(default_factory=ArchConfig)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown names onto ``arch_config`` so callers write ``config.hidden_size``.

        Instance-dict fields win first; only a miss reaches the nested
        architecture object. :class:`AttributeError` is raised when neither
        surface owns the name so pickle and ``getattr`` stay consistent.
        """

        arch_config = self.__dict__.get("arch_config")
        if arch_config is not None and hasattr(arch_config, name):
            return getattr(arch_config, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the instance dict only; ``__getattr__`` forwarding is not pickleable."""

        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the instance dict without re-entering ``__getattr__``."""

        self.__dict__.update(state)

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        """Copy known architecture fields from *source_model_dict*; reject extras."""
        valid_fields = {item.name for item in fields(self.arch_config)}
        for key, value in source_model_dict.items():
            if key not in valid_fields:
                raise AttributeError(f"{type(self.arch_config).__name__} has no field {key!r}")
            setattr(self.arch_config, key, value)
        post_init = getattr(self.arch_config, "__post_init__", None)
        if callable(post_init):
            post_init()

    def update_model_config(self, source_model_dict: dict[str, Any]) -> None:
        """Copy known runtime fields; refuse a wholesale ``arch_config`` replace."""
        if "arch_config" in source_model_dict:
            raise ValueError("Source model config must not replace arch_config")
        valid_fields = {item.name for item in fields(self)}
        for key, value in source_model_dict.items():
            if key not in valid_fields:
                logger.warning("%s does not contain field %r", type(self).__name__, key)
                raise AttributeError(f"Invalid field: {key}")
            setattr(self, key, value)
        post_init = getattr(self, "__post_init__", None)
        if callable(post_init):
            post_init()


# ──────────────────────────────────────────────────────────────────────────
# Diffusion transformer defaults — compile predicates follow FSDP unless set
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class DiTArchConfig(ArchConfig):
    """Architecture fields common to diffusion transformers."""

    _fsdp_shard_conditions: list[Any] = field(default_factory=list)
    _compile_conditions: list[Any] = field(default_factory=list)
    param_names_mapping: dict[str, Any] = field(default_factory=dict)
    reverse_param_names_mapping: dict[str, Any] = field(default_factory=dict)
    lora_param_names_mapping: dict[str, Any] = field(default_factory=dict)
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    exclude_lora_layers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Default compile shard predicates to the FSDP set so both stay aligned."""

        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()


@dataclass
class DiTConfig(ModelConfig):
    """Runtime configuration common to diffusion transformers."""

    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)
    prefix: str = ""
    quant_config: Any | None = None

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "dit-config") -> Any:
        """Register ``--{prefix}.prefix`` and ``--{prefix}.quant-config`` on *parser*."""
        destination = prefix.replace("-", "_")
        parser.add_argument(
            f"--{prefix}.prefix",
            type=str,
            dest=f"{destination}.prefix",
            default=DiTConfig.prefix,
            help="Prefix for the diffusion transformer",
        )
        parser.add_argument(
            f"--{prefix}.quant-config",
            type=str,
            dest=f"{destination}.quant_config",
            default=None,
            help="Quantization configuration",
        )
        return parser


# Prefer this unambiguous name in new code. ``ModelConfig`` remains for
# compatibility with existing model-family subclasses and serialized targets.
DiffusionModelConfig = ModelConfig


__all__ = [
    "ArchConfig",
    "DiTArchConfig",
    "DiTConfig",
    "DiffusionModelConfig",
    "ModelConfig",
    "build_kwargs_from_config",
    "require_config_value",
]
