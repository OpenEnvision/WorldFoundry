"""Strict safetensors inspection and key remapping for Echo-Memory DiTs.

The Echo release stores extra action / SSM / spatial-memory tensors on top
of a Wan2.1-T2V-1.3B backbone.  Before :class:`~...loaders.module.NativeModuleLoader`
constructs :class:`.echo_memory.EchoWanModel`, callers inspect the header
only (no weight materialization) so the architecture — which blocks own
``action_mlp``, ``self_attn_with_action``, ``block_wise_ssm``,
``videossm_hybrid``, and whether a spatial memory adapter is present —
is known at graph-build time.

``normalize_echo_state_dict`` strips vendor prefixes (``module.``,
``model.``, ``dit.``, ``_forward_module.``) and relocates spatial-memory
keys under ``memory_adapter.*`` to match the native module tree.
``load_echo_checkpoint`` is the strict import used by research tooling;
the production denoiser factory uses the converter + native loader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..networks.echo_memory.architecture import (
    EchoArchitectureError,
    EchoCheckpointArchitecture,
    validate_echo_checkpoint_architecture,
)
from ..networks.echo_memory.schema import EchoMemoryRecipe

_BLOCK_PATTERNS = {
    "action": re.compile(r"^blocks\.(\d+)\.action_mlp\."),
    "action_attention": re.compile(r"^blocks\.(\d+)\.self_attn_with_action\."),
    "block_ssm": re.compile(r"^blocks\.(\d+)\.block_wise_ssm\."),
    "video_ssm": re.compile(r"^blocks\.(\d+)\.videossm_hybrid\."),
}
_STATE_PREFIXES = ("module.", "model.", "dit.", "_forward_module.")


EchoCheckpointError = EchoArchitectureError


@dataclass(frozen=True)
class EchoCheckpointInspection:
    """Header-only inspection result used before model construction."""

    path: str
    tensor_count: int
    architecture: EchoCheckpointArchitecture
    normalized_prefix: str | None = None


@dataclass(frozen=True)
class EchoCheckpointLoadReport:
    """Evidence returned after an exact key/shape import."""

    path: str
    tensor_count: int
    model_tensor_count: int
    architecture: EchoCheckpointArchitecture


def _strip_common_prefix(keys: tuple[str, ...]) -> tuple[tuple[str, ...], str | None]:
    for prefix in _STATE_PREFIXES:
        if keys and all(key.startswith(prefix) for key in keys):
            return tuple(key[len(prefix) :] for key in keys), prefix
    return keys, None


def _architecture_from_shapes(shapes: Mapping[str, tuple[int, ...]]) -> EchoCheckpointArchitecture:
    raw_keys = tuple(shapes)
    keys, _ = _strip_common_prefix(raw_keys)
    keys = tuple(key[len("memory_adapter.") :] if key.startswith("memory_adapter.") else key for key in keys)
    normalized_shapes = {normalized: tuple(shapes[original]) for original, normalized in zip(raw_keys, keys)}

    block_sets: dict[str, set[int]] = {name: set() for name in _BLOCK_PATTERNS}
    for key in keys:
        for name, pattern in _BLOCK_PATTERNS.items():
            match = pattern.match(key)
            if match is not None:
                block_sets[name].add(int(match.group(1)))

    spatial_shape = None
    for key in (
        "spatial_memory_module.spatial_to_tokens",
        "spatial_memory_module.mix",
    ):
        if key in normalized_shapes:
            shape = normalized_shapes[key]
            if len(shape) != 2:
                raise EchoCheckpointError(f"{key} must be rank two, got {shape}")
            spatial_shape = (int(shape[0]), int(shape[1]))
            break

    return EchoCheckpointArchitecture(
        action_blocks=tuple(sorted(block_sets["action"])),
        action_attention_blocks=tuple(sorted(block_sets["action_attention"])),
        block_ssm_blocks=tuple(sorted(block_sets["block_ssm"])),
        video_ssm_blocks=tuple(sorted(block_sets["video_ssm"])),
        spatial_grid_shape=spatial_shape,
        has_spatial_readout=any(key.startswith("spatial_memory_readout_module.") for key in keys),
    )


def inspect_echo_checkpoint(
    path: str | Path,
    recipe: EchoMemoryRecipe,
    *,
    num_blocks: int = 30,
) -> EchoCheckpointInspection:
    """Inspect a safetensors header and validate it without loading weights."""

    requested_checkpoint = Path(path).expanduser()
    checkpoint = requested_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Echo-Memory checkpoint not found: {requested_checkpoint}")
    # Hugging Face snapshots expose user-facing filenames as symlinks to
    # extensionless content-addressed blobs.  The shared checkpoint resolver may
    # already have resolved that symlink before this function is called, so an
    # extensionless path is valid here.  Still reject explicit non-safetensors
    # suffixes; ``safe_open`` below remains the authoritative format check for
    # both named files and content-addressed blobs.
    if requested_checkpoint.suffix and requested_checkpoint.suffix != ".safetensors":
        raise EchoCheckpointError(f"Echo-Memory checkpoints must use safetensors, got {requested_checkpoint.name!r}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("safetensors is required to inspect Echo-Memory checkpoints") from exc

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        shapes = {key: tuple(int(value) for value in handle.get_slice(key).get_shape()) for key in keys}
    normalized_keys, prefix = _strip_common_prefix(keys)
    architecture = _architecture_from_shapes(shapes)
    validate_echo_checkpoint_architecture(architecture, recipe, num_blocks=num_blocks)
    return EchoCheckpointInspection(
        path=str(checkpoint),
        tensor_count=len(normalized_keys),
        architecture=architecture,
        normalized_prefix=prefix,
    )


def inspect_echo_checkpoint_shapes(
    shapes: Mapping[str, tuple[int, ...]],
    recipe: EchoMemoryRecipe,
    *,
    num_blocks: int = 30,
) -> EchoCheckpointArchitecture:
    """Validate an in-memory key/shape header (useful for research tooling/tests)."""

    architecture = _architecture_from_shapes(shapes)
    validate_echo_checkpoint_architecture(architecture, recipe, num_blocks=num_blocks)
    return architecture


def normalize_echo_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite Echo checkpoint keys onto the native :class:`EchoWanModel` tree."""
    keys = tuple(state_dict)
    normalized_keys, prefix = _strip_common_prefix(keys)
    result: dict[str, Any] = {}
    for original, key in zip(keys, normalized_keys):
        if key == "spatial_memory_module.mix":
            key = "memory_adapter.spatial_memory_module.spatial_to_tokens"
        elif key.startswith("spatial_memory_module."):
            key = "memory_adapter." + key
        elif key.startswith("spatial_memory_readout_module."):
            key = "memory_adapter." + key
        if key in result:
            raise EchoCheckpointError(
                f"checkpoint normalization produced duplicate tensor {key!r} from prefix {prefix!r}"
            )
        result[key] = state_dict[original]
    return result


def load_echo_checkpoint(
    model: Any,
    path: str | Path,
    recipe: EchoMemoryRecipe,
    *,
    inspection: EchoCheckpointInspection | None = None,
) -> EchoCheckpointLoadReport:
    """Load a full Echo DiT checkpoint with exact key and shape coverage."""

    checked = inspection or inspect_echo_checkpoint(
        path,
        recipe,
        num_blocks=len(model.blocks),
    )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required to load Echo-Memory checkpoints") from exc

    raw_state = load_file(checked.path, device="cpu")
    state = normalize_echo_state_dict(raw_state)
    model_state = model.state_dict()
    missing = sorted(set(model_state).difference(state))
    unexpected = sorted(set(state).difference(model_state))
    mismatched = sorted(
        key for key in set(model_state).intersection(state) if tuple(model_state[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        shape_details = [(key, tuple(state[key].shape), tuple(model_state[key].shape)) for key in mismatched[:12]]
        detail = f"missing={missing[:12]}, unexpected={unexpected[:12]}, shape_mismatch={shape_details}"
        raise EchoCheckpointError(f"Echo checkpoint does not exactly match native Wan model: {detail}")
    model.load_state_dict(state, strict=True)
    return EchoCheckpointLoadReport(
        path=checked.path,
        tensor_count=len(state),
        model_tensor_count=len(model_state),
        architecture=checked.architecture,
    )


__all__ = [
    "EchoCheckpointArchitecture",
    "EchoCheckpointError",
    "EchoCheckpointInspection",
    "EchoCheckpointLoadReport",
    "inspect_echo_checkpoint",
    "inspect_echo_checkpoint_shapes",
    "load_echo_checkpoint",
    "normalize_echo_state_dict",
    "validate_echo_checkpoint_architecture",
]
