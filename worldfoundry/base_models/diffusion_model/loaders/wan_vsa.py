"""Strict checkpoint plumbing for FastVideo-style Wan VSA-QAT gates.

VSA is not a drop-in sparse kernel for an ordinary Wan checkpoint.  The
published FastVideo model adds one learned ``dim -> dim`` compression gate to
every transformer block.  This module detects that architecture from tensor
headers before model construction and canonicalizes the few released key
layouts without ever inventing randomly initialized gates for dense weights.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .materialize import MaterializedCheckpoint

_GATE_NAMES = frozenset(("gate_compress", "to_gate_compress"))
_GATE_OWNERS = frozenset(("attn1", "self_attn"))
_GATE_PARAMETERS = frozenset(("bias", "weight"))


def canonical_wan_vsa_gate_key(name: str) -> str | None:
    """Return the native gate key, or ``None`` for an unrelated parameter.

    Released checkpoints have used a block-level ``to_gate_compress`` module,
    while Diffusers-style variants may nest it below ``attn1``.  WorldFoundry
    owns the projection on ``self_attn`` because the sparse processor is
    installed there.  Any key that mentions a gate but does not match one of
    these exact layouts is rejected instead of being silently ignored.
    """

    if "gate_compress" not in name:
        return None
    parts = name.split(".")
    if len(parts) == 4:
        root, block, gate_name, parameter = parts
        owner = None
    elif len(parts) == 5:
        root, block, owner, gate_name, parameter = parts
    else:
        raise KeyError(f"unsupported Wan VSA gate parameter: {name}")
    if (
        root != "blocks"
        or not block.isdigit()
        or gate_name not in _GATE_NAMES
        or parameter not in _GATE_PARAMETERS
        or (owner is not None and owner not in _GATE_OWNERS)
    ):
        raise KeyError(f"unsupported Wan VSA gate parameter: {name}")
    return f"blocks.{int(block)}.self_attn.gate_compress.{parameter}"


def convert_wan_vsa_gate_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Canonicalize VSA gate aliases and reject duplicate target parameters."""

    converted: dict[str, object] = {}
    sources: dict[str, str] = {}
    for source, value in state_dict.items():
        target = canonical_wan_vsa_gate_key(source) or source
        if target in converted:
            previous = sources[target]
            raise KeyError(
                "Wan VSA gate conversion produced duplicate parameter "
                f"{target!r} from {previous!r} and {source!r}"
            )
        converted[target] = value
        sources[target] = source
    return converted


def _tensor_shape(name: str, value: object) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(size) for size in value.shape)
    if isinstance(value, (tuple, list)) and all(
        isinstance(size, int) and not isinstance(size, bool) for size in value
    ):
        return tuple(int(size) for size in value)
    get_shape = getattr(value, "get_shape", None)
    if callable(get_shape):
        return tuple(int(size) for size in get_shape())
    raise TypeError(
        f"Wan VSA gate parameter {name!r} has no inspectable tensor shape"
    )


def validate_wan_vsa_gate_shapes(
    state_dict: Mapping[str, object],
    *,
    dim: int,
    num_layers: int,
) -> bool:
    """Validate the all-or-nothing per-layer VSA-QAT gate contract.

    Returns ``False`` only when the checkpoint has no gate parameters at all.
    Once any gate key is present, every layer must contain exactly one weight
    and bias with the canonical shapes.  Partial, extra, malformed, or aliased
    duplicate gates fail before model allocation.
    """

    if isinstance(dim, bool) or int(dim) <= 0:
        raise ValueError(f"Wan hidden dimension must be positive, got {dim!r}")
    if isinstance(num_layers, bool) or int(num_layers) <= 0:
        raise ValueError(f"Wan layer count must be positive, got {num_layers!r}")
    dim = int(dim)
    num_layers = int(num_layers)

    gate_values: dict[str, object] = {}
    gate_sources: dict[str, str] = {}
    for source, value in state_dict.items():
        target = canonical_wan_vsa_gate_key(source)
        if target is None:
            continue
        if target in gate_values:
            raise KeyError(
                "Wan VSA gate checkpoint maps multiple parameters to "
                f"{target!r}: {gate_sources[target]!r}, {source!r}"
            )
        gate_values[target] = value
        gate_sources[target] = source
    if not gate_values:
        return False

    expected = {
        f"blocks.{layer}.self_attn.gate_compress.{parameter}"
        for layer in range(num_layers)
        for parameter in ("weight", "bias")
    }
    actual = set(gate_values)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Wan VSA-QAT gate checkpoint must contain weight and bias for "
            f"every layer; missing={missing}, unexpected={unexpected}"
        )

    mismatches: list[str] = []
    for name, value in gate_values.items():
        expected_shape = (dim, dim) if name.endswith(".weight") else (dim,)
        actual_shape = _tensor_shape(gate_sources[name], value)
        if actual_shape != expected_shape:
            mismatches.append(
                f"{gate_sources[name]}: expected {expected_shape}, got {actual_shape}"
            )
    if mismatches:
        raise ValueError(
            "Wan VSA-QAT gate tensor shape mismatch: " + "; ".join(mismatches)
        )
    return True


def _indexed_safetensor_paths(path: Path) -> tuple[Path, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors index: {path}") from error
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"safetensors index has no mapping-valued weight_map: {path}")
    names = tuple(dict.fromkeys(str(value) for value in weight_map.values()))
    if not names:
        raise ValueError(f"safetensors index contains no shard names: {path}")
    shards: list[Path] = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe safetensors shard path {name!r} in {path}")
        shard = path.parent / relative
        if not shard.is_file():
            raise FileNotFoundError(f"safetensors index shard does not exist: {shard}")
        shards.append(shard)
    return tuple(shards)


def _checkpoint_paths(checkpoint: MaterializedCheckpoint) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in checkpoint.paths:
        if path.is_dir():
            indexes = sorted(path.glob("*.safetensors.index.json"))
            if indexes:
                for index in indexes:
                    paths.extend(_indexed_safetensor_paths(index))
            else:
                paths.extend(sorted(path.glob("*.safetensors")))
                paths.extend(sorted(path.glob("*.bin")))
                paths.extend(sorted(path.glob("*.pt")))
                paths.extend(sorted(path.glob("*.pth")))
        elif path.name.endswith(".safetensors.index.json"):
            paths.extend(_indexed_safetensor_paths(path))
        else:
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _checkpoint_gate_shapes(
    checkpoint: MaterializedCheckpoint,
) -> Mapping[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for path in _checkpoint_paths(checkpoint):
        if path.suffix == ".safetensors":
            try:
                from safetensors import safe_open
            except ImportError as error:
                raise RuntimeError(
                    "Wan VSA checkpoint detection requires the safetensors package"
                ) from error
            with safe_open(path, framework="pt", device="cpu") as handle:
                entries: dict[str, Any] = {
                    name: handle.get_slice(name).get_shape()
                    for name in handle.keys()
                    if "gate_compress" in name
                }
        else:
            # PyTorch pickle formats have no standalone tensor header.  Reuse
            # the safe weights-only loader and immediately discard all
            # non-gate entries.  Public VSA-QAT releases are safetensors, so
            # this compatibility path does not penalize their startup.
            from worldfoundry.core.model_loading import load_keys_dict

            loaded = load_keys_dict(str(path))
            entries = {
                name: value
                for name, value in loaded.items()
                if "gate_compress" in name
            }
        for name, value in entries.items():
            if name in shapes:
                raise ValueError(
                    f"Wan VSA gate tensor {name!r} occurs in multiple checkpoint shards"
                )
            shapes[name] = _tensor_shape(name, value)
    return shapes


def resolve_wan_vsa_gate_config(
    checkpoint: MaterializedCheckpoint,
    *,
    dim: int,
    num_layers: int,
) -> Mapping[str, object]:
    """Return the conditional constructor flag from checkpoint tensor headers."""

    gate_shapes = _checkpoint_gate_shapes(checkpoint)
    enabled = validate_wan_vsa_gate_shapes(
        gate_shapes,
        dim=dim,
        num_layers=num_layers,
    )
    return {"vsa_gate_compress": True} if enabled else {}


__all__ = [
    "canonical_wan_vsa_gate_key",
    "convert_wan_vsa_gate_state_dict",
    "resolve_wan_vsa_gate_config",
    "validate_wan_vsa_gate_shapes",
]
