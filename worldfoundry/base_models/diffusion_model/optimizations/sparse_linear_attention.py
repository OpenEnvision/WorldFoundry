"""Pinned, fail-closed adapters for FastVideo SLA and SageSLA attention.

FastVideo's SLA backend is a learned Sparse-Linear Attention implementation:
the block-sparse result is combined with a linear-attention branch projected by
one checkpoint-trained ``proj_l`` per Wan block.  It is deliberately separate
from LightX2V's similarly named SLA *mask* composition.

This module owns only the boundary contract needed by a future Wan processor:

* resolve every layer's learned ``proj_l`` from an audited checkpoint layout;
* construct the exact pinned FastVideo implementation class;
* validate BSHD tensors and provider output without a dense fallback; and
* return tensor-free, JSON-safe provider/weight/runtime receipts.

Provider source identity is resolved once during adapter construction and held
in a process cache.  No git command, source hashing, or checkpoint hashing is
performed in :meth:`FastVideoSLAAdapter.forward`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import operator
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

PINNED_FASTVIDEO_COMMIT = "1b2b2a0161bc6b3b80158d1fa6380a051c6530c7"

_UPSTREAM_MODULE = "fastvideo.attention.backends.sla"
_REFERENCE_PROVIDER_PATHS = {
    "fastvideo_sla": f"{_UPSTREAM_MODULE}.SLAAttentionImpl",
    "fastvideo_sagesla": f"{_UPSTREAM_MODULE}.SageSLAAttentionImpl",
}
_PROVIDER_CLASS_NAMES = {
    "fastvideo_sla": "SLAAttentionImpl",
    "fastvideo_sagesla": "SageSLAAttentionImpl",
}
_CHECKPOINT_PATTERNS = (
    (
        "fastvideo_diffusers",
        re.compile(
            r"^blocks\.(?P<layer>\d+)\.attn1\.attn_impl\.proj_l\."
            r"(?P<parameter>weight|bias)$"
        ),
    ),
    (
        "turbodiffusion_original",
        re.compile(
            r"^blocks\.(?P<layer>\d+)\.self_attn\.attn_op\.local_attn\."
            r"proj_l\.(?P<parameter>weight|bias)$"
        ),
    ),
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_IDENTITY_LOCK = RLock()


class FastVideoSLAUnavailableError(RuntimeError):
    """The pinned FastVideo provider cannot be proven or constructed."""


class FastVideoSLAContractError(RuntimeError):
    """Checkpoint, tensor, or provider output violated the SLA contract."""


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        ) from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer, got {normalized}")
    return normalized


def _layer_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"layer_index must be a non-negative integer, got {value!r}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"layer_index must be a non-negative integer, got {value!r}"
        ) from exc
    if normalized < 0:
        raise ValueError(
            f"layer_index must be a non-negative integer, got {normalized}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class FastVideoSLAConfig:
    """Exact FastVideo learned-SLA selection, never a LightX2V mask alias."""

    kind: str
    topk_ratio: float | None = None
    feature_map: str = "softmax"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().casefold().replace("-", "_")
        if kind not in _REFERENCE_PROVIDER_PATHS:
            raise ValueError(
                "FastVideo learned SLA kind must be 'fastvideo_sla' or "
                f"'fastvideo_sagesla', got {self.kind!r}; LightX2V SLA mask "
                "kinds belong to sparse_mask_attention.py"
            )
        object.__setattr__(self, "kind", kind)
        ratio = self.topk_ratio
        if ratio is None:
            ratio = 0.1 if kind == "fastvideo_sla" else 0.5
        ratio = float(ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"topk_ratio must be finite and in (0, 1], got {ratio!r}"
            )
        object.__setattr__(self, "topk_ratio", ratio)
        feature_map = str(self.feature_map).strip().casefold()
        if feature_map not in {"softmax", "elu", "relu"}:
            raise ValueError(
                "feature_map must be one of ['elu', 'relu', 'softmax'], got "
                f"{self.feature_map!r}"
            )
        object.__setattr__(self, "feature_map", feature_map)


@dataclass(frozen=True, slots=True)
class FastVideoSLALayerSpec:
    """Wan layer geometry required to construct one FastVideo implementation."""

    layer_index: int
    num_heads: int
    head_size: int
    softmax_scale: float | None = None
    prefix: str | None = None

    def __post_init__(self) -> None:
        layer_index = _layer_index(self.layer_index)
        num_heads = _positive_int("num_heads", self.num_heads)
        head_size = _positive_int("head_size", self.head_size)
        if head_size not in {64, 128}:
            raise ValueError(
                "FastVideo SLA/SageSLA support head_size 64 or 128, got "
                f"{head_size}"
            )
        scale = self.softmax_scale
        if scale is not None:
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "softmax_scale must be positive and finite, got "
                    f"{scale!r}"
                )
        prefix = self.prefix
        if prefix is None:
            prefix = f"blocks.{layer_index}.attn1"
        prefix = str(prefix).strip()
        if not prefix:
            raise ValueError("layer prefix cannot be empty")
        object.__setattr__(self, "layer_index", layer_index)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "head_size", head_size)
        object.__setattr__(self, "softmax_scale", scale)
        object.__setattr__(self, "prefix", prefix)


@dataclass(frozen=True, slots=True)
class FastVideoSLASourceIdentity:
    """Pinned FastVideo runtime-source evidence captured outside the hot path."""

    commit: str | None
    clean: bool | None
    fingerprint: str | None
    root: str | None
    source_file: str | None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> FastVideoSLASourceIdentity:
        return cls(
            commit=(
                str(value["provider_source_commit"])
                if isinstance(value.get("provider_source_commit"), str)
                else None
            ),
            clean=(
                value.get("provider_source_clean")
                if isinstance(value.get("provider_source_clean"), bool)
                else None
            ),
            fingerprint=(
                str(value["provider_source_fingerprint"])
                if isinstance(value.get("provider_source_fingerprint"), str)
                else None
            ),
            root=(
                str(value["provider_source_root"])
                if isinstance(value.get("provider_source_root"), str)
                else None
            ),
            source_file=(
                str(value["provider_source_file"])
                if isinstance(value.get("provider_source_file"), str)
                else None
            ),
        )

    def to_receipt(self) -> dict[str, Any]:
        return {
            "provider_source_commit": self.commit,
            "provider_source_clean": self.clean,
            "provider_source_fingerprint": self.fingerprint,
            "provider_source_root": self.root,
            "provider_source_file": self.source_file,
        }


@dataclass(frozen=True, slots=True)
class FastVideoSLAProjectionWeights:
    """Normalized float32 projection tensors for exactly one Wan layer."""

    layer_index: int
    weight: torch.Tensor
    bias: torch.Tensor
    source_keys: tuple[str, str]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class FastVideoSLAResolvedCheckpoint:
    """Complete, zero-based layer projection checkpoint."""

    layout: str
    projections: Mapping[int, FastVideoSLAProjectionWeights]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class FastVideoSLAMetadata:
    """Strict injected-provider metadata seam matching FastVideo attributes."""

    current_timestep: int
    topk_ratio: float


@dataclass(frozen=True, slots=True)
class FastVideoSLAResult:
    """One provider output plus tensor-free runtime evidence."""

    output: torch.Tensor
    receipt: Mapping[str, Any]


def _normalize_layer_specs(
    layer_specs: Sequence[FastVideoSLALayerSpec],
) -> dict[int, FastVideoSLALayerSpec]:
    if isinstance(layer_specs, (str, bytes)):
        raise TypeError("layer_specs must be a sequence of FastVideoSLALayerSpec")
    normalized: dict[int, FastVideoSLALayerSpec] = {}
    for spec in layer_specs:
        if not isinstance(spec, FastVideoSLALayerSpec):
            raise TypeError(
                "layer_specs must contain only FastVideoSLALayerSpec values"
            )
        if spec.layer_index in normalized:
            raise FastVideoSLAContractError(
                f"duplicate FastVideo SLA layer spec for layer {spec.layer_index}"
            )
        normalized[spec.layer_index] = spec
    if not normalized:
        raise FastVideoSLAContractError(
            "FastVideo SLA requires at least one Wan self-attention layer"
        )
    expected = list(range(len(normalized)))
    actual = sorted(normalized)
    if actual != expected:
        raise FastVideoSLAContractError(
            "FastVideo SLA layer specs must cover every zero-based Wan block: "
            f"expected={expected}, actual={actual}"
        )
    return {index: normalized[index] for index in actual}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    normalized = tensor.detach().to(device="cpu").contiguous()
    return normalized.view(torch.uint8).numpy().tobytes()


def _projection_fingerprint(
    layer_index: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    for name, tensor in (("weight", weight), ("bias", bias)):
        digest.update(f"{layer_index}:{name}".encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_fastvideo_sla_projection_weights(
    checkpoint_state_dict: Mapping[str, Any],
    layer_specs: Sequence[FastVideoSLALayerSpec],
) -> FastVideoSLAResolvedCheckpoint:
    """Resolve complete Turbo-original or FastVideo/Diffusers ``proj_l`` keys.

    The two audited layouts are:

    * ``blocks.<i>.attn1.attn_impl.proj_l.{weight,bias}``
    * ``blocks.<i>.self_attn.attn_op.local_attn.proj_l.{weight,bias}``

    Aliases may not be mixed.  Any unsupported ``proj_l`` key is rejected so a
    conversion typo cannot silently leave an upstream zero-initialized module.
    """

    if not isinstance(checkpoint_state_dict, Mapping):
        raise TypeError("checkpoint_state_dict must be a mapping")
    specs = _normalize_layer_specs(layer_specs)
    resolved: dict[tuple[int, str], tuple[str, str, Any]] = {}
    duplicate_roles: list[str] = []
    unsupported_keys: list[str] = []
    layouts: set[str] = set()
    for raw_key, value in checkpoint_state_dict.items():
        if not isinstance(raw_key, str):
            raise FastVideoSLAContractError(
                "FastVideo SLA checkpoint keys must be strings, got "
                f"{type(raw_key).__name__}: {raw_key!r}"
            )
        match_record: tuple[str, re.Match[str]] | None = None
        for layout, pattern in _CHECKPOINT_PATTERNS:
            match = pattern.fullmatch(raw_key)
            if match is not None:
                match_record = (layout, match)
                break
        if match_record is None:
            if "proj_l" in raw_key:
                unsupported_keys.append(raw_key)
            continue
        layout, match = match_record
        layer_index = int(match.group("layer"))
        parameter = match.group("parameter")
        role = (layer_index, parameter)
        if role in resolved:
            duplicate_roles.append(
                f"layer={layer_index} parameter={parameter}: "
                f"{resolved[role][1]!r} and {raw_key!r}"
            )
            continue
        resolved[role] = (layout, raw_key, value)
        layouts.add(layout)
    if unsupported_keys:
        raise FastVideoSLAContractError(
            "unsupported FastVideo SLA proj_l checkpoint keys: "
            f"{sorted(unsupported_keys)}"
        )
    if duplicate_roles:
        raise FastVideoSLAContractError(
            "duplicate FastVideo SLA checkpoint mappings: "
            + "; ".join(sorted(duplicate_roles))
        )
    if len(layouts) > 1:
        raise FastVideoSLAContractError(
            "FastVideo SLA checkpoint may not mix Turbo-original and "
            f"FastVideo/Diffusers proj_l layouts: {sorted(layouts)}"
        )
    unexpected_layers = sorted({layer for layer, _ in resolved} - set(specs))
    if unexpected_layers:
        raise FastVideoSLAContractError(
            "FastVideo SLA checkpoint contains unexpected projection layers: "
            f"{unexpected_layers}"
        )
    missing = [
        f"layer={layer_index} parameter={parameter}"
        for layer_index in specs
        for parameter in ("weight", "bias")
        if (layer_index, parameter) not in resolved
    ]
    if missing:
        raise FastVideoSLAContractError(
            "FastVideo SLA requires learned proj_l weight and bias for every "
            f"Wan block; missing={missing}"
        )
    projections: dict[int, FastVideoSLAProjectionWeights] = {}
    for layer_index, spec in specs.items():
        tensors: dict[str, torch.Tensor] = {}
        source_keys: dict[str, str] = {}
        for parameter, expected_shape in (
            ("weight", (spec.head_size, spec.head_size)),
            ("bias", (spec.head_size,)),
        ):
            _, source_key, raw_tensor = resolved[(layer_index, parameter)]
            if not isinstance(raw_tensor, torch.Tensor):
                raise FastVideoSLAContractError(
                    f"{source_key!r} must be a Tensor, got "
                    f"{type(raw_tensor).__name__}"
                )
            if raw_tensor.layout is not torch.strided:
                raise FastVideoSLAContractError(
                    f"{source_key!r} must be a strided dense tensor"
                )
            if not raw_tensor.dtype.is_floating_point:
                raise FastVideoSLAContractError(
                    f"{source_key!r} must use a floating dtype, got "
                    f"{raw_tensor.dtype}"
                )
            if tuple(raw_tensor.shape) != expected_shape:
                raise FastVideoSLAContractError(
                    f"{source_key!r} has shape {tuple(raw_tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            normalized = raw_tensor.detach().to(
                device="cpu",
                dtype=torch.float32,
            ).contiguous().clone()
            if not bool(torch.isfinite(normalized).all().item()):
                raise FastVideoSLAContractError(
                    f"{source_key!r} contains non-finite learned weights"
                )
            tensors[parameter] = normalized
            source_keys[parameter] = source_key
        if not bool(torch.count_nonzero(tensors["weight"]).item()):
            raise FastVideoSLAContractError(
                f"layer {layer_index} proj_l.weight is all zero; refusing "
                "FastVideo's untrained default initialization"
            )
        fingerprint = _projection_fingerprint(
            layer_index,
            tensors["weight"],
            tensors["bias"],
        )
        projections[layer_index] = FastVideoSLAProjectionWeights(
            layer_index=layer_index,
            weight=tensors["weight"],
            bias=tensors["bias"],
            source_keys=(source_keys["weight"], source_keys["bias"]),
            fingerprint=fingerprint,
        )
    layout = next(iter(layouts), "unknown")
    global_digest = hashlib.sha256()
    global_digest.update(layout.encode("ascii"))
    global_digest.update(b"\n")
    for layer_index, projection in projections.items():
        global_digest.update(str(layer_index).encode("ascii"))
        global_digest.update(b":")
        global_digest.update(projection.fingerprint.encode("ascii"))
        global_digest.update(b"\n")
    return FastVideoSLAResolvedCheckpoint(
        layout=layout,
        projections=MappingProxyType(projections),
        fingerprint=global_digest.hexdigest(),
    )


def split_fastvideo_sla_projection_weights(
    checkpoint_state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate recognized learned-SLA tensors from ordinary model weights.

    FastVideo publishes ``proj_l`` inside the transformer checkpoint even
    though the canonical Wan graph does not own that layer.  The dense state
    must therefore be restored without those auxiliary tensors, while the
    exact original keys remain available to :class:`FastVideoSLAAdapter`.

    Unknown keys containing ``proj_l`` are rejected instead of being silently
    discarded.  That keeps a renamed or partially converted SLA checkpoint
    from loading as an ordinary dense Wan model.
    """

    if not isinstance(checkpoint_state_dict, Mapping):
        raise TypeError("checkpoint_state_dict must be a mapping")
    model_state: dict[str, Any] = {}
    projection_state: dict[str, Any] = {}
    unsupported: list[str] = []
    for raw_key, value in checkpoint_state_dict.items():
        if not isinstance(raw_key, str):
            raise FastVideoSLAContractError(
                "FastVideo SLA checkpoint keys must be strings, got "
                f"{type(raw_key).__name__}: {raw_key!r}"
            )
        recognized = any(pattern.fullmatch(raw_key) for _, pattern in _CHECKPOINT_PATTERNS)
        if recognized:
            projection_state[raw_key] = value
        elif "proj_l" in raw_key:
            unsupported.append(raw_key)
        else:
            model_state[raw_key] = value
    if unsupported:
        raise FastVideoSLAContractError(
            "unsupported FastVideo SLA proj_l checkpoint keys: "
            f"{sorted(unsupported)}"
        )
    return model_state, projection_state


_SOURCE_IDENTITIES: dict[str, FastVideoSLASourceIdentity] = {}


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _unknown_source_identity(
    *,
    source_file: Path | None,
    root: Path | None = None,
) -> FastVideoSLASourceIdentity:
    source_hash = _file_sha256(source_file) if source_file is not None else None
    return FastVideoSLASourceIdentity(
        commit=None,
        clean=None,
        fingerprint=source_hash,
        root=str(root) if root is not None else None,
        source_file=str(source_file) if source_file is not None else None,
    )


def _provider_source_identity(module: Any) -> FastVideoSLASourceIdentity:
    """Resolve one FastVideo checkout identity and cache it off the hot path."""

    source_value = getattr(module, "__file__", None)
    if not isinstance(source_value, (str, bytes)):
        return _unknown_source_identity(source_file=None)
    source_path = Path(source_value).resolve()
    cache_key = str(source_path)
    with _SOURCE_IDENTITY_LOCK:
        cached = _SOURCE_IDENTITIES.get(cache_key)
        if cached is not None:
            return cached
        checkout_root = next(
            (
                parent
                for parent in (source_path.parent, *source_path.parents)
                if (parent / ".git").exists()
            ),
            None,
        )
        if checkout_root is None:
            identity = _unknown_source_identity(source_file=source_path)
            _SOURCE_IDENTITIES[cache_key] = identity
            return identity
        scoped_paths = [
            name
            for name in ("fastvideo", "fastvideo-kernel")
            if (checkout_root / name).exists()
        ]
        if not scoped_paths:
            try:
                scoped_paths = [source_path.relative_to(checkout_root).as_posix()]
            except ValueError:
                scoped_paths = []
        try:
            revision = subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout_root),
                    "rev-parse",
                    "--verify",
                    "HEAD",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout_root),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    *scoped_paths,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            identity = _unknown_source_identity(
                source_file=source_path,
                root=checkout_root,
            )
            _SOURCE_IDENTITIES[cache_key] = identity
            return identity
        commit = revision.stdout.strip().casefold()
        if revision.returncode != 0 or _HEX_COMMIT.fullmatch(commit) is None:
            commit = None
        clean = status.returncode == 0 and not status.stdout.strip()
        source_files = [source_path]
        kernel_source = (
            checkout_root
            / "fastvideo-kernel"
            / "python"
            / "fastvideo_kernel"
            / "triton_kernels"
            / "sla_triton.py"
        )
        if kernel_source.is_file():
            source_files.append(kernel_source)
        digest = hashlib.sha256()
        digest.update(f"{commit}\0{clean}\0{status.stdout}".encode())
        for path in sorted(source_files, key=str):
            file_hash = _file_sha256(path)
            digest.update(str(path.relative_to(checkout_root)).encode())
            digest.update(b"\0")
            digest.update(str(file_hash).encode("ascii"))
            digest.update(b"\n")
        identity = FastVideoSLASourceIdentity(
            commit=commit,
            clean=clean,
            fingerprint=digest.hexdigest(),
            root=str(checkout_root),
            source_file=str(source_path),
        )
        _SOURCE_IDENTITIES[cache_key] = identity
        return identity


def _require_pinned_clean_source(identity: FastVideoSLASourceIdentity) -> None:
    issues: list[str] = []
    if identity.commit != PINNED_FASTVIDEO_COMMIT:
        issues.append(
            f"commit={identity.commit!r}, expected {PINNED_FASTVIDEO_COMMIT!r}"
        )
    if identity.clean is not True:
        issues.append(f"clean={identity.clean!r}, expected True")
    if (
        not isinstance(identity.fingerprint, str)
        or _HEX_SHA256.fullmatch(identity.fingerprint) is None
    ):
        issues.append("source fingerprint is missing or malformed")
    if issues:
        raise FastVideoSLAUnavailableError(
            "FastVideo SLA requires the pinned clean upstream source: "
            + "; ".join(issues)
        )


@dataclass(frozen=True, slots=True)
class _UpstreamBundle:
    provider_type: type[nn.Module]
    metadata_type: type[Any]
    source_identity: FastVideoSLASourceIdentity


def _load_upstream_bundle(kind: str) -> _UpstreamBundle:
    try:
        module = importlib.import_module(_UPSTREAM_MODULE)
    except Exception as exc:
        raise FastVideoSLAUnavailableError(
            f"cannot import pinned FastVideo SLA provider module: {exc}"
        ) from exc
    class_name = _PROVIDER_CLASS_NAMES[kind]
    provider_type = getattr(module, class_name, None)
    metadata_type = getattr(module, "SLAAttentionMetadata", None)
    if not isinstance(provider_type, type) or not issubclass(provider_type, nn.Module):
        raise FastVideoSLAUnavailableError(
            f"{_UPSTREAM_MODULE}.{class_name} is missing or is not an nn.Module"
        )
    provider_path = f"{provider_type.__module__}.{provider_type.__name__}"
    if provider_path != _REFERENCE_PROVIDER_PATHS[kind]:
        raise FastVideoSLAUnavailableError(
            "FastVideo SLA provider identity mismatch: "
            f"got {provider_path!r}, expected {_REFERENCE_PROVIDER_PATHS[kind]!r}"
        )
    if not isinstance(metadata_type, type):
        raise FastVideoSLAUnavailableError(
            f"{_UPSTREAM_MODULE}.SLAAttentionMetadata is missing"
        )
    source_identity = _provider_source_identity(module)
    _require_pinned_clean_source(source_identity)
    return _UpstreamBundle(
        provider_type=provider_type,
        metadata_type=metadata_type,
        source_identity=source_identity,
    )


def fastvideo_sla_runtime_expectation(
    config_or_kind: FastVideoSLAConfig | str,
) -> dict[str, Any]:
    """Resolve canonical pinned-provider evidence before a benchmark run.

    The benchmark keeps this trusted expectation separate from the later
    request receipt.  Resolution imports the exact upstream module and rejects
    a different commit, a dirty provider tree, an unexpected class, or a
    malformed source fingerprint before CUDA execution begins.
    """

    config = (
        config_or_kind
        if isinstance(config_or_kind, FastVideoSLAConfig)
        else FastVideoSLAConfig(kind=config_or_kind)
    )
    bundle = _load_upstream_bundle(config.kind)
    family = (
        "fastvideo/sparse-linear-attention"
        if config.kind == "fastvideo_sla"
        else "fastvideo/sage-sparse-linear-attention"
    )
    return {
        "kind": config.kind,
        "provider_path": _REFERENCE_PROVIDER_PATHS[config.kind],
        "provider_family": family,
        "commit": PINNED_FASTVIDEO_COMMIT,
        "source_fingerprint": bundle.source_identity.fingerprint,
        "source_root": bundle.source_identity.root,
        "source_file": bundle.source_identity.source_file,
    }


ProviderFactory = Callable[
    [FastVideoSLALayerSpec, FastVideoSLAConfig],
    nn.Module,
]
MetadataFactory = Callable[[int, float], Any]


def _tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "device": str(value.device),
        "dtype": str(value.dtype),
        "contiguous": value.is_contiguous(),
    }


def _class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


class FastVideoSLAAdapter(nn.Module):
    """Call one immutable pinned FastVideo SLA provider per Wan block.

    Production construction imports the exact upstream implementation.  Tests
    may inject a provider factory only while explicitly enabling the CPU test
    seam and supplying a still-pinned source identity.  Receipts always expose
    that injection and therefore never claim reference-class parity for fakes.

    Future Wan wiring should register this adapter on the denoiser model, move
    it to the execution device without changing its float32 ``proj_l`` dtype,
    then call ``adapter(layer_index, q, k, v, current_timestep=step)`` from the
    self-attention processor.
    """

    def __init__(
        self,
        config: FastVideoSLAConfig,
        *,
        layer_specs: Sequence[FastVideoSLALayerSpec],
        checkpoint_state_dict: Mapping[str, Any],
        provider_factory: ProviderFactory | None = None,
        metadata_factory: MetadataFactory | None = None,
        source_identity: FastVideoSLASourceIdentity | Mapping[str, Any] | None = None,
        allow_non_cuda_for_tests: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(config, FastVideoSLAConfig):
            raise TypeError("config must be a FastVideoSLAConfig")
        self.config = config
        self._layer_specs = _normalize_layer_specs(layer_specs)
        self._checkpoint = resolve_fastvideo_sla_projection_weights(
            checkpoint_state_dict,
            tuple(self._layer_specs.values()),
        )
        self._allow_non_cuda_for_tests = bool(allow_non_cuda_for_tests)
        self._injected_test_provider = provider_factory is not None
        if provider_factory is None:
            if metadata_factory is not None or source_identity is not None:
                raise ValueError(
                    "metadata/source injection requires an injected provider_factory"
                )
            bundle = _load_upstream_bundle(config.kind)
            identity = bundle.source_identity

            def production_factory(
                spec: FastVideoSLALayerSpec,
                selected: FastVideoSLAConfig,
            ) -> nn.Module:
                kwargs: dict[str, Any] = {
                    "num_heads": spec.num_heads,
                    "head_size": spec.head_size,
                    "causal": False,
                    "softmax_scale": spec.softmax_scale,
                    "num_kv_heads": spec.num_heads,
                    "prefix": spec.prefix,
                    "topk_ratio": selected.topk_ratio,
                    "feature_map": selected.feature_map,
                    "use_bf16": True,
                }
                if selected.kind == "fastvideo_sla":
                    kwargs.update({"BLKQ": 128, "BLKK": 64})
                try:
                    return bundle.provider_type(**kwargs)
                except Exception as exc:
                    raise FastVideoSLAUnavailableError(
                        f"FastVideo provider construction failed for layer "
                        f"{spec.layer_index}: {type(exc).__name__}: {exc}"
                    ) from exc

            selected_factory = production_factory
            def production_metadata_factory(timestep: int, ratio: float) -> Any:
                return bundle.metadata_type(
                    current_timestep=timestep,
                    topk_ratio=ratio,
                )

            selected_metadata_factory: MetadataFactory = production_metadata_factory
        else:
            if not self._allow_non_cuda_for_tests:
                raise ValueError(
                    "provider_factory is a test-only seam and requires "
                    "allow_non_cuda_for_tests=True"
                )
            if source_identity is None:
                raise ValueError(
                    "an injected provider requires explicit pinned source_identity"
                )
            identity = (
                source_identity
                if isinstance(source_identity, FastVideoSLASourceIdentity)
                else FastVideoSLASourceIdentity.from_mapping(source_identity)
                if isinstance(source_identity, Mapping)
                else None
            )
            if identity is None:
                raise TypeError(
                    "source_identity must be FastVideoSLASourceIdentity or a mapping"
                )
            _require_pinned_clean_source(identity)
            selected_factory = provider_factory
            def injected_metadata_factory(timestep: int, ratio: float) -> Any:
                return FastVideoSLAMetadata(
                    current_timestep=timestep,
                    topk_ratio=ratio,
                )

            selected_metadata_factory = metadata_factory or injected_metadata_factory
        self._source_identity = identity
        self._metadata_factory = selected_metadata_factory
        self.providers = nn.ModuleDict()
        self._provider_paths: dict[int, str] = {}
        self._layer_weight_fingerprints: dict[int, str] = {}
        provider_ids: set[int] = set()
        projection_ids: set[int] = set()
        parameter_storage: set[tuple[int, int]] = set()
        for layer_index, spec in self._layer_specs.items():
            try:
                provider = selected_factory(spec, config)
            except FastVideoSLAUnavailableError:
                raise
            except Exception as exc:
                raise FastVideoSLAUnavailableError(
                    f"injected FastVideo SLA provider construction failed for "
                    f"layer {layer_index}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(provider, nn.Module):
                raise FastVideoSLAContractError(
                    f"layer {layer_index} provider must be an nn.Module, got "
                    f"{type(provider).__name__}"
                )
            if id(provider) in provider_ids:
                raise FastVideoSLAContractError(
                    "FastVideo SLA provider_factory reused one module across layers"
                )
            provider_ids.add(id(provider))
            projection = getattr(provider, "proj_l", None)
            if not isinstance(projection, nn.Linear):
                raise FastVideoSLAContractError(
                    f"layer {layer_index} provider has no nn.Linear proj_l"
                )
            if id(projection) in projection_ids:
                raise FastVideoSLAContractError(
                    "FastVideo SLA providers share one proj_l module across layers"
                )
            projection_ids.add(id(projection))
            expected_weight_shape = (spec.head_size, spec.head_size)
            expected_bias_shape = (spec.head_size,)
            if (
                tuple(projection.weight.shape) != expected_weight_shape
                or projection.bias is None
                or tuple(projection.bias.shape) != expected_bias_shape
            ):
                raise FastVideoSLAContractError(
                    f"layer {layer_index} provider proj_l geometry is malformed"
                )
            learned = self._checkpoint.projections[layer_index]
            with torch.no_grad():
                projection.weight.copy_(
                    learned.weight.to(
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    )
                )
                projection.bias.copy_(
                    learned.bias.to(
                        device=projection.bias.device,
                        dtype=projection.bias.dtype,
                    )
                )
            if projection.weight.dtype is not torch.float32:
                raise FastVideoSLAContractError(
                    f"layer {layer_index} provider proj_l must remain float32, got "
                    f"{projection.weight.dtype}"
                )
            actual_fingerprint = _projection_fingerprint(
                layer_index,
                projection.weight,
                projection.bias,
            )
            if actual_fingerprint != learned.fingerprint:
                raise FastVideoSLAContractError(
                    f"layer {layer_index} learned proj_l changed during provider load"
                )
            storage_key = (
                projection.weight.untyped_storage().data_ptr(),
                projection.bias.untyped_storage().data_ptr(),
            )
            if storage_key in parameter_storage:
                raise FastVideoSLAContractError(
                    "FastVideo SLA providers share projection parameter storage"
                )
            parameter_storage.add(storage_key)
            provider.requires_grad_(False)
            provider.eval()
            self.providers[str(layer_index)] = provider
            self._provider_paths[layer_index] = _class_path(provider)
            self._layer_weight_fingerprints[layer_index] = actual_fingerprint
        if not self._injected_test_provider:
            expected_path = _REFERENCE_PROVIDER_PATHS[config.kind]
            wrong_paths = sorted(
                {
                    path
                    for path in self._provider_paths.values()
                    if path != expected_path
                }
            )
            if wrong_paths:
                raise FastVideoSLAUnavailableError(
                    f"constructed unexpected FastVideo provider classes: {wrong_paths}"
                )
        provider_payload = {
            "kind": config.kind,
            "topk_ratio": config.topk_ratio,
            "feature_map": config.feature_map,
            "reference_provider": _REFERENCE_PROVIDER_PATHS[config.kind],
            "provider_paths": self._provider_paths,
            "source_fingerprint": identity.fingerprint,
            "layers": {
                index: {
                    "num_heads": spec.num_heads,
                    "head_size": spec.head_size,
                    "softmax_scale": spec.softmax_scale,
                }
                for index, spec in self._layer_specs.items()
            },
        }
        self._provider_fingerprint = hashlib.sha256(
            json.dumps(
                provider_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._receipt_lock = RLock()
        self._attempts = 0
        self._calls = 0
        self._errors = 0
        self._layer_attempts = {index: 0 for index in self._layer_specs}
        self._layer_calls = {index: 0 for index in self._layer_specs}
        self._layer_errors = {index: 0 for index in self._layer_specs}
        super().train(False)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> FastVideoSLAAdapter:
        """Move provider state while preserving FastVideo's FP32 ``proj_l``.

        A parent ``model.to(dtype=...)`` recursively calls ``_apply`` on every
        child.  FastVideo intentionally evaluates the learned projection under
        autocast from FP32 master weights, so allowing that parent conversion
        to turn ``proj_l`` into BF16 would no longer match the reference
        implementation.  Probe the requested placement, then move providers by
        device only and re-check their immutable fingerprints.
        """

        first_projection = next(iter(self.providers.values())).proj_l
        probe = torch.empty(
            0,
            device=first_projection.weight.device,
            dtype=torch.float32,
        )
        transformed = fn(probe)
        target_device = transformed.device
        for layer_index, provider in (
            (int(name), value) for name, value in self.providers.items()
        ):
            provider.to(device=target_device)
            projection = provider.proj_l
            if (
                projection.weight.dtype is not torch.float32
                or projection.bias is None
                or projection.bias.dtype is not torch.float32
            ):
                raise FastVideoSLAContractError(
                    f"layer {layer_index} proj_l changed dtype during placement"
                )
            if (
                _projection_fingerprint(
                    layer_index,
                    projection.weight,
                    projection.bias,
                )
                != self._layer_weight_fingerprints[layer_index]
            ):
                raise FastVideoSLAContractError(
                    f"layer {layer_index} proj_l changed during placement"
                )
        return self

    def _validate_inputs(
        self,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> FastVideoSLALayerSpec:
        if layer_index not in self._layer_specs:
            raise FastVideoSLAContractError(
                f"FastVideo SLA layer {layer_index} was not installed"
            )
        values = {"q": q, "k": k, "v": v}
        if not all(isinstance(value, torch.Tensor) for value in values.values()):
            malformed = {
                name: type(value).__name__
                for name, value in values.items()
                if not isinstance(value, torch.Tensor)
            }
            raise FastVideoSLAContractError(
                f"FastVideo SLA q/k/v must be Tensors, got {malformed}"
            )
        if any(value.ndim != 4 for value in values.values()):
            raise FastVideoSLAContractError(
                "FastVideo SLA requires BSHD q/k/v tensors; shapes="
                f"{ {name: tuple(value.shape) for name, value in values.items()} }"
            )
        if q.shape != k.shape or q.shape != v.shape:
            raise FastVideoSLAContractError(
                "FastVideo SLA q/k/v BSHD shapes must match: "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if q.device != k.device or q.device != v.device:
            raise FastVideoSLAContractError(
                "FastVideo SLA q/k/v devices must match: "
                f"q={q.device}, k={k.device}, v={v.device}"
            )
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise FastVideoSLAContractError(
                "FastVideo SLA q/k/v dtypes must match: "
                f"q={q.dtype}, k={k.dtype}, v={v.dtype}"
            )
        if not all(value.is_contiguous() for value in values.values()):
            raise FastVideoSLAContractError(
                "FastVideo SLA requires contiguous BSHD q/k/v tensors"
            )
        if q.shape[0] <= 0 or q.shape[1] <= 0:
            raise FastVideoSLAContractError(
                "FastVideo SLA batch and sequence dimensions must be positive"
            )
        spec = self._layer_specs[layer_index]
        if tuple(q.shape[2:]) != (spec.num_heads, spec.head_size):
            raise FastVideoSLAContractError(
                f"layer {layer_index} BSHD head geometry is {tuple(q.shape[2:])}, "
                f"expected {(spec.num_heads, spec.head_size)}"
            )
        if not self._allow_non_cuda_for_tests:
            if q.device.type != "cuda":
                raise FastVideoSLAUnavailableError(
                    f"FastVideo SLA requires CUDA tensors, got {q.device}"
                )
            if q.dtype is not torch.bfloat16:
                raise FastVideoSLAContractError(
                    f"FastVideo SLA requires bfloat16 tensors, got {q.dtype}"
                )
        elif q.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise FastVideoSLAContractError(
                f"FastVideo SLA test seam requires a floating compute dtype, got {q.dtype}"
            )
        projection = self.providers[str(layer_index)].proj_l
        if projection.weight.device != q.device or projection.bias.device != q.device:
            raise FastVideoSLAContractError(
                f"layer {layer_index} proj_l is on {projection.weight.device}, "
                f"but q/k/v are on {q.device}; move the adapter before inference"
            )
        if (
            projection.weight.dtype is not torch.float32
            or projection.bias.dtype is not torch.float32
        ):
            raise FastVideoSLAContractError(
                f"layer {layer_index} proj_l must remain float32 at runtime"
            )
        return spec

    def _record_attempt(self, layer_index: int) -> None:
        with self._receipt_lock:
            self._attempts += 1
            self._layer_attempts[layer_index] += 1

    def _record_error(self, layer_index: int) -> None:
        with self._receipt_lock:
            self._errors += 1
            self._layer_errors[layer_index] += 1

    def _record_success(self, layer_index: int) -> tuple[int, int]:
        with self._receipt_lock:
            self._calls += 1
            self._layer_calls[layer_index] += 1
            return self._calls, self._layer_calls[layer_index]

    def forward(
        self,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        current_timestep: int,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> FastVideoSLAResult:
        """Execute one layer's exact upstream provider without dense fallback."""

        normalized_layer = _layer_index(layer_index)
        spec = self._validate_inputs(normalized_layer, q, k, v)
        if isinstance(current_timestep, bool):
            raise FastVideoSLAContractError(
                "current_timestep must be a non-negative integer"
            )
        try:
            timestep = operator.index(current_timestep)
        except TypeError as exc:
            raise FastVideoSLAContractError(
                "current_timestep must be a non-negative integer"
            ) from exc
        if timestep < 0:
            raise FastVideoSLAContractError(
                "current_timestep must be a non-negative integer"
            )
        metadata = self._metadata_factory(timestep, float(self.config.topk_ratio))
        provider = self.providers[str(normalized_layer)]
        self._record_attempt(normalized_layer)
        try:
            if on_provider_attempt is not None:
                on_provider_attempt()
            device_context = (
                torch.cuda.device(q.device)
                if q.device.type == "cuda"
                else nullcontext()
            )
            with device_context:
                output = provider(q, k, v, metadata)
        except Exception as exc:
            self._record_error(normalized_layer)
            raise FastVideoSLAUnavailableError(
                f"FastVideo {self.config.kind} provider failed for layer "
                f"{normalized_layer}: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            if not isinstance(output, torch.Tensor):
                raise FastVideoSLAContractError(
                    f"FastVideo provider returned {type(output).__name__}, "
                    "expected Tensor"
                )
            if output.shape != q.shape:
                raise FastVideoSLAContractError(
                    f"FastVideo provider returned shape {tuple(output.shape)}, "
                    f"expected BSHD {tuple(q.shape)}"
                )
            if output.device != q.device or output.dtype != q.dtype:
                raise FastVideoSLAContractError(
                    "FastVideo provider changed device/dtype: "
                    f"got device={output.device}, dtype={output.dtype}; "
                    f"expected device={q.device}, dtype={q.dtype}"
                )
        except FastVideoSLAContractError:
            self._record_error(normalized_layer)
            raise
        call_index, layer_call_index = self._record_success(normalized_layer)
        reference_path = _REFERENCE_PROVIDER_PATHS[self.config.kind]
        actual_path = self._provider_paths[normalized_layer]
        learned = self._checkpoint.projections[normalized_layer]
        receipt: dict[str, Any] = {
            "algorithm": self.config.kind,
            "provider_family": (
                "fastvideo/sparse-linear-attention"
                if self.config.kind == "fastvideo_sla"
                else "fastvideo/sage-sparse-linear-attention"
            ),
            "provider_path": actual_path,
            "reference_provider_path": reference_path,
            "provider_fingerprint": self._provider_fingerprint,
            "injected_test_provider": self._injected_test_provider,
            "reference_fastvideo_commit": PINNED_FASTVIDEO_COMMIT,
            **self._source_identity.to_receipt(),
            "reference_parity_verified": (
                not self._injected_test_provider
                and actual_path == reference_path
                and self._source_identity.commit == PINNED_FASTVIDEO_COMMIT
                and self._source_identity.clean is True
            ),
            "checkpoint_layout": self._checkpoint.layout,
            "projection_source_keys": list(learned.source_keys),
            "projection_weight_fingerprint": learned.fingerprint,
            "all_projection_weights_fingerprint": self._checkpoint.fingerprint,
            "layer_index": normalized_layer,
            "layer_prefix": spec.prefix,
            "call_index": call_index,
            "layer_call_index": layer_call_index,
            "provider_calls": 1,
            "current_timestep": timestep,
            "topk_ratio": self.config.topk_ratio,
            "feature_map": self.config.feature_map,
            "q": _tensor_contract(q),
            "k": _tensor_contract(k),
            "v": _tensor_contract(v),
            "output": _tensor_contract(output),
            "runtime_effective": True,
        }
        return FastVideoSLAResult(
            output=output,
            receipt=receipt,
        )

    def runtime_report(self) -> dict[str, Any]:
        """Return tensor-free, concurrency-safe lifetime counters."""

        with self._receipt_lock:
            return {
                "kind": self.config.kind,
                "provider_fingerprint": self._provider_fingerprint,
                "checkpoint_layout": self._checkpoint.layout,
                "all_projection_weights_fingerprint": self._checkpoint.fingerprint,
                "expected_layers": list(self._layer_specs),
                "attempts": self._attempts,
                "calls": self._calls,
                "errors": self._errors,
                "layer_attempts": dict(self._layer_attempts),
                "layer_calls": dict(self._layer_calls),
                "layer_errors": dict(self._layer_errors),
                **self._source_identity.to_receipt(),
            }


def fastvideo_sla_checkpoint_key_patterns() -> dict[str, str]:
    """Expose the two audited projection layouts for loader wiring/docs."""

    return {
        "fastvideo_diffusers": (
            "blocks.<layer>.attn1.attn_impl.proj_l.{weight,bias}"
        ),
        "turbodiffusion_original": (
            "blocks.<layer>.self_attn.attn_op.local_attn.proj_l.{weight,bias}"
        ),
    }


__all__ = [
    "FastVideoSLAAdapter",
    "FastVideoSLAConfig",
    "FastVideoSLAContractError",
    "FastVideoSLALayerSpec",
    "FastVideoSLAMetadata",
    "FastVideoSLAProjectionWeights",
    "FastVideoSLAResolvedCheckpoint",
    "FastVideoSLAResult",
    "FastVideoSLASourceIdentity",
    "FastVideoSLAUnavailableError",
    "PINNED_FASTVIDEO_COMMIT",
    "fastvideo_sla_checkpoint_key_patterns",
    "fastvideo_sla_runtime_expectation",
    "resolve_fastvideo_sla_projection_weights",
    "split_fastvideo_sla_projection_weights",
]
