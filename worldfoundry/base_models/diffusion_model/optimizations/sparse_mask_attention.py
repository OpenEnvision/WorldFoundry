"""Fail-closed adapters for LightX2V's optional Wan sparse-attention lanes.

The algorithms in this module are intentionally delegated to an importable
LightX2V installation instead of being reimplemented with subtly different
masks or kernels.  WorldFoundry owns configuration validation, tensor-boundary
validation, batch adaptation, and JSON-safe runtime receipts; pinned LightX2V
owns the mask generators and CUDA providers.

No dense fallback exists here.  A missing dependency, incomplete provider, or
malformed result raises before it can be reported as sparse execution.  The
Wan processor may choose an explicitly audited dense fallback for compatible
algorithms, but that decision remains outside this adapter.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import operator as index_operator
import subprocess
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import torch

PINNED_LIGHTX2V_COMMIT = "6fb7c1362b89d4908a9ea197bac4fbd7482ee2d5"

_KINDS = frozenset(
    {
        "dynamic_sparse",
        "sparge",
        "nbhd",
        "lightx2v_sla_mask",
        "flexblock",
        "lightx2v_spas_sage",
        "draft_attn",
        "radial_attn",
        "rainfusion_attn",
        "svg_attn",
        "svg2_attn",
        "lightx2v_svg_mask",
    }
)
_DYNAMIC_OPERATORS = frozenset(
    {"triton", "triton_ar", "sage2", "sage3", "fa4", "magi"}
)
_GENERAL_OPERATOR_PATHS = {
    "triton": "sla_triton_operator",
    "sage2": "spas_sage2_operator",
    "sage3": "spas_sage3_operator",
    "fa4": "spas_fa4_operator",
    "magi": "magi_operator",
    "flex_block": "flex_block_operator",
    "flashinfer": "flashinfer_operator",
}
_MASK_GENERATOR_PATHS = {
    "sla": "sla_mask_generator",
    "sparge": "sparge_mask_generator",
    "nbhd": "nbhd_mask_generator",
    "svg": "svg_mask_generator",
}
_KERNEL_SYMBOLS = {
    "triton": "lightx2v.common.ops.attn.kernels.sla_kernel._attention.apply",
    "triton_ar": "lightx2v.common.ops.attn.kernels.sla_kernel_ar._attention_ar.apply",
    "sage2": "lightx2v.common.ops.attn.utils.sparge_util.sage2_block_sparse_attn",
    "sage3": "sageattn3_sparse.sage3_block_sparse_attn",
    "fa4": "flash_attn.cute.flash_attn_func",
    "magi": "magi_attention.functional.flex_flash_attn_func",
    "flex_block": "flex_block_attn.flex_block_attn_func",
    "flashinfer": "flashinfer.sparse.VariableBlockSparseAttentionWrapper.run",
    "meansim_sage2": "spas_sage_attn.core.spas_sage2_attn_meansim_topk_cuda",
    "flex_attention": "torch.nn.attention.flex_attention.flex_attention",
}
_DIRECT_PROVIDER_SPECS = {
    "draft_attn": (
        "lightx2v.common.ops.attn.draft_attn",
        "DraftAttnWeight",
    ),
    "radial_attn": (
        "lightx2v.common.ops.attn.radial_attn",
        "RadialAttnWeight",
    ),
    "rainfusion_attn": (
        "lightx2v.common.ops.attn.rainfusion_attn",
        "RainfusionAttnWeight",
    ),
    "svg_attn": (
        "lightx2v.common.ops.attn.svg_attn",
        "SvgAttnWeight",
    ),
    "svg2_attn": (
        "lightx2v.common.ops.attn.svg2_attn",
        "Svg2AttnWeight",
    ),
}
_GRID_LOCAL_KINDS = frozenset(
    {
        "nbhd",
        "draft_attn",
        "radial_attn",
        "rainfusion_attn",
        "svg_attn",
        "svg2_attn",
        "lightx2v_svg_mask",
    }
)
_REQUEST_LOCAL_KINDS = frozenset(
    {"rainfusion_attn", "svg2_attn", "lightx2v_svg_mask"}
)
_SLA_TOPK_KINDS = frozenset(
    {
        "dynamic_sparse",
        "lightx2v_sla_mask",
        "flexblock",
        "lightx2v_spas_sage",
    }
)
_MAX_PROVIDER_REQUESTS = 32
_NBHD_MASK_CLASS_LOCK = RLock()
_SVG_MASK_CLASS_LOCK = RLock()
_FLASHINFER_RUNTIME_LOCK = RLock()
_PROVIDER_SOURCE_IDENTITY_LOCK = RLock()
_PROVIDER_GIT_TIMEOUT_SECONDS = 30.0
_PROVIDER_GIT_ATTEMPTS = 2


@dataclass(slots=True)
class _FlashinferRuntime:
    """One device-local FlashInfer plan/workspace protected across requests."""

    wrapper: Any
    lock: RLock
    mask: torch.Tensor | None = None
    plan_key: tuple[Any, ...] | None = None


_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
_FLASHINFER_RUNTIMES: dict[
    tuple[int, str, int | None, str], _FlashinferRuntime
] = {}
_PROVIDER_SOURCE_IDENTITIES: dict[str, dict[str, Any]] = {}


class LightX2VSparseUnavailableError(RuntimeError):
    """The selected pinned-upstream provider cannot be constructed or called."""


def _run_provider_git(
    checkout_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Run one provenance query with a remote-filesystem-safe retry budget.

    Reference checkouts commonly live on shared filesystems where a three-second
    metadata stall is possible even when the repository is healthy.  Returning
    and caching an unknown identity after that transient stall poisons every
    block receipt for the lifetime of the process.  Provenance is resolved
    during model preparation, not attention execution, so a bounded retry here
    has no effect on the measured generation hot path.
    """

    command = ["git", "-C", str(checkout_root), *arguments]
    last_error: OSError | subprocess.TimeoutExpired | None = None
    for _attempt in range(_PROVIDER_GIT_ATTEMPTS):
        try:
            return subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_PROVIDER_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = exc
    raise LightX2VSparseUnavailableError(
        "could not audit the LightX2V provider checkout after "
        f"{_PROVIDER_GIT_ATTEMPTS} attempts: {' '.join(command)}"
    ) from last_error


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        normalized = index_operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        ) from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer, got {normalized}")
    return normalized


def _positive_coefficients(values: Any) -> tuple[float, ...]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "nbhd_coefficient must be a non-empty sequence of positive numbers"
        ) from exc
    if not normalized or any(
        not math.isfinite(value) or value <= 0.0 for value in normalized
    ):
        raise ValueError(
            "nbhd_coefficient must be a non-empty sequence of positive finite numbers"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class LightX2VSparseConfig:
    """One explicit LightX2V sparse algorithm/provider selection.

    ``kind`` names the user-visible algorithm.  ``operator`` selects the
    upstream CUDA implementation.  Direct LightX2V lanes use their published
    defaults (DynamicSparse, Sparge meansim+Sage2, NBHD+Magi); setting another
    compatible operator routes through LightX2V's GeneralSparse composition.
    """

    kind: str
    sparsity_ratio: float = 0.8
    operator: str | None = None
    nbhd_coefficient: tuple[float, ...] = (1.0, 0.5, 0.056)
    nbhd_min_width: float = 1.0
    attnmap_frame_num: int | None = None
    per_block_mean: bool = False
    pool_size: int = 128
    skip_timesteps: int = -1
    dense_attn_type: str = "flash_attn3"
    svg_sample_mse_max_row: int = 10000
    svg_num_sampled_rows: int = 64
    svg_context_length: int = 0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().casefold().replace("-", "_")
        kind_aliases = {
            "dynamic": "dynamic_sparse",
            "dynamicsparse": "dynamic_sparse",
            "flex_block": "flexblock",
            # These compatibility aliases mean LightX2V's SLA QK block mask,
            # not FastVideo's learned Sparse-Linear Attention backend.
            "sla": "lightx2v_sla_mask",
            "sla_mask": "lightx2v_sla_mask",
            "sagesla": "lightx2v_spas_sage",
            "sage_sla": "lightx2v_spas_sage",
            "spas_sage": "lightx2v_spas_sage",
            "draft": "draft_attn",
            "radial": "radial_attn",
            "rainfusion": "rainfusion_attn",
            "svg": "svg_attn",
            "svg2": "svg2_attn",
            "svg_mask": "lightx2v_svg_mask",
            "svg_mask_generator": "lightx2v_svg_mask",
        }
        kind = kind_aliases.get(kind, kind)
        if kind not in _KINDS:
            raise ValueError(
                f"unsupported LightX2V sparse kind {self.kind!r}; "
                f"expected one of {sorted(_KINDS)}"
            )
        object.__setattr__(self, "kind", kind)

        sparsity = float(self.sparsity_ratio)
        if not math.isfinite(sparsity) or not 0.0 <= sparsity < 1.0:
            raise ValueError(
                f"sparsity_ratio must be finite and in [0, 1), got {sparsity!r}"
            )
        object.__setattr__(self, "sparsity_ratio", sparsity)

        operator_name = self.operator
        if operator_name is None:
            operator_name = {
                "dynamic_sparse": "triton",
                "sparge": "meansim_sage2",
                "nbhd": "magi",
                "lightx2v_sla_mask": "triton",
                "flexblock": "flex_block",
                "lightx2v_spas_sage": "sage2",
                "draft_attn": "magi",
                "radial_attn": "magi",
                "rainfusion_attn": "flashinfer",
                "svg_attn": "flex_attention",
                "svg2_attn": "flashinfer",
                "lightx2v_svg_mask": "magi",
            }[kind]
        operator_name = str(operator_name).strip().casefold().replace("-", "_")
        operator_aliases = {
            "flexblock": "flex_block",
            "sage": "sage2",
            "sparge": "meansim_sage2",
        }
        operator_name = operator_aliases.get(operator_name, operator_name)
        if kind == "dynamic_sparse":
            allowed = _DYNAMIC_OPERATORS
        elif kind == "sparge":
            allowed = frozenset({"meansim_sage2", *_GENERAL_OPERATOR_PATHS})
        elif kind == "nbhd":
            allowed = frozenset(_GENERAL_OPERATOR_PATHS)
        elif kind == "flexblock":
            allowed = frozenset({"flex_block"})
        elif kind == "lightx2v_spas_sage":
            allowed = frozenset({"sage2", "sage3"})
        elif kind in {"draft_attn", "radial_attn"}:
            allowed = frozenset({"magi"})
        elif kind == "rainfusion_attn":
            allowed = frozenset(_GENERAL_OPERATOR_PATHS)
        elif kind == "svg_attn":
            allowed = frozenset({"flex_attention"})
        elif kind == "svg2_attn":
            allowed = frozenset({"flashinfer"})
        elif kind == "lightx2v_svg_mask":
            allowed = frozenset(_GENERAL_OPERATOR_PATHS)
        else:
            allowed = frozenset(_GENERAL_OPERATOR_PATHS)
        if operator_name not in allowed:
            raise ValueError(
                f"operator {operator_name!r} is invalid for {kind}; "
                f"expected one of {sorted(allowed)}"
            )
        object.__setattr__(self, "operator", operator_name)

        coefficients = _positive_coefficients(self.nbhd_coefficient)
        object.__setattr__(self, "nbhd_coefficient", coefficients)
        minimum_width = float(self.nbhd_min_width)
        if not math.isfinite(minimum_width) or minimum_width <= 0.0:
            raise ValueError(
                f"nbhd_min_width must be positive and finite, got {minimum_width!r}"
            )
        object.__setattr__(self, "nbhd_min_width", minimum_width)
        if self.attnmap_frame_num is not None:
            object.__setattr__(
                self,
                "attnmap_frame_num",
                _positive_int("attnmap_frame_num", self.attnmap_frame_num),
            )
        if not isinstance(self.per_block_mean, bool):
            raise TypeError("per_block_mean must be a bool")
        object.__setattr__(self, "pool_size", _positive_int("pool_size", self.pool_size))
        if isinstance(self.skip_timesteps, bool):
            raise ValueError("skip_timesteps must be an integer >= -1")
        try:
            skip_timesteps = index_operator.index(self.skip_timesteps)
        except TypeError as exc:
            raise ValueError("skip_timesteps must be an integer >= -1") from exc
        if skip_timesteps < -1:
            raise ValueError("skip_timesteps must be an integer >= -1")
        object.__setattr__(self, "skip_timesteps", skip_timesteps)
        dense_attn_type = str(self.dense_attn_type).strip()
        if not dense_attn_type:
            raise ValueError("dense_attn_type cannot be empty")
        object.__setattr__(self, "dense_attn_type", dense_attn_type)
        object.__setattr__(
            self,
            "svg_sample_mse_max_row",
            _positive_int("svg_sample_mse_max_row", self.svg_sample_mse_max_row),
        )
        object.__setattr__(
            self,
            "svg_num_sampled_rows",
            _positive_int("svg_num_sampled_rows", self.svg_num_sampled_rows),
        )
        if isinstance(self.svg_context_length, bool):
            raise ValueError("svg_context_length must be a non-negative integer")
        try:
            svg_context_length = index_operator.index(self.svg_context_length)
        except TypeError as exc:
            raise ValueError(
                "svg_context_length must be a non-negative integer"
            ) from exc
        if svg_context_length < 0:
            raise ValueError("svg_context_length must be a non-negative integer")
        object.__setattr__(self, "svg_context_length", svg_context_length)


@dataclass(frozen=True, slots=True)
class LightX2VSparseResult:
    """Validated BSHD output plus immutable JSON-compatible provider evidence."""

    output: torch.Tensor
    receipt: Mapping[str, Any]


def _import_symbol(module_name: str, symbol: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise LightX2VSparseUnavailableError(
            f"cannot import LightX2V provider module {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    value = getattr(module, symbol, None)
    if value is None:
        raise LightX2VSparseUnavailableError(
            f"LightX2V provider module {module_name!r} has no {symbol!r}"
        )
    return value


def _configured_subclass(base: type[Any], name: str, attrs: dict[str, Any]) -> type[Any]:
    """Configure class-attribute-driven LightX2V providers without global mutation."""

    return type(name, (base,), attrs)


def _device_key(device: torch.device) -> tuple[str, int | None]:
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return device.type, index


def _device_context(device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    _, index = _device_key(device)
    return torch.cuda.device(index)


def _isolated_nbhd_mask_generator(mask_generator: Any) -> Any:
    """Delegate NBHD generation while isolating LightX2V's class cache."""

    base = type(mask_generator)

    def isolated_init(
        self: Any,
        q_block_size: int = 128,
        k_block_size: int = 128,
        sparse_setting: Mapping[str, Any] | None = None,
        attnmap_frame_num: int | None = None,
    ) -> None:
        base.__init__(
            self,
            q_block_size,
            k_block_size,
            {} if sparse_setting is None else dict(sparse_setting),
            attnmap_frame_num,
        )
        self._worldfoundry_seqlen = None
        self._worldfoundry_mask = None

    def isolated_call(self: Any, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Upstream currently addresses ``NbhdMaskGenerator.seqlen/mask`` by
        # class name rather than ``type(self)``. Temporarily project this
        # provider's private cache into those slots under a process-local lock,
        # execute the exact upstream generator, then restore external state.
        with _NBHD_MASK_CLASS_LOCK:
            previous_seqlen = getattr(base, "seqlen", None)
            previous_mask = getattr(base, "mask", None)
            base.seqlen = self._worldfoundry_seqlen
            base.mask = self._worldfoundry_mask
            try:
                result = base.__call__(self, q, k)
                self._worldfoundry_seqlen = getattr(base, "seqlen", None)
                self._worldfoundry_mask = getattr(base, "mask", None)
            finally:
                base.seqlen = previous_seqlen
                base.mask = previous_mask
        return result

    isolated_type = _configured_subclass(
        base,
        "WorldFoundryIsolatedNbhdMaskGenerator",
        {"__init__": isolated_init, "__call__": isolated_call},
    )
    return isolated_type(
        mask_generator.q_block_size,
        mask_generator.k_block_size,
        mask_generator.sparse_setting,
        mask_generator.attnmap_frame_num,
    )


def _isolated_svg_mask_generator(mask_generator: Any) -> Any:
    """Project SVG's upstream class cache into request-local instance state."""

    base = type(mask_generator)

    def isolated_prepare_mask(self: Any, q: torch.Tensor) -> None:
        with _SVG_MASK_CLASS_LOCK:
            previous = {
                name: getattr(base, name, None)
                for name in ("seqlen", "attention_masks", "mask")
            }
            base.seqlen = getattr(self, "_worldfoundry_seqlen", None)
            base.attention_masks = getattr(
                self, "_worldfoundry_attention_masks", None
            )
            base.mask = getattr(self, "_worldfoundry_mask", None)
            try:
                base.prepare_mask(self, q)
                self._worldfoundry_seqlen = getattr(base, "seqlen", None)
                self._worldfoundry_attention_masks = getattr(
                    base, "attention_masks", None
                )
                self._worldfoundry_mask = getattr(base, "mask", None)
                self.attention_masks = self._worldfoundry_attention_masks
                self.mask = self._worldfoundry_mask
            finally:
                for name, value in previous.items():
                    setattr(base, name, value)

    def isolated_call(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        del k
        isolated_prepare_mask(self, q)
        return self._worldfoundry_mask

    isolated_type = _configured_subclass(
        base,
        "WorldFoundryIsolatedSvgMaskGenerator",
        {
            "prepare_mask": isolated_prepare_mask,
            "__call__": isolated_call,
            "seqlen": None,
            "attention_masks": None,
            "mask": None,
        },
    )
    isolated = isolated_type(
        mask_generator.q_block_size,
        mask_generator.k_block_size,
        mask_generator.sparse_setting,
        mask_generator.attnmap_frame_num,
    )
    isolated._worldfoundry_seqlen = None
    isolated._worldfoundry_attention_masks = None
    isolated._worldfoundry_mask = None
    return isolated


def _flashinfer_wrapper_type() -> type[Any]:
    module_name = "lightx2v.common.ops.attn.sparse_operator"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise LightX2VSparseUnavailableError(
            f"cannot import LightX2V FlashInfer operator module: {exc}"
        ) from exc
    dependency = getattr(module, "flashinfer", None)
    wrapper_type = getattr(
        getattr(dependency, "sparse", None),
        "VariableBlockSparseAttentionWrapper",
        None,
    )
    if not callable(wrapper_type):
        raise LightX2VSparseUnavailableError(
            "LightX2V FlashInfer operator requires callable "
            "flashinfer.sparse.VariableBlockSparseAttentionWrapper"
        )
    return wrapper_type


def _flashinfer_runtime(
    wrapper_type: type[Any],
    device: torch.device,
    *,
    backend: str = "fa2",
) -> _FlashinferRuntime:
    if device.type != "cuda":
        raise LightX2VSparseUnavailableError(
            f"LightX2V FlashInfer requires CUDA, got {device.type}"
        )
    device_type, device_index = _device_key(device)
    key = (id(wrapper_type), device_type, device_index, backend)
    with _FLASHINFER_RUNTIME_LOCK:
        runtime = _FLASHINFER_RUNTIMES.get(key)
        if runtime is not None:
            return runtime
        with _device_context(device):
            workspace = torch.empty(
                _FLASHINFER_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            )
            wrapper = wrapper_type(workspace, backend=backend)
        if not callable(getattr(wrapper, "plan", None)) or not callable(
            getattr(wrapper, "run", None)
        ):
            raise LightX2VSparseUnavailableError(
                "FlashInfer sparse wrapper must expose callable plan/run"
            )
        runtime = _FlashinferRuntime(wrapper=wrapper, lock=RLock())
        _FLASHINFER_RUNTIMES[key] = runtime
        return runtime


def _isolated_flashinfer_operator(
    base: type[Any],
    operator_setting: Mapping[str, Any],
) -> Any:
    """Fix upstream's positional constructor bug and class-global plan race."""

    wrapper_type = _flashinfer_wrapper_type()

    def isolated_init(
        self: Any,
        operator_setting: Mapping[str, Any] | None = None,
    ) -> None:
        self.q_block_size = 128
        self.k_block_size = 128
        self.operator_setting = (
            {} if operator_setting is None else dict(operator_setting)
        )

    def isolated_call(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del cu_seqlens_q, cu_seqlens_kv, max_seqlen_kv
        seqlen, head_num, head_dim = q.shape
        kv_seqlen = k.shape[0]
        softmax_scale = kwargs.get("softmax_scale")
        block_mask = mask.squeeze(0)
        if block_mask.ndim != 3:
            raise RuntimeError(
                "LightX2V FlashInfer mask must be [H,Q_blocks,K_blocks], got "
                f"{tuple(block_mask.shape)}"
            )
        runtime = _flashinfer_runtime(wrapper_type, q.device)
        plan_key = (
            tuple(block_mask.shape),
            seqlen,
            kv_seqlen,
            head_num,
            head_dim,
            q.dtype,
            k.dtype,
            str(q.device),
            str(k.device),
            self.q_block_size,
            self.k_block_size,
            softmax_scale,
        )
        with runtime.lock, _device_context(q.device):
            if (
                runtime.mask is None
                or runtime.plan_key != plan_key
                or not torch.equal(block_mask, runtime.mask)
            ):
                _, q_block_num, k_block_num = block_mask.shape
                block_row_sz = torch.full(
                    (head_num, q_block_num),
                    self.q_block_size,
                    dtype=torch.int32,
                    device=q.device,
                )
                block_row_sz[:, -1] = seqlen - self.q_block_size * (
                    q_block_num - 1
                )
                block_col_sz = torch.full(
                    (head_num, k_block_num),
                    self.k_block_size,
                    dtype=torch.int32,
                    device=k.device,
                )
                block_col_sz[:, -1] = kv_seqlen - self.k_block_size * (
                    k_block_num - 1
                )
                runtime.wrapper.plan(
                    block_mask_map=block_mask,
                    block_row_sz=block_row_sz,
                    block_col_sz=block_col_sz,
                    num_qo_heads=head_num,
                    num_kv_heads=head_num,
                    head_dim=head_dim,
                    q_data_type=q.dtype,
                    kv_data_type=k.dtype,
                    sm_scale=softmax_scale,
                )
                runtime.mask = block_mask.clone()
                runtime.plan_key = plan_key
            output = runtime.wrapper.run(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
            ).transpose(0, 1)
        return output.reshape(max_seqlen_q or seqlen, -1)

    operator_type = _configured_subclass(
        base,
        "WorldFoundryIsolatedFlashinferOperator",
        {"__init__": isolated_init, "__call__": isolated_call},
    )
    return operator_type(operator_setting)


def _svg2_flashinfer_forward(
    wrapper_type: type[Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    *,
    is_cpu: bool,
) -> torch.Tensor:
    """Run SVG2 with a reusable wrapper and a fresh dynamic-mask plan."""

    batch, heads, sequence, head_dim = q.shape
    q_clusters = block_row_sz.shape[-1]
    k_clusters = block_col_sz.shape[-1]
    expected_mask = (batch, heads, q_clusters, k_clusters)
    if tuple(block_mask_map.shape) != expected_mask:
        raise RuntimeError(
            "LightX2V SVG2 block mask shape mismatch: "
            f"got {tuple(block_mask_map.shape)}, expected {expected_mask}"
        )
    if is_cpu and any(
        tensor.device.type != "cpu"
        for tensor in (block_mask_map, block_row_sz, block_col_sz)
    ):
        raise RuntimeError("LightX2V SVG2 CPU planning metadata must stay on CPU")
    if not torch.all(
        block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0]
    ) or not torch.all(
        block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0]
    ):
        raise RuntimeError(
            "LightX2V SVG2 requires equal per-head query/key sequence totals"
        )

    runtime = _flashinfer_runtime(wrapper_type, q.device, backend="auto")
    flat_mask = block_mask_map.reshape(batch * heads, q_clusters, k_clusters)
    flat_rows = block_row_sz.reshape(batch * heads, q_clusters)
    flat_cols = block_col_sz.reshape(batch * heads, k_clusters)
    with runtime.lock, _device_context(q.device):
        # SVG2's K-means map is data-dependent. Reuse the expensive wrapper and
        # its internal buffers, but never reuse a plan across dynamic maps.
        runtime.wrapper.plan(
            block_mask_map=flat_mask,
            block_row_sz=flat_rows,
            block_col_sz=flat_cols,
            num_qo_heads=batch * heads,
            num_kv_heads=batch * heads,
            head_dim=head_dim,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        output = runtime.wrapper.run(
            q.reshape(batch * heads, sequence, head_dim),
            k.reshape(batch * heads, sequence, head_dim),
            v.reshape(batch * heads, sequence, head_dim),
        )
    return output.reshape(batch, heads, sequence, head_dim)


def _provider_identity(provider: Any) -> str:
    cls = type(provider)
    for base in cls.__mro__:
        if base is cls or base is object:
            continue
        if base.__module__.startswith("lightx2v."):
            return f"{base.__module__}.{base.__name__}.apply"
    return f"{cls.__module__}.{cls.__name__}.apply"


def _callable_identity(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(module, str) and isinstance(name, str):
        return f"{module}.{name}"
    return type(value).__module__ + "." + type(value).__qualname__


def _resolve_callable(path: str) -> Any:
    """Resolve a dotted callable without treating a successful import as proof."""

    parts = path.split(".")
    module: Any | None = None
    remainder: list[str] = []
    for split_at in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:split_at]))
        except (ImportError, OSError):
            continue
        remainder = parts[split_at:]
        break
    if module is None:
        raise LightX2VSparseUnavailableError(
            f"cannot import required LightX2V kernel symbol {path!r}"
        )
    value = module
    for name in remainder:
        value = getattr(value, name, None)
        if value is None:
            raise LightX2VSparseUnavailableError(
                f"required LightX2V kernel symbol {path!r} is missing at {name!r}"
            )
    if not callable(value):
        raise LightX2VSparseUnavailableError(
            f"required LightX2V kernel symbol {path!r} is not callable"
        )
    return value


def _sage2_capability_symbols(device: torch.device) -> list[str]:
    """Resolve the exact Sparge/Sage2 kernel branch selected by pinned upstream."""

    module = importlib.import_module(
        "lightx2v.common.ops.attn.utils.sparge_util"
    )
    qattn = getattr(module, "qattn", None)
    fused = getattr(module, "fused", None)
    if qattn is None:
        raise LightX2VSparseUnavailableError(
            "LightX2V Sage2 requires spas_sage_attn._qattn"
        )
    capability = torch.cuda.get_device_capability(device)
    arch = f"sm{capability[0]}{capability[1]}"
    supported = {"sm80", "sm86", "sm87", "sm89", "sm90", "sm120"}
    if arch not in supported:
        raise LightX2VSparseUnavailableError(
            f"LightX2V Sage2 has no audited upstream branch for {arch}; "
            f"expected one of {sorted(supported)}"
        )
    symbols: list[tuple[Any, str]] = []
    if arch in {"sm80", "sm86", "sm87"}:
        symbols.append(
            (
                getattr(
                    qattn,
                    "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold",
                    None,
                ),
                "spas_sage_attn._qattn."
                "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold",
            )
        )
    elif arch == "sm90":
        symbols.append(
            (
                getattr(
                    qattn,
                    "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90",
                    None,
                ),
                "spas_sage_attn._qattn."
                "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90",
            )
        )
    else:
        candidates = (
            (
                getattr(
                    module,
                    "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
                    None,
                ),
                "spas_sage_attn._qattn."
                "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
            ),
            (
                getattr(
                    qattn,
                    "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
                    None,
                ),
                "spas_sage_attn._qattn."
                "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
            ),
        )
        selected = next((candidate for candidate in candidates if callable(candidate[0])), None)
        if selected is None:
            raise LightX2VSparseUnavailableError(
                f"LightX2V Sage2 has no callable {arch} qattn branch"
            )
        symbols.append(selected)
    if arch not in {"sm80", "sm86", "sm87"}:
        for name in ("transpose_pad_permute_cuda", "scale_fuse_quant_cuda"):
            symbols.append(
                (
                    getattr(fused, name, None),
                    f"spas_sage_attn._fused.{name}",
                )
            )
    missing = [name for value, name in symbols if not callable(value)]
    if missing:
        raise LightX2VSparseUnavailableError(
            f"LightX2V Sage2 kernel ABI is incomplete for {arch}: {missing}"
        )
    return [name for _, name in symbols]


def lightx2v_sparse_capability_contract(
    config: LightX2VSparseConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Fail-closed package, symbol, ABI, and GPU preflight for one provider."""

    if device.type != "cuda":
        raise LightX2VSparseUnavailableError(
            f"LightX2V sparse capability preflight requires CUDA, got {device.type}"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise LightX2VSparseUnavailableError(
            "LightX2V BF16 sparse providers require NVIDIA SM80 or newer, got "
            f"sm{capability[0]}{capability[1]}"
        )
    required: list[str] = []
    if config.kind in _DIRECT_PROVIDER_SPECS:
        module_name, class_name = _DIRECT_PROVIDER_SPECS[config.kind]
        provider_type = _import_symbol(module_name, class_name)
        if not callable(provider_type):
            raise LightX2VSparseUnavailableError(
                f"LightX2V provider {module_name}.{class_name} is not callable"
            )
        required.append(f"{module_name}.{class_name}.apply")
    elif config.kind == "dynamic_sparse":
        _import_symbol(
            "lightx2v.common.ops.attn.dynamic_sparse_attn",
            "DynamicSparseAttnWeight",
        )
    else:
        _import_symbol(
            "lightx2v.common.ops.attn.general_sparse_attn",
            "GeneralSparseAttnWeight",
        )

    operator_name = str(config.operator)
    if operator_name == "sage2":
        required.extend(_sage2_capability_symbols(device))
    elif operator_name == "meansim_sage2":
        _resolve_callable(_KERNEL_SYMBOLS[operator_name])
        required.append(_KERNEL_SYMBOLS[operator_name])
    elif operator_name == "sage3":
        if capability not in {(9, 0), (12, 0)}:
            raise LightX2VSparseUnavailableError(
                "pinned LightX2V SageAttention3 sparse integration is audited only "
                f"for SM90/SM120, got sm{capability[0]}{capability[1]}"
            )
        _resolve_callable(_KERNEL_SYMBOLS[operator_name])
        required.append(_KERNEL_SYMBOLS[operator_name])
    elif operator_name == "fa4":
        if capability not in {(9, 0), (10, 0), (12, 0)}:
            raise LightX2VSparseUnavailableError(
                "FlashAttention-4 sparse integration requires Hopper or Blackwell, "
                f"got sm{capability[0]}{capability[1]}"
            )
        _resolve_callable(_KERNEL_SYMBOLS[operator_name])
        required.append(_KERNEL_SYMBOLS[operator_name])
    elif operator_name == "flashinfer":
        _flashinfer_wrapper_type()
        required.append(_KERNEL_SYMBOLS[operator_name])
    else:
        _resolve_callable(_KERNEL_SYMBOLS[operator_name])
        required.append(_KERNEL_SYMBOLS[operator_name])

    if config.kind == "draft_attn":
        module = importlib.import_module(_DIRECT_PROVIDER_SPECS[config.kind][0])
        dense = getattr(module, "flash_attn_varlen_func", None)
        if not callable(dense):
            raise LightX2VSparseUnavailableError(
                "LightX2V DraftAttention requires a callable FA2 or FA3 varlen kernel"
            )
        required.append(_callable_identity(dense))
    elif config.kind == "rainfusion_attn":
        registry_module = importlib.import_module(
            "lightx2v.utils.registry_factory"
        )
        registry = getattr(registry_module, "ATTN_WEIGHT_REGISTER", {})
        dense_type = registry.get(config.dense_attn_type)
        if dense_type is None or not callable(getattr(dense_type, "apply", None)):
            raise LightX2VSparseUnavailableError(
                "LightX2V RainFusion dense provider is unavailable: "
                f"{config.dense_attn_type!r}"
            )
        required.append(
            f"{dense_type.__module__}.{dense_type.__name__}.apply"
        )
    elif config.kind in {"svg_attn", "lightx2v_svg_mask"}:
        module = importlib.import_module("lightx2v.common.ops.attn.svg_attn")
        for name in ("wan_sparse_head_placement", "wan_hidden_states_placement"):
            if not callable(getattr(module, name, None)):
                raise LightX2VSparseUnavailableError(
                    f"LightX2V SVG provider is missing callable {name}"
                )
            required.append(f"lightx2v.common.ops.attn.svg_attn.{name}")
    elif config.kind == "svg2_attn":
        module = importlib.import_module("lightx2v.common.ops.attn.svg2_attn")
        dependency = getattr(module, "flashinfer", None)
        wrapper = getattr(
            getattr(dependency, "sparse", None),
            "VariableBlockSparseAttentionWrapper",
            None,
        )
        if not callable(wrapper):
            raise LightX2VSparseUnavailableError(
                "LightX2V SVG2 requires FlashInfer VariableBlockSparseAttentionWrapper"
            )
        required.append(_KERNEL_SYMBOLS["flashinfer"])

    return {
        "preflight_passed": True,
        "cuda_compute_capability": [capability[0], capability[1]],
        "provider_class": (
            ".".join(_DIRECT_PROVIDER_SPECS[config.kind])
            if config.kind in _DIRECT_PROVIDER_SPECS
            else (
                "lightx2v.common.ops.attn.dynamic_sparse_attn.DynamicSparseAttnWeight"
                if config.kind == "dynamic_sparse"
                else "lightx2v.common.ops.attn.general_sparse_attn.GeneralSparseAttnWeight"
            )
        ),
        "required_symbols": sorted(set(required)),
    }


def lightx2v_kernel_symbol(config: LightX2VSparseConfig) -> str:
    """Return the exact upstream kernel symbol required by this config."""

    symbol = _KERNEL_SYMBOLS.get(str(config.operator))
    if symbol is None:
        raise AssertionError(
            f"no canonical kernel symbol for {config.kind}/{config.operator}"
        )
    return symbol


def lightx2v_provider_family(config: LightX2VSparseConfig) -> str:
    """Name the upstream algorithm family without conflating FastVideo SLA."""

    if config.kind == "dynamic_sparse":
        return "lightx2v/dynamic-sparse"
    if config.kind == "sparge" and config.operator == "meansim_sage2":
        return "lightx2v/sparge-direct"
    if config.kind == "nbhd" and config.operator == "magi":
        return "lightx2v/nbhd-direct"
    direct_families = {
        "draft_attn": "lightx2v/draft-attention",
        "radial_attn": "lightx2v/radial-attention",
        "rainfusion_attn": "lightx2v/rainfusion-attention",
        "svg_attn": "lightx2v/svg-attention",
        "svg2_attn": "lightx2v/svg2-attention",
        "lightx2v_svg_mask": "lightx2v/svg-mask-general-sparse",
    }
    if config.kind in direct_families:
        return direct_families[config.kind]
    return "lightx2v/general-sparse"


def _provider_source_identity(provider: Any) -> dict[str, Any]:
    """Resolve and cache checkout commit/dirty/fingerprint off the hot path."""

    unknown = {
        "provider_source_commit": None,
        "provider_source_clean": None,
        "provider_source_fingerprint": None,
        "provider_source_root": None,
    }
    provider_module: str | None = None
    for base in type(provider).__mro__:
        if base is object:
            continue
        if base.__module__.startswith("lightx2v."):
            provider_module = base.__module__
            break
    if provider_module is None:
        return unknown
    try:
        module = importlib.import_module(provider_module)
    except (ImportError, OSError):
        return unknown
    source_file = getattr(module, "__file__", None)
    if not source_file:
        return unknown
    source_path = Path(source_file).resolve()
    cache_key = str(source_path)
    with _PROVIDER_SOURCE_IDENTITY_LOCK:
        cached = _PROVIDER_SOURCE_IDENTITIES.get(cache_key)
        if cached is not None:
            return dict(cached)
    checkout_root = next(
        (parent for parent in (source_path.parent, *source_path.parents) if (parent / ".git").exists()),
        None,
    )
    if checkout_root is None:
        try:
            fingerprint = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError:
            fingerprint = None
        identity = {
            **unknown,
            "provider_source_fingerprint": fingerprint,
        }
        with _PROVIDER_SOURCE_IDENTITY_LOCK:
            _PROVIDER_SOURCE_IDENTITIES[cache_key] = identity
        return dict(identity)
    revision = _run_provider_git(checkout_root, "rev-parse", "HEAD")
    status = _run_provider_git(
        checkout_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    commit = revision.stdout.strip().casefold()
    if revision.returncode != 0 or len(commit) != 40:
        commit = None
    clean = status.returncode == 0 and not status.stdout.strip()
    dirty_payload = status.stdout
    if not clean:
        try:
            diff = subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout_root),
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    "HEAD",
                    "--",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            dirty_payload += "\n<diff-unavailable>"
        else:
            dirty_payload += "\n" + diff.stdout
    fingerprint = hashlib.sha256(
        f"{commit}\0{clean}\0{dirty_payload}".encode()
    ).hexdigest()
    identity = {
        "provider_source_commit": commit,
        "provider_source_clean": clean,
        "provider_source_fingerprint": fingerprint,
        "provider_source_root": str(checkout_root),
    }
    with _PROVIDER_SOURCE_IDENTITY_LOCK:
        _PROVIDER_SOURCE_IDENTITIES[cache_key] = identity
    return dict(identity)


def _provider_source_commit(provider: Any) -> str | None:
    """Compatibility accessor backed by the process-level identity cache."""

    commit = _provider_source_identity(provider)["provider_source_commit"]
    return str(commit) if isinstance(commit, str) else None


def _build_dynamic_provider(config: LightX2VSparseConfig) -> Any:
    module_name = "lightx2v.common.ops.attn.dynamic_sparse_attn"
    base = _import_symbol(module_name, "DynamicSparseAttnWeight")
    provider_type = _configured_subclass(
        base,
        "WorldFoundryDynamicSparseAttnWeight",
        {
            "sparsity_ratio": config.sparsity_ratio,
            "operator": config.operator,
            "per_block_mean": config.per_block_mean,
        },
    )
    return provider_type()


def _build_direct_sparge_provider(config: LightX2VSparseConfig) -> Any:
    module_name = "lightx2v.common.ops.attn.sparge_attn"
    module = importlib.import_module(module_name)
    dependency = getattr(module, "spas_sage_attn", None)
    kernel = getattr(getattr(dependency, "core", None), "spas_sage2_attn_meansim_topk_cuda", None)
    if not callable(kernel):
        raise LightX2VSparseUnavailableError(
            "LightX2V Sparge requires callable "
            "spas_sage_attn.core.spas_sage2_attn_meansim_topk_cuda"
        )
    base = getattr(module, "SpargeAttnWeight", None)
    if base is None:
        raise LightX2VSparseUnavailableError(
            f"LightX2V provider module {module_name!r} has no 'SpargeAttnWeight'"
        )
    provider_type = _configured_subclass(
        base,
        "WorldFoundrySpargeAttnWeight",
        {"sparsity_ratio": config.sparsity_ratio},
    )
    return provider_type()


def _build_direct_nbhd_provider(
    config: LightX2VSparseConfig,
    frame_num: int,
) -> Any:
    module_name = "lightx2v.common.ops.attn.nbhd_attn"
    module = importlib.import_module(module_name)
    if not callable(getattr(module, "magi_ffa_func", None)):
        raise LightX2VSparseUnavailableError(
            "LightX2V NBHD requires callable "
            "magi_attention.functional.flex_flash_attn_func"
        )
    base = getattr(module, "NbhdAttnWeight", None)
    if base is None:
        raise LightX2VSparseUnavailableError(
            f"LightX2V provider module {module_name!r} has no 'NbhdAttnWeight'"
        )
    provider_type = _configured_subclass(
        base,
        "WorldFoundryNbhdAttnWeight",
        {
            "attnmap_frame_num": frame_num,
            "coefficient": list(config.nbhd_coefficient),
            "min_width": config.nbhd_min_width,
            # Upstream caches these on the class.  Per-adapter subclasses keep
            # different requests/grids isolated from one another.
            "seqlen": None,
            "q_ranges": None,
            "k_ranges": None,
            "attn_type_map": None,
        },
    )
    return provider_type()


def _rainfusion_operator(config: LightX2VSparseConfig, module: Any) -> Any:
    """Build RainFusion's exact upstream operator with request-safe FlashInfer."""

    registry_module = importlib.import_module("lightx2v.utils.registry_factory")
    sparse_registry = getattr(registry_module, "SPARSE_OPERATOR_REGISTER", {})
    dense_registry = getattr(registry_module, "ATTN_WEIGHT_REGISTER", {})
    sparse_name = _GENERAL_OPERATOR_PATHS[str(config.operator)]
    sparse_type = sparse_registry.get(sparse_name)
    dense_type = dense_registry.get(config.dense_attn_type)
    if sparse_type is None:
        raise LightX2VSparseUnavailableError(
            f"LightX2V RainFusion sparse operator {sparse_name!r} is unavailable"
        )
    if dense_type is None:
        raise LightX2VSparseUnavailableError(
            "LightX2V RainFusion dense attention provider is unavailable: "
            f"{config.dense_attn_type!r}"
        )
    if config.operator == "flashinfer":
        sparse_operator = _isolated_flashinfer_operator(sparse_type, {})
    else:
        try:
            sparse_operator = sparse_type(
                q_block_size=config.pool_size,
                k_block_size=config.pool_size,
                operator_setting={},
            )
        except TypeError:
            sparse_operator = sparse_type({})
    if (
        getattr(sparse_operator, "q_block_size", config.pool_size)
        != config.pool_size
        or getattr(sparse_operator, "k_block_size", config.pool_size)
        != config.pool_size
    ):
        raise LightX2VSparseUnavailableError(
            f"LightX2V {sparse_name} does not support RainFusion "
            f"pool_size={config.pool_size}"
        )
    dense_provider = dense_type()
    if not callable(getattr(dense_provider, "apply", None)):
        raise LightX2VSparseUnavailableError(
            f"LightX2V dense provider {config.dense_attn_type!r} has no apply"
        )
    operator_type = getattr(module, "CudaRainfusionSparseOperator", None)
    if operator_type is None:
        raise LightX2VSparseUnavailableError(
            "LightX2V rainfusion module has no CudaRainfusionSparseOperator"
        )
    operator = object.__new__(operator_type)
    operator.pool_size = config.pool_size
    operator.dense_attn_type = config.dense_attn_type
    operator.operator_setting = {}
    operator.sparse_operator = sparse_operator
    operator.dense_attn = dense_provider
    return operator


def _build_direct_provider(
    config: LightX2VSparseConfig,
    frame_num: int | None,
) -> Any:
    module_name, class_name = _DIRECT_PROVIDER_SPECS[config.kind]
    module = importlib.import_module(module_name)
    base = getattr(module, class_name, None)
    if base is None:
        raise LightX2VSparseUnavailableError(
            f"LightX2V provider module {module_name!r} has no {class_name!r}"
        )
    attrs: dict[str, Any] = {}
    adaptations: list[str] = []
    if config.kind == "draft_attn":
        attrs.update(
            sparsity_ratio=config.sparsity_ratio,
            reorg_idx_dict={},
            restore_idx_dict={},
            bucket_offsets_dict={},
        )
        adaptations.append("provider_local_grid_index_cache")
    elif config.kind == "radial_attn":
        if frame_num is None:
            raise ValueError("Radial attention requires attnmap_frame_num or grid")
        attrs.update(
            attnmap_frame_num=frame_num,
            seqlen=None,
            q_ranges=None,
            k_ranges=None,
            attn_type_map=None,
        )
        adaptations.append("provider_local_radial_mask_cache")
    elif config.kind == "rainfusion_attn":
        attrs.update(
            pool_size=config.pool_size,
            sparsity=config.sparsity_ratio,
            skip_timesteps=config.skip_timesteps,
            text_len=0,
            txt_first=False,
            backend="cuda_sparse",
            sparse_operator=_GENERAL_OPERATOR_PATHS[str(config.operator)],
            dense_attn_type=config.dense_attn_type,
            operator_setting={},
            _operator=None,
            _operator_key=None,
            _base_blockmask=None,
            _grid_size=None,
            _step_index=None,
        )

        def get_operator(cls: type[Any]) -> Any:
            if cls._operator is None:
                cls._operator = _rainfusion_operator(config, module)
                cls._operator_key = (
                    cls.backend,
                    cls.sparse_operator,
                    cls.dense_attn_type,
                    cls.pool_size,
                )
            return cls._operator

        attrs["_get_operator"] = classmethod(get_operator)
        adaptations.extend(
            [
                "request_local_dynamic_mask",
                "device_local_sparse_operator",
            ]
        )
        if config.operator == "flashinfer":
            adaptations.append("flashinfer_plan_isolation")
    elif config.kind == "svg_attn":
        if frame_num is None:
            raise ValueError("SVG attention requires attnmap_frame_num or grid")
        attrs.update(
            attnmap_frame_num=frame_num,
            seqlen=None,
            attention_masks=None,
            block_mask=None,
        )
        adaptations.append("provider_local_svg_mask_cache")
    elif config.kind == "svg2_attn":
        dependency = getattr(module, "flashinfer", None)
        wrapper_type = getattr(
            getattr(dependency, "sparse", None),
            "VariableBlockSparseAttentionWrapper",
            None,
        )
        if not callable(wrapper_type):
            raise LightX2VSparseUnavailableError(
                "LightX2V SVG2 requires callable FlashInfer variable-block sparse "
                "attention"
            )

        def dynamic_block_sparse_fwd_flashinfer(
            self: Any,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            block_mask_map: torch.Tensor,
            block_row_sz: torch.Tensor,
            block_col_sz: torch.Tensor,
            is_cpu: bool = True,
        ) -> torch.Tensor:
            del self
            return _svg2_flashinfer_forward(
                wrapper_type,
                q,
                k,
                v,
                block_mask_map,
                block_row_sz,
                block_col_sz,
                is_cpu=is_cpu,
            )

        attrs.update(
            centroids_init=False,
            dynamic_block_sparse_fwd_flashinfer=(
                dynamic_block_sparse_fwd_flashinfer
            ),
        )
        adaptations.extend(
            [
                "request_local_kmeans_centroids",
                "device_local_flashinfer_wrapper_reuse",
                "flashinfer_uint8_128m_workspace",
            ]
        )
    provider_type = _configured_subclass(
        base,
        "WorldFoundry" + class_name,
        attrs,
    )
    provider = provider_type()
    if adaptations:
        provider._worldfoundry_adapter_receipt = {
            "compatibility_adaptations": adaptations,
            "state_isolation": "device/grid/request-safe",
        }
    return provider


def _general_mask_kind(config: LightX2VSparseConfig) -> str:
    if config.kind in {
        "lightx2v_sla_mask",
        "flexblock",
        "lightx2v_spas_sage",
    }:
        return "sla"
    if config.kind == "sparge":
        return "sparge"
    if config.kind == "nbhd":
        return "nbhd"
    if config.kind == "lightx2v_svg_mask":
        return "svg"
    raise AssertionError(f"no GeneralSparse mask for {config.kind!r}")


def _build_general_provider(
    config: LightX2VSparseConfig,
    frame_num: int | None,
) -> Any:
    base = _import_symbol(
        "lightx2v.common.ops.attn.general_sparse_attn",
        "GeneralSparseAttnWeight",
    )
    mask_kind = _general_mask_kind(config)
    sparse_setting: dict[str, Any]
    if mask_kind == "nbhd":
        if frame_num is None:
            raise ValueError("NBHD sparse attention requires attnmap_frame_num or grid")
        sparse_setting = {
            "nbhd_coefficient": list(config.nbhd_coefficient),
            "nbhd_min_width": config.nbhd_min_width,
        }
    elif mask_kind == "svg":
        if frame_num is None:
            raise ValueError("SVG mask attention requires attnmap_frame_num or grid")
        sparse_setting = {
            "svg_sample_mse_max_row": config.svg_sample_mse_max_row,
            "svg_num_sampled_rows": config.svg_num_sampled_rows,
            "svg_context_length": config.svg_context_length,
            "svg_sparsity": config.sparsity_ratio,
        }
    else:
        sparse_setting = {"sparsity_ratio": config.sparsity_ratio}
    operator_setting = (
        {"per_block_mean": config.per_block_mean}
        if config.operator == "sage3"
        else {}
    )
    provider_attrs: dict[str, Any] = {
        "sparse_mask_generator": _MASK_GENERATOR_PATHS[mask_kind],
        "sparse_operator": _GENERAL_OPERATOR_PATHS[str(config.operator)],
        "sparse_setting": sparse_setting,
        "operator_setting": operator_setting,
        "attnmap_frame_num": frame_num,
    }
    adaptations: list[str] = []
    if config.operator == "flashinfer":
        flashinfer_base = _import_symbol(
            "lightx2v.common.ops.attn.sparse_operator",
            "FlashinferOperator",
        )

        def setup_flashinfer(instance: Any) -> None:
            instance.operator = _isolated_flashinfer_operator(
                flashinfer_base,
                instance.operator_setting,
            )

        provider_attrs["_setup_operator"] = setup_flashinfer
        adaptations.append(
            "flashinfer_constructor_and_device_plan_isolation"
        )
    if mask_kind == "nbhd":

        def setup_nbhd_mask(instance: Any) -> None:
            base._setup_mask_generator(instance)
            instance.mask_generator = _isolated_nbhd_mask_generator(
                instance.mask_generator
            )

        provider_attrs["_setup_mask_generator"] = setup_nbhd_mask
        adaptations.append("nbhd_provider_local_mask_cache")
    if mask_kind == "svg":
        def setup_svg_mask(instance: Any) -> None:
            base._setup_mask_generator(instance)
            instance.mask_generator = _isolated_svg_mask_generator(
                instance.mask_generator
            )

        provider_attrs["_setup_mask_generator"] = setup_svg_mask
        adaptations.append("request_local_svg_mask_generator")
    provider_type = _configured_subclass(
        base,
        "WorldFoundryGeneralSparseAttnWeight",
        provider_attrs,
    )
    provider = provider_type()
    if adaptations:
        provider._worldfoundry_adapter_receipt = {
            "compatibility_adaptations": adaptations,
            "state_isolation": "device/grid/request-safe",
        }
    return provider


def _build_provider(config: LightX2VSparseConfig, frame_num: int | None) -> Any:
    try:
        if config.kind in _DIRECT_PROVIDER_SPECS:
            provider = _build_direct_provider(config, frame_num)
        elif config.kind == "dynamic_sparse":
            provider = _build_dynamic_provider(config)
        elif config.kind == "sparge" and config.operator == "meansim_sage2":
            provider = _build_direct_sparge_provider(config)
        elif config.kind == "nbhd" and config.operator == "magi":
            if frame_num is None:
                raise ValueError(
                    "NBHD sparse attention requires attnmap_frame_num or grid"
                )
            provider = _build_direct_nbhd_provider(config, frame_num)
        else:
            provider = _build_general_provider(config, frame_num)
    except LightX2VSparseUnavailableError:
        raise
    except (ImportError, OSError) as exc:
        raise LightX2VSparseUnavailableError(
            f"cannot construct LightX2V {config.kind}/{config.operator} provider: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise LightX2VSparseUnavailableError(
            f"LightX2V {config.kind}/{config.operator} provider initialization "
            f"failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not callable(getattr(provider, "apply", None)):
        raise LightX2VSparseUnavailableError(
            f"LightX2V {config.kind}/{config.operator} provider has no callable apply"
        )
    return provider


def _tensor_contract(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
    }


def _validate_nonempty_sla_selection(
    config: LightX2VSparseConfig,
    provider: Any,
    sequence: int,
    *,
    require_metadata: bool,
) -> None:
    """Reject an upstream SLA top-k geometry that selects no key blocks.

    Pinned LightX2V intentionally computes ``int(topk_ratio * K_blocks)``.
    For short sequences or very high sparsity that value can be zero; its
    CUDA providers then receive an empty LUT and may return finite but
    uninitialized output.  Adjusting top-k here would change the pinned
    algorithm, so the auditable behavior is to fail before the kernel call.
    """

    if config.kind not in _SLA_TOPK_KINDS:
        return
    if config.kind == "dynamic_sparse":
        k_block_size = getattr(provider, "BLKK", None)
        topk_ratio = getattr(provider, "topk", None)
    else:
        mask_generator = getattr(provider, "mask_generator", None)
        k_block_size = getattr(mask_generator, "k_block_size", None)
        topk_ratio = getattr(mask_generator, "topk_ratio", None)
    if k_block_size is None or topk_ratio is None:
        if require_metadata:
            raise LightX2VSparseUnavailableError(
                "pinned LightX2V SLA provider does not expose the block/top-k "
                "metadata required for a non-empty-mask preflight"
            )
        return
    try:
        normalized_block_size = _positive_int("SLA key block size", k_block_size)
        normalized_ratio = float(topk_ratio)
    except (TypeError, ValueError) as exc:
        raise LightX2VSparseUnavailableError(
            "pinned LightX2V SLA provider exposes invalid block/top-k metadata"
        ) from exc
    if not math.isfinite(normalized_ratio) or not 0.0 <= normalized_ratio <= 1.0:
        raise LightX2VSparseUnavailableError(
            "pinned LightX2V SLA provider exposes an invalid top-k ratio: "
            f"{normalized_ratio!r}"
        )
    key_blocks = math.ceil(sequence / normalized_block_size)
    selected_blocks = min(key_blocks, int(normalized_ratio * key_blocks))
    if selected_blocks == 0:
        raise LightX2VSparseUnavailableError(
            "pinned LightX2V SLA top-k would select zero key blocks "
            f"(sequence={sequence}, k_block_size={normalized_block_size}, "
            f"key_blocks={key_blocks}, topk_ratio={normalized_ratio}); "
            "refusing to launch a CUDA kernel with an empty LUT"
        )


class LightX2VSparseAdapter:
    """Adapt BSHD Wan tensors to one pinned LightX2V ``apply`` provider."""

    def __init__(
        self,
        config: LightX2VSparseConfig,
        *,
        provider: Any | None = None,
        provider_path: str | None = None,
        allow_non_cuda_for_tests: bool = False,
    ) -> None:
        if not isinstance(config, LightX2VSparseConfig):
            raise TypeError("config must be a LightX2VSparseConfig")
        self.config = config
        self._injected_provider = provider
        self._injected_provider_path = provider_path
        self._allow_non_cuda_for_tests = allow_non_cuda_for_tests
        self._providers: OrderedDict[
            tuple[Any, ...],
            tuple[
                Any,
                str,
                str,
                RLock,
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ],
        ] = OrderedDict()
        self._provider_cache_lock = RLock()
        self._last_provider_key: tuple[Any, ...] | None = None
        self._last_provider_entry: tuple[
            Any,
            str,
            str,
            RLock,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ] | None = None
        self._last_validated_sla_provider: Any | None = None
        self._last_validated_sla_sequence: int | None = None
        self._capability_contracts: dict[
            tuple[str, int | None], dict[str, Any]
        ] = {}

    def _capability_for(self, device: torch.device) -> dict[str, Any]:
        """Resolve one device capability contract once per adapter.

        Importing LightX2V's sparse package and optional CUDA providers can
        take several seconds. Keeping that work behind ``_provider_for`` made
        the first denoise step pay an import/preflight tax even though public
        LightX2V constructs the same providers while loading its model. This
        cache lets the loader preflight/materialize providers explicitly while
        preserving the same fail-closed checks at runtime.
        """

        key = _device_key(device)
        with self._provider_cache_lock:
            cached = self._capability_contracts.get(key)
            if cached is not None:
                return dict(cached)
            if self._injected_provider is not None:
                contract = {
                    "preflight_passed": None,
                    "cuda_compute_capability": None,
                    "provider_class": _provider_identity(
                        self._injected_provider
                    ),
                    "required_symbols": [],
                    "injected_test_provider": True,
                }
            elif self._allow_non_cuda_for_tests and device.type != "cuda":
                contract = {
                    "preflight_passed": None,
                    "cuda_compute_capability": None,
                    "provider_class": None,
                    "required_symbols": [],
                    "non_cuda_test_bypass": True,
                }
            else:
                contract = lightx2v_sparse_capability_contract(
                    self.config,
                    device,
                )
            frozen = dict(contract)
            self._capability_contracts[key] = frozen
            return dict(frozen)

    def prepare(
        self,
        *,
        device: torch.device | str,
        heads: int,
        head_dim: int,
        grid: tuple[int, int, int] | None = None,
        request_key: str | None = None,
    ) -> dict[str, Any]:
        """Preflight and, where shape-safe, construct the upstream provider.

        This deliberately does not launch ``provider.apply`` and cannot count
        as runtime-effective sparse attention. Providers whose state is tied
        to a complete 3D grid or request remain deferred, but their costly
        imports and CUDA ABI checks are still completed during model loading.
        """

        resolved_device = torch.device(device)
        normalized_heads = _positive_int("heads", heads)
        normalized_head_dim = _positive_int("head_dim", head_dim)
        normalized_grid: tuple[int, int, int] | None = None
        if grid is not None:
            try:
                raw_grid = tuple(grid)
            except TypeError as exc:
                raise ValueError(
                    f"grid must contain three positive integers, got {grid!r}"
                ) from exc
            if len(raw_grid) != 3:
                raise ValueError(
                    f"grid must contain three positive integers, got {raw_grid!r}"
                )
            normalized_grid = tuple(
                _positive_int("grid dimension", value) for value in raw_grid
            )

        capability = self._capability_for(resolved_device)
        needs_grid = self.config.kind in _GRID_LOCAL_KINDS
        needs_request = self.config.kind in _REQUEST_LOCAL_KINDS
        deferred_reason = None
        if needs_grid and normalized_grid is None:
            deferred_reason = "complete runtime grid required"
        elif needs_request and not request_key:
            deferred_reason = "request-local provider state required"
        if deferred_reason is not None:
            return {
                "preflight_completed": True,
                "provider_materialized": False,
                "deferred_reason": deferred_reason,
                "device": str(resolved_device),
                "capability_contract": capability,
            }

        (
            _provider,
            provider_path,
            adapter_path,
            _provider_lock,
            source_identity,
            provider_capability,
            _receipt_base,
        ) = self._provider_for(
            normalized_grid,
            resolved_device,
            request_key=request_key,
            heads=normalized_heads,
            head_dim=normalized_head_dim,
        )
        return {
            "preflight_completed": True,
            "provider_materialized": True,
            "deferred_reason": None,
            "device": str(resolved_device),
            "provider_path": provider_path,
            "adapter_path": adapter_path,
            **source_identity,
            "capability_contract": provider_capability,
        }

    def _frame_num(self, grid: tuple[int, int, int] | None) -> int | None:
        configured = self.config.attnmap_frame_num
        if grid is None:
            frame_num = configured
        else:
            frame_num = _positive_int("grid temporal size", grid[0])
        if configured is not None and frame_num is not None and configured != frame_num:
            raise ValueError(
                "attnmap_frame_num/grid mismatch: "
                f"configured={configured}, grid temporal size={frame_num}"
            )
        if (
            self.config.kind == "nbhd"
            and frame_num is not None
            and len(self.config.nbhd_coefficient) > frame_num
        ):
            raise ValueError(
                "NBHD coefficient length must not exceed the temporal grid: "
                f"len={len(self.config.nbhd_coefficient)}, frames={frame_num}"
            )
        return frame_num

    def _provider_for(
        self,
        grid: tuple[int, int, int] | None,
        device: torch.device,
        *,
        request_key: str | None,
        heads: int,
        head_dim: int,
    ) -> tuple[
        Any,
        str,
        str,
        RLock,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        frame_num = self._frame_num(grid)
        device_type, device_index = _device_key(device)
        grid_key = (
            tuple(grid)
            if self.config.kind in _GRID_LOCAL_KINDS and grid is not None
            else None
        )
        local_request_key = (
            request_key or "legacy"
            if self.config.kind in _REQUEST_LOCAL_KINDS
            else None
        )
        geometry_key = (
            (heads, head_dim) if self.config.kind == "svg_attn" else None
        )
        key = (
            device_type,
            device_index,
            grid_key,
            local_request_key,
            geometry_key,
        )
        # Each Wan layer owns one adapter and normally uses a single immutable
        # provider geometry for its lifetime.  Keep the common path free of
        # RLock acquisition and OrderedDict LRU writes; a concurrent request
        # which changes the key simply takes the locked slow path below.
        fast_entry = self._last_provider_entry
        if fast_entry is not None and key == self._last_provider_key:
            return fast_entry
        with self._provider_cache_lock:
            cached = self._providers.get(key)
            if cached is not None:
                self._providers.move_to_end(key)
                self._last_provider_key = key
                self._last_provider_entry = cached
                return cached
            capability_contract = self._capability_for(device)
            if self._injected_provider is not None:
                provider = self._injected_provider
                provider_path = (
                    self._injected_provider_path or _provider_identity(provider)
                )
            else:
                with _device_context(device):
                    provider = _build_provider(self.config, frame_num)
                provider_path = lightx2v_kernel_symbol(self.config)
            if self.config.kind == "svg_attn":
                prepare = getattr(type(provider), "prepare", None)
                if not callable(prepare):
                    raise LightX2VSparseUnavailableError(
                        "LightX2V SVG provider has no callable prepare"
                    )
                with _device_context(device):
                    prepare(
                        head_num=heads,
                        head_dim=head_dim,
                        sample_mse_max_row=self.config.svg_sample_mse_max_row,
                        num_sampled_rows=self.config.svg_num_sampled_rows,
                        context_length=self.config.svg_context_length,
                        sparsity=self.config.sparsity_ratio,
                    )
            adapter_path = _provider_identity(provider)
            if not callable(getattr(provider, "apply", None)):
                raise LightX2VSparseUnavailableError(
                    f"provider {adapter_path!r} has no callable apply"
                )
            source_identity = (
                {
                    "provider_source_commit": None,
                    "provider_source_clean": None,
                    "provider_source_fingerprint": None,
                    "provider_source_root": None,
                }
                if self._injected_provider is not None
                else _provider_source_identity(provider)
            )
            provider_config = {
                "sparsity_ratio": self.config.sparsity_ratio,
                "operator": self.config.operator,
                "nbhd_coefficient": list(self.config.nbhd_coefficient),
                "nbhd_min_width": self.config.nbhd_min_width,
                # The effective temporal grid is inserted per call because
                # DynamicSparse reuses one provider across dynamic grids.
                "attnmap_frame_num": None,
                "per_block_mean": self.config.per_block_mean,
                "pool_size": self.config.pool_size,
                "skip_timesteps": self.config.skip_timesteps,
                "dense_attn_type": self.config.dense_attn_type,
                "svg_sample_mse_max_row": self.config.svg_sample_mse_max_row,
                "svg_num_sampled_rows": self.config.svg_num_sampled_rows,
                "svg_context_length": self.config.svg_context_length,
            }
            receipt_base: dict[str, Any] = {
                "algorithm": self.config.kind,
                "provider_family": lightx2v_provider_family(self.config),
                "operator": self.config.operator,
                "canonical_sparse_provider_path": provider_path,
                "adapter_path": adapter_path,
                "reference_lightx2v_commit": PINNED_LIGHTX2V_COMMIT,
                **source_identity,
                "sparsity_ratio": self.config.sparsity_ratio,
                "provider_config": provider_config,
                "runtime_effective": True,
                "capability_contract": capability_contract,
                "reference_parity_verified": (
                    source_identity["provider_source_commit"]
                    == PINNED_LIGHTX2V_COMMIT
                    and source_identity["provider_source_clean"] is True
                ),
            }
            provider_adaptation = getattr(
                provider,
                "_worldfoundry_adapter_receipt",
                None,
            )
            if isinstance(provider_adaptation, Mapping):
                receipt_base["provider_adaptation"] = dict(
                    provider_adaptation
                )
            cached = (
                provider,
                provider_path,
                adapter_path,
                RLock(),
                source_identity,
                capability_contract,
                receipt_base,
            )
            self._providers[key] = cached
            self._providers.move_to_end(key)
            while len(self._providers) > _MAX_PROVIDER_REQUESTS:
                self._providers.popitem(last=False)
            self._last_provider_key = key
            self._last_provider_entry = cached
            return cached

    def _validate_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid: tuple[int, int, int] | None,
    ) -> None:
        if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            raise TypeError("LightX2V sparse q/k/v must all be tensors")
        if q.ndim != 4:
            raise ValueError(
                f"LightX2V sparse q/k/v must use BSHD rank-4 layout, got {q.ndim}D"
            )
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "LightX2V sparse q/k/v shapes must match exactly; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if q.device != k.device or q.device != v.device:
            raise ValueError("LightX2V sparse q/k/v devices must match")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("LightX2V sparse q/k/v dtypes must match")
        if any(size <= 0 for size in q.shape):
            raise ValueError(f"LightX2V sparse q/k/v dimensions must be positive: {tuple(q.shape)}")
        if not self._allow_non_cuda_for_tests:
            if q.device.type != "cuda":
                raise LightX2VSparseUnavailableError(
                    f"LightX2V sparse providers require CUDA tensors, got {q.device.type}"
                )
            if q.dtype is not torch.bfloat16:
                raise LightX2VSparseUnavailableError(
                    f"LightX2V sparse providers require bfloat16 tensors, got {q.dtype}"
                )
        if grid is not None:
            normalized_grid = tuple(_positive_int("grid dimension", size) for size in grid)
            if len(normalized_grid) != 3:
                raise ValueError(f"grid must contain three dimensions, got {grid!r}")
            if math.prod(normalized_grid) != q.shape[1]:
                raise ValueError(
                    f"grid {normalized_grid} contains {math.prod(normalized_grid)} tokens, "
                    f"but q/k/v contain {q.shape[1]}"
                )
        elif self.config.kind in _GRID_LOCAL_KINDS:
            raise ValueError(
                f"{self.config.kind} requires the complete 3D grid for "
                "provider-local cache isolation"
            )

    def _provider_execution(
        self,
        *,
        layer_idx: int | None,
        step_index: int | None,
    ) -> str:
        if self.config.kind == "draft_attn":
            if layer_idx is None:
                raise ValueError("Draft attention requires an explicit Wan block index")
            return "provider_dense" if layer_idx < 1 else "sparse"
        if self.config.kind == "rainfusion_attn":
            if step_index is None:
                raise ValueError(
                    "RainFusion attention requires an explicit denoise step index"
                )
            if step_index < self.config.skip_timesteps:
                return "provider_dense"
        return "sparse"

    def _provider_kwargs(
        self,
        *,
        q: torch.Tensor,
        sequence: int,
        grid: tuple[int, int, int] | None,
        layer_idx: int | None,
        step_index: int | None,
        softmax_scale: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_seqlen_q": sequence,
            "max_seqlen_kv": sequence,
            "softmax_scale": softmax_scale,
        }
        # DynamicSparse derives its one-sequence layout from q/k and none of
        # its upstream operators consumes cu_seqlens.  Allocating this CUDA
        # tensor in every Wan layer added a measurable launch-path bubble.
        if self.config.kind != "dynamic_sparse":
            cu_seqlens = torch.tensor(
                [0, sequence],
                dtype=torch.int32,
                device=q.device,
            )
            kwargs.update(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            )
        if self.config.kind == "draft_attn":
            assert grid is not None
            kwargs.update(
                block_idx=layer_idx,
                scheduler=SimpleNamespace(
                    latents=SimpleNamespace(
                        shape=(1, 1, grid[1], grid[2])
                    ),
                    patch_size=(1, 1, 1),
                    step_index=step_index,
                ),
            )
        elif self.config.kind == "rainfusion_attn":
            assert grid is not None
            kwargs.update(
                grid_sizes=grid,
                scheduler=SimpleNamespace(step_index=step_index),
            )
        return kwargs

    def _effective_provider_path(self, provider: Any, execution: str) -> str:
        if execution != "provider_dense":
            return lightx2v_kernel_symbol(self.config)
        if self.config.kind == "draft_attn":
            module = importlib.import_module(_DIRECT_PROVIDER_SPECS["draft_attn"][0])
            dense = getattr(module, "flash_attn_varlen_func", None)
            if not callable(dense):
                raise LightX2VSparseUnavailableError(
                    "DraftAttention selected dense layer 0 without a callable FA kernel"
                )
            return _callable_identity(dense)
        if self.config.kind == "rainfusion_attn":
            operator = type(provider)._get_operator()
            return _provider_identity(operator.dense_attn)
        raise AssertionError(
            f"no provider-dense path is defined for {self.config.kind}"
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid: tuple[int, int, int] | None = None,
        softmax_scale: float | None = None,
        layer_idx: int | None = None,
        step_index: int | None = None,
        request_key: str | None = None,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> LightX2VSparseResult:
        self._validate_inputs(q, k, v, grid)
        (
            provider,
            provider_path,
            adapter_path,
            provider_lock,
            _source_identity,
            _capability_contract,
            receipt_base,
        ) = self._provider_for(
            grid,
            q.device,
            request_key=request_key,
            heads=q.shape[2],
            head_dim=q.shape[3],
        )
        batch, sequence, heads, head_dim = q.shape
        if (
            provider is not self._last_validated_sla_provider
            or sequence != self._last_validated_sla_sequence
        ):
            _validate_nonempty_sla_selection(
                self.config,
                provider,
                sequence,
                # CPU-only contract tests commonly inject or monkeypatch a
                # provider that deliberately implements only ``apply``.  A real
                # CUDA launch must expose enough pinned-upstream metadata to prove
                # that the selected LUT cannot be empty.
                require_metadata=not self._allow_non_cuda_for_tests,
            )
            self._last_validated_sla_provider = provider
            self._last_validated_sla_sequence = sequence
        execution = self._provider_execution(
            layer_idx=layer_idx,
            step_index=step_index,
        )
        outputs: list[torch.Tensor] = []
        if on_provider_attempt is not None:
            on_provider_attempt()
        prepare_mask = getattr(provider, "prepare_mask", None)
        if (
            self.config.kind == "nbhd"
            and self.config.operator == "magi"
            and callable(prepare_mask)
        ):
            # The configured direct-NBHD subclass is unique to this full grid.
            # Pre-populate its class cache once under a narrow lock, then the
            # expensive provider kernels can run without serializing requests.
            with provider_lock, _device_context(q.device):
                prepare_mask(seqlen=sequence, head_num=heads)
        serialized = self.config.kind in _GRID_LOCAL_KINDS
        provider_context = provider_lock if serialized else nullcontext()
        with provider_context, _device_context(q.device):
            for batch_index in range(batch):
                provider_kwargs = self._provider_kwargs(
                    q=q[batch_index],
                    sequence=sequence,
                    grid=grid,
                    layer_idx=layer_idx,
                    step_index=step_index,
                    softmax_scale=softmax_scale,
                )
                result = provider.apply(
                    q[batch_index],
                    k[batch_index],
                    v[batch_index],
                    **provider_kwargs,
                )
                if not isinstance(result, torch.Tensor):
                    raise RuntimeError(
                        f"LightX2V provider {provider_path} returned "
                        f"{type(result).__name__}, expected Tensor"
                    )
                expected_flat = (sequence, heads * head_dim)
                expected_heads = (sequence, heads, head_dim)
                if tuple(result.shape) == expected_flat:
                    result = result.reshape(expected_heads)
                elif tuple(result.shape) != expected_heads:
                    raise RuntimeError(
                        f"LightX2V provider {provider_path} returned shape "
                        f"{tuple(result.shape)}, expected {expected_flat} or "
                        f"{expected_heads}"
                    )
                if result.device != q.device or result.dtype != q.dtype:
                    raise RuntimeError(
                        f"LightX2V provider {provider_path} changed device/dtype: "
                        f"got device={result.device}, dtype={result.dtype}; "
                        f"expected device={q.device}, dtype={q.dtype}"
                    )
                outputs.append(result)
        output = (
            outputs[0].unsqueeze(0)
            if batch == 1
            else torch.stack(outputs, dim=0)
        )
        if output.shape != q.shape:
            raise RuntimeError(
                f"LightX2V provider {provider_path} produced final shape "
                f"{tuple(output.shape)}, expected {tuple(q.shape)}"
            )
        effective_provider_path = (
            self._injected_provider_path or provider_path
            if self._injected_provider is not None
            else self._effective_provider_path(provider, execution)
        )
        receipt: dict[str, Any] = dict(receipt_base)
        provider_config = dict(receipt_base["provider_config"])
        provider_config["attnmap_frame_num"] = self._frame_num(grid)
        receipt.update({
            "provider_path": effective_provider_path,
            "provider_calls": batch,
            "grid": list(grid) if grid is not None else None,
            "provider_config": provider_config,
            "input": _tensor_contract(q),
            "output": _tensor_contract(output),
            "execution": execution,
            "sparse_kernel_executed": execution == "sparse",
            "provider_dense_executed": execution == "provider_dense",
        })
        if self.config.kind == "nbhd":
            receipt["attnmap_frame_num"] = self._frame_num(grid)
            receipt["nbhd_coefficient"] = list(self.config.nbhd_coefficient)
            receipt["nbhd_min_width"] = self.config.nbhd_min_width
        return LightX2VSparseResult(output=output, receipt=receipt)


def parse_lightx2v_sparse(option: Any) -> LightX2VSparseConfig:
    """Parse a kind string or strict mapping into a validated config."""

    if isinstance(option, LightX2VSparseConfig):
        return option
    if isinstance(option, str):
        return LightX2VSparseConfig(kind=option)
    if not isinstance(option, Mapping):
        raise TypeError(
            "LightX2V sparse attention must be a kind string, mapping, or "
            "LightX2VSparseConfig"
        )
    allowed = {
        "kind",
        "sparsity_ratio",
        "operator",
        "nbhd_coefficient",
        "nbhd_min_width",
        "attnmap_frame_num",
        "per_block_mean",
        "pool_size",
        "skip_timesteps",
        "dense_attn_type",
        "svg_sample_mse_max_row",
        "svg_num_sampled_rows",
        "svg_context_length",
    }
    unknown = sorted(set(option) - allowed)
    if unknown:
        raise ValueError(f"unknown LightX2V sparse-attention fields: {unknown}")
    if "kind" not in option:
        raise ValueError("LightX2V sparse-attention mapping requires 'kind'")
    return LightX2VSparseConfig(**dict(option))


__all__ = [
    "LightX2VSparseAdapter",
    "LightX2VSparseConfig",
    "LightX2VSparseResult",
    "LightX2VSparseUnavailableError",
    "PINNED_LIGHTX2V_COMMIT",
    "lightx2v_kernel_symbol",
    "lightx2v_provider_family",
    "lightx2v_sparse_capability_contract",
    "parse_lightx2v_sparse",
]
