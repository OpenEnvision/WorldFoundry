"""Accelerator capability and policy profiles for in-tree kernels.

Profiles are feature-based (SM, HIP vs CUDA, memory), not a product-name
whitelist. Mixed A100/H100 nodes stay correct when the active tensor
device changes after import. Unknown future SMs keep working through
PyTorch unless ``WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS`` opts them into
qualification.

:data:`QUALIFIED_TRITON_CAPABILITIES` is eligibility, not a benchmark
claim. Each op still runs its runtime predicate and optional numerical
autotune. HIP is never mistaken for NVIDIA CUDA.

Not this module:
    Workload predicates and size thresholds per operator live in
    :mod:`.diffusion`. Persistent winner records live in :mod:`.autotune_cache`.

Public surface:

- :data:`QUALIFIED_TRITON_CAPABILITIES` — SM pairs allowed to attempt Triton.
- :class:`KernelDeviceProfile` / :func:`kernel_device_profile` — one device.
- :func:`detected_kernel_device_profiles` — every visible CUDA/ROCm device.
- :func:`default_kernel_thresholds` / :func:`triton_tensor_eligible` /
  :func:`profile_supports_dtype` — launch-policy helpers.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch

# Architectures qualified for the small, portable in-tree Triton fusion set.
# Product names are intentionally absent: A30/A100 share SM80, A10/30-series
# share SM86, L4/L40/40-series share SM89, and H100/H200 share SM90. Unknown
# future devices remain fully functional through PyTorch unless explicitly
# enabled for qualification with WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS. This
# target list controls eligibility, not a claim of physical benchmarking; each
# workload still passes its runtime predicate and optional numerical autotune.
QUALIFIED_TRITON_CAPABILITIES = frozenset(
    {
        (7, 0),  # Volta: V100
        (7, 5),  # Turing: T4 / RTX 20
        (8, 0),  # Ampere data center: A30 / A100
        (8, 6),  # Ampere: A10 / A40 / RTX 30
        (8, 7),  # Ampere embedded: Jetson Orin
        (8, 9),  # Ada: L4 / L40 / RTX 40
        (9, 0),  # Hopper: H100 / H200
        (10, 0),  # Blackwell data center: B200 / GB200
        (10, 3),  # Blackwell data center: B300 / GB300
        (11, 0),  # Blackwell embedded: Jetson T5000 / T4000
        (12, 0),  # Blackwell client: RTX 50
        (12, 1),  # Blackwell GB10-class
    }
)


@dataclass(frozen=True, slots=True)
class KernelDeviceProfile:
    """Hardware facts and conservative eager-kernel thresholds."""

    device: str
    name: str
    runtime: str
    family: str
    compute_capability: tuple[int, int] | None
    total_memory: int | None
    multiprocessors: int | None
    shared_memory_per_multiprocessor: int | None
    warp_size: int | None
    supports_triton: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_fp8: bool
    supports_fp4: bool
    residual_gate_min_elements: int
    adaln_min_elements: int

    def to_dict(self) -> dict[str, object]:
        """JSON-safe profile; compute capability becomes a two-int list."""

        payload = asdict(self)
        if self.compute_capability is not None:
            payload["compute_capability"] = list(self.compute_capability)
        return payload


# ──────────────────────────────────────────────────────────────────────────
# Family / thresholds — conservative element counts until autotune measures
# ──────────────────────────────────────────────────────────────────────────


def _cuda_family(capability: tuple[int, int] | None) -> str:
    """Map SM major/minor to a policy family; unknown SMs stay named, not refused."""

    if capability is None:
        return "cuda-unknown"
    major, minor = capability
    if major == 7:
        return "volta" if minor == 0 else "turing"
    if major == 8:
        return "ada" if minor == 9 else "ampere"
    if major == 9:
        return "hopper"
    if major == 10:
        return "blackwell-datacenter"
    if major == 11:
        return "blackwell-embedded"
    if major == 12:
        return "blackwell-consumer"
    return f"cuda-sm{major}{minor}"


def _default_thresholds(family: str) -> tuple[int, int]:
    """Return ``(residual_gate, AdaLN)`` element floors for one architecture family."""

    mib = 1024 * 1024
    if family in {"volta", "turing", "cuda-unknown", "rocm"} or family.startswith("cuda-sm"):
        # Legacy/untested targets prefer vendor kernels until enough work is
        # available to amortise a custom launch.
        return 16 * mib, 8 * mib
    if family in {"blackwell-datacenter", "blackwell-embedded", "blackwell-consumer"}:
        # Native Blackwell pointwise/reduction kernels are very fast.  These
        # thresholds remain conservative until each Blackwell target is
        # physically calibrated and its runtime autotune record is available.
        return 16 * mib, 8 * mib
    if family == "hopper":
        return 12 * mib, 6 * mib
    # Ampere and Ada share the currently validated baseline.
    return 8 * mib, 4 * mib


def default_kernel_thresholds(capability: tuple[int, int] | None) -> tuple[int, int]:
    """Return ``(residual_gate, AdaLN)`` thresholds for a CUDA capability."""

    residual_min, adaln_min = _default_thresholds(_cuda_family(capability))
    if capability == (8, 0):
        # Public-dispatch measurements on A100 show the fused LayerNorm/RMSNorm
        # modulation kernels cross PyTorch consistently at roughly 6 Mi
        # elements; 4 Mi can regress common hidden sizes by 5-20%.
        adaln_min = 6 * 1024 * 1024
    return residual_min, adaln_min


def _normalise_cuda_device(device: torch.device | str | int | None) -> torch.device:
    """Pin a bare ``cuda`` handle to the calling thread's current device."""

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if isinstance(device, int):
        return torch.device("cuda", device)
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return parsed


def kernel_device_profile(device: torch.device | str | int | None = None) -> KernelDeviceProfile:
    """Return a cached profile for one concrete accelerator device."""

    parsed = _normalise_cuda_device(device)
    if parsed.type != "cuda" or not torch.cuda.is_available():
        return KernelDeviceProfile(
            device=str(parsed),
            name=parsed.type,
            runtime=parsed.type,
            family=parsed.type,
            compute_capability=None,
            total_memory=None,
            multiprocessors=None,
            shared_memory_per_multiprocessor=None,
            warp_size=None,
            supports_triton=False,
            supports_fp16=parsed.type not in {"cpu", "mps"},
            supports_bf16=False,
            supports_fp8=False,
            supports_fp4=False,
            residual_gate_min_elements=2**63 - 1,
            adaln_min_elements=2**63 - 1,
        )
    index = torch.cuda.current_device() if parsed.index is None else int(parsed.index)
    allow_untested = os.getenv("WORLDFOUNDRY_ALLOW_UNTESTED_GPU_KERNELS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return _cuda_profile_cached(index, bool(getattr(torch.version, "hip", None)), allow_untested)


@lru_cache(maxsize=64)
def _cuda_profile_cached(index: int, hip: bool, allow_untested: bool) -> KernelDeviceProfile:
    """Build one CUDA/HIP profile; HIP never claims Triton until CDNA is qualified."""

    properties = torch.cuda.get_device_properties(index)
    name = str(getattr(properties, "name", f"GPU {index}"))
    total_memory = int(getattr(properties, "total_memory", 0)) or None
    multiprocessors = int(getattr(properties, "multi_processor_count", 0)) or None
    shared_memory_per_multiprocessor = int(getattr(properties, "shared_memory_per_multiprocessor", 0)) or None
    warp_size = int(getattr(properties, "warp_size", 0)) or None
    if hip:
        family = "rocm"
        capability = None
        residual_min, adaln_min = _default_thresholds(family)
        return KernelDeviceProfile(
            device=f"cuda:{index}",
            name=name,
            runtime="rocm",
            family=family,
            compute_capability=capability,
            total_memory=total_memory,
            multiprocessors=multiprocessors,
            shared_memory_per_multiprocessor=shared_memory_per_multiprocessor,
            warp_size=warp_size,
            # The present kernels use CUDA-specific validation and have not
            # yet been qualified on CDNA/RDNA. PyTorch remains the safe path.
            supports_triton=False,
            supports_fp16=True,
            supports_bf16=bool(torch.cuda.is_bf16_supported()),
            supports_fp8=False,
            supports_fp4=False,
            residual_gate_min_elements=residual_min,
            adaln_min_elements=adaln_min,
        )

    major, minor = torch.cuda.get_device_capability(index)
    capability = (int(major), int(minor))
    family = _cuda_family(capability)
    residual_min, adaln_min = default_kernel_thresholds(capability)
    supports_bf16 = capability >= (8, 0)
    supports_fp8 = capability >= (8, 9)
    supports_fp4 = capability >= (10, 0)
    return KernelDeviceProfile(
        device=f"cuda:{index}",
        name=name,
        runtime="cuda",
        family=family,
        compute_capability=capability,
        total_memory=total_memory,
        multiprocessors=multiprocessors,
        shared_memory_per_multiprocessor=shared_memory_per_multiprocessor,
        warp_size=warp_size,
        supports_triton=capability in QUALIFIED_TRITON_CAPABILITIES or allow_untested,
        supports_fp16=capability >= (5, 3),
        supports_bf16=supports_bf16,
        supports_fp8=supports_fp8,
        supports_fp4=supports_fp4,
        residual_gate_min_elements=residual_min,
        adaln_min_elements=adaln_min,
    )


# ──────────────────────────────────────────────────────────────────────────
# Eligibility — dtype + SM gates used by every in-tree Triton predicate
# ──────────────────────────────────────────────────────────────────────────


def profile_supports_dtype(profile: KernelDeviceProfile, dtype: torch.dtype) -> bool:
    """Return whether *profile* may launch the current in-tree fused kernels for *dtype*.

    FP8/FP4 flags describe hardware presence only. Unlisted dtypes (int8, int32)
    are refused here; quantized GEMM has its own contract.
    """

    if dtype == torch.float16:
        return profile.supports_fp16
    if dtype == torch.bfloat16:
        return profile.supports_bf16
    if dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        return profile.supports_fp8
    return dtype == torch.float32


def triton_tensor_eligible(tensor: torch.Tensor) -> bool:
    """Return whether the current in-tree Triton kernels support ``tensor``."""

    if not tensor.is_cuda:
        return False
    profile = kernel_device_profile(tensor.device)
    return profile.supports_triton and profile_supports_dtype(profile, tensor.dtype)


def detected_kernel_device_profiles() -> tuple[KernelDeviceProfile, ...]:
    """Describe all visible CUDA/ROCm devices without assuming homogeneity."""

    if not torch.cuda.is_available():
        return (kernel_device_profile(torch.device("cpu")),)
    return tuple(kernel_device_profile(index) for index in range(torch.cuda.device_count()))


__all__ = [
    "KernelDeviceProfile",
    "QUALIFIED_TRITON_CAPABILITIES",
    "default_kernel_thresholds",
    "detected_kernel_device_profiles",
    "kernel_device_profile",
    "profile_supports_dtype",
    "triton_tensor_eligible",
]
