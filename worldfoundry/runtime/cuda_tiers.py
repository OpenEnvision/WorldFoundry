"""CUDA wheel-tier helpers for unified WorldFoundry runtime environments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from worldfoundry.core.io.paths import conda_envs_root_path

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CUDA_TIER = "cu128"
SUPPORTED_CUDA_TIERS = ("cu121", "cu124", "cu128")
UNIFIED_ENV_NAME_TEMPLATE = "worldfoundry-unified-{tier}"
TORCH_WHEEL_INDEX_TEMPLATE = "https://download.pytorch.org/whl/{tier}"
TIER_MINIMUM_DRIVER = {
    "cu121": (12, 1),
    "cu124": (12, 4),
    "cu128": (12, 8),
}
# Keep each install range inside the releases published by its CUDA wheel
# index. These are resolver constraints, not lockfiles.
TIER_TORCH_SPECS: dict[str, dict[str, str]] = {
    "cu121": {
        "torch": "torch>=2.4,<2.6",
        "torchvision": "torchvision>=0.19,<0.21",
        "torchaudio": "torchaudio>=2.4,<2.6",
    },
    "cu124": {
        "torch": "torch>=2.4,<2.7",
        "torchvision": "torchvision>=0.19,<0.22",
        "torchaudio": "torchaudio>=2.4,<2.7",
    },
    "cu128": {
        "torch": "torch>=2.7,<2.12.0",
        "torchvision": "torchvision>=0.22,<0.27.0",
        "torchaudio": "torchaudio>=2.7,<2.12.0",
    },
}

_LEGACY_PROFILE_TO_TIER = {
    "cu113": "cu128",
    "cu118": "cu128",
    "cu121": "cu121",
    "cu124": "cu124",
    "cu126": "cu128",
    "cu128": "cu128",
    "cu129": "cu128",
    "cu130": "cu128",
}


def normalize_cuda_profile(cuda_profile: str) -> str:
    """Reduce legacy or annotated profiles to a canonical ``cuXYZ`` bucket."""

    profile = str(cuda_profile or "").strip()
    if profile in {"", "cpu", "prepare_only"}:
        return profile
    match = re.match(r"(cu[0-9]{3})", profile)
    if match:
        return match.group(1)
    return profile


def cuda_version_tuple(version: str | None) -> tuple[int, int]:
    """Parse a CUDA version string into a ``(major, minor)`` integer tuple."""
    if not version:
        return (0, 0)
    major, _, minor = version.partition(".")
    try:
        return (int(major), int(minor or 0))
    except ValueError:
        return (0, 0)


def detect_nvidia_driver_cuda() -> str | None:
    """Detect the NVIDIA driver CUDA version via ``nvidia-smi`` or env override.

    The env override is read on every call so tests and long-lived processes
    that change ``WORLDFOUNDRY_DETECTED_DRIVER_CUDA`` see the update; only the
    ``nvidia-smi`` probe result is cached.
    """
    override = os.environ.get("WORLDFOUNDRY_DETECTED_DRIVER_CUDA")
    if override:
        return override
    return _detect_nvidia_driver_cuda_from_smi()


@lru_cache(maxsize=1)
def _detect_nvidia_driver_cuda_from_smi() -> str | None:
    """Probe ``nvidia-smi`` once per process for the driver CUDA version."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        completed = subprocess.run(
            [nvidia_smi],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", completed.stdout or "")
    return match.group(1) if match else None


def best_cuda_tier_for_driver(
    driver_cuda: str | None = None,
    *,
    allow_missing_driver: bool = True,
) -> str:
    """Pick the highest supported tier that the local driver can run.

    Historical runtime callers retain the ``cu128`` fallback when no driver is
    visible. Installer auto-detection disables that fallback so a CPU host does
    not silently select a CUDA wheel tier and attempt a flash-attn source build.
    """

    detected = driver_cuda if driver_cuda is not None else detect_nvidia_driver_cuda()
    driver = cuda_version_tuple(detected)
    if driver == (0, 0):
        if allow_missing_driver:
            return DEFAULT_CUDA_TIER
        raise ValueError(
            "No NVIDIA driver CUDA version detected. Pass an explicit --cuda tier "
            f"({'/'.join(SUPPORTED_CUDA_TIERS)}); on CPU-only hosts also use "
            "--allow-no-cuda --skip-flash-attn."
        )
    if driver >= (12, 8):
        return "cu128"
    if driver >= (12, 4):
        return "cu124"
    if driver >= (12, 1):
        return "cu121"
    return DEFAULT_CUDA_TIER


def _tier_minimum_driver(tier: str) -> tuple[int, int]:
    """Return the minimum ``(major, minor)`` driver version required by *tier*."""
    return TIER_MINIMUM_DRIVER.get(tier, (12, 8))


def _tier_rank(tier: str) -> int:
    """Return a numeric ranking for *tier* used for comparison."""
    return {"cu121": 1, "cu124": 2, "cu128": 3}.get(tier, 3)


def cap_tier_to_driver(tier: str, driver_cuda: str | None = None) -> str:
    """Clamp a requested tier to the highest tier supported by the local driver."""

    if tier not in SUPPORTED_CUDA_TIERS:
        return tier
    driver = cuda_version_tuple(driver_cuda or detect_nvidia_driver_cuda())
    if driver == (0, 0):
        return tier
    allowed = "cu121"
    for candidate in SUPPORTED_CUDA_TIERS:
        if driver >= _tier_minimum_driver(candidate):
            allowed = candidate
    if _tier_rank(tier) <= _tier_rank(allowed):
        return tier
    return allowed


def resolve_cuda_tier(
    cuda_profile: str,
    *,
    driver_cuda: str | None = None,
    preferred_tier: str | None = None,
) -> str:
    """Map a model/runtime profile to one install tier among cu121/cu124/cu128."""

    normalized = normalize_cuda_profile(cuda_profile)
    if normalized in {"", "cpu", "prepare_only"}:
        return normalized

    tier = _LEGACY_PROFILE_TO_TIER.get(normalized, best_cuda_tier_for_driver(driver_cuda))
    if tier not in SUPPORTED_CUDA_TIERS:
        tier = best_cuda_tier_for_driver(driver_cuda)

    if preferred_tier in SUPPORTED_CUDA_TIERS:
        tier = preferred_tier
    return cap_tier_to_driver(tier, driver_cuda)


def unified_env_name(tier: str = DEFAULT_CUDA_TIER) -> str:
    """Return the conda environment name for the unified tier *tier*."""
    normalized = normalize_cuda_profile(tier)
    if normalized in {"", "cpu", "prepare_only"}:
        return f"worldfoundry-{normalized or 'cpu'}"
    if normalized not in SUPPORTED_CUDA_TIERS:
        normalized = DEFAULT_CUDA_TIER
    return UNIFIED_ENV_NAME_TEMPLATE.format(tier=normalized)


def torch_wheel_index_url(tier: str = DEFAULT_CUDA_TIER) -> str:
    """Return the PyTorch wheel index URL for the given CUDA tier."""
    normalized = normalize_cuda_profile(tier)
    if normalized not in SUPPORTED_CUDA_TIERS:
        normalized = DEFAULT_CUDA_TIER
    return TORCH_WHEEL_INDEX_TEMPLATE.format(tier=normalized)


def torch_specs_for_tier(tier: str = DEFAULT_CUDA_TIER) -> dict[str, str]:
    """Return a copy of the torch package constraints for one CUDA tier."""

    normalized = normalize_cuda_profile(tier)
    if normalized not in TIER_TORCH_SPECS:
        normalized = DEFAULT_CUDA_TIER
    return dict(TIER_TORCH_SPECS[normalized])


def _unified_env_prefix(tier: str = DEFAULT_CUDA_TIER) -> Path:
    """Resolve the filesystem prefix for the unified-tier conda environment."""
    override = os.environ.get("WORLDFOUNDRY_UNIFIED_ENV_PREFIX")
    if override:
        return Path(override).expanduser()
    return conda_envs_root_path() / unified_env_name(tier)


def unified_env_exists(tier: str = DEFAULT_CUDA_TIER) -> bool:
    """Return whether the unified-tier conda environment is installed."""
    return (_unified_env_prefix(tier) / "bin" / "python").is_file()


def unified_env_enabled() -> bool:
    """Return whether the unified env routing policy is active.

    When ``WORLDFOUNDRY_USE_UNIFIED_ENV`` is ``auto`` (default), the policy is
    enabled if any unified-tier environment directory exists on disk.
    """
    value = os.environ.get("WORLDFOUNDRY_USE_UNIFIED_ENV", "auto").strip().lower()
    if value in {"", "auto"}:
        return any(unified_env_exists(tier) for tier in SUPPORTED_CUDA_TIERS)
    return value not in {"0", "false", "no", "off"}


def preferred_unified_tier() -> str | None:
    """Return the user-specified preferred CUDA tier, or ``None`` if unset."""
    value = os.environ.get("WORLDFOUNDRY_CUDA_PROFILE") or os.environ.get("WORLDFOUNDRY_CUDA_TIER")
    if not value:
        return None
    normalized = normalize_cuda_profile(value)
    return normalized if normalized in SUPPORTED_CUDA_TIERS else None


def resolve_install_tier(requested: str | None = None, *, driver_cuda: str | None = None) -> str:
    """Resolve user-facing installer input to one supported CUDA wheel tier."""

    value = normalize_cuda_profile(
        requested
        or os.environ.get("WORLDFOUNDRY_CUDA_PROFILE")
        or os.environ.get("WORLDFOUNDRY_CUDA_TIER")
        or "auto"
    )
    if value in {"", "auto"}:
        return best_cuda_tier_for_driver(driver_cuda, allow_missing_driver=False)
    value = _LEGACY_PROFILE_TO_TIER.get(value, value)
    if value not in SUPPORTED_CUDA_TIERS:
        raise ValueError(
            f"Unsupported CUDA tier {requested!r}; use auto, "
            + ", ".join(SUPPORTED_CUDA_TIERS)
        )
    return cap_tier_to_driver(value, driver_cuda)


def cuda_tier_report(requested: str | None = None, *, driver_cuda: str | None = None) -> dict[str, object]:
    """Build a diagnostic report summarizing CUDA tier resolution.

    Args:
        requested: User-specified tier preference, or ``None`` for auto-detection.
        driver_cuda: Override for the detected driver CUDA version.
    """
    tier = resolve_install_tier(requested, driver_cuda=driver_cuda)
    detected_driver = driver_cuda or detect_nvidia_driver_cuda()
    return {
        "requested": requested or "auto",
        "tier": tier,
        "driver_cuda": detected_driver,
        "minimum_driver_cuda": ".".join(str(part) for part in _tier_minimum_driver(tier)),
        "env_name": unified_env_name(tier),
        "torch_index_url": torch_wheel_index_url(tier),
        "torch_specs": torch_specs_for_tier(tier),
        "supported_tiers": list(SUPPORTED_CUDA_TIERS),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a WorldFoundry CUDA wheel tier.")
    parser.add_argument("--requested", default=None, help="auto, cu128, cu124, or cu121.")
    parser.add_argument("--driver-cuda", default=None, help="Override detected nvidia-smi CUDA version.")
    parser.add_argument("--field", choices=("json", "tier", "env_name", "torch_index_url"), default="json")
    args = parser.parse_args(argv)

    payload = cuda_tier_report(args.requested, driver_cuda=args.driver_cuda)
    if args.field == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        print(payload[args.field])
    return 0


__all__ = [
    "DEFAULT_CUDA_TIER",
    "SUPPORTED_CUDA_TIERS",
    "TIER_MINIMUM_DRIVER",
    "TIER_TORCH_SPECS",
    "UNIFIED_ENV_NAME_TEMPLATE",
    "TORCH_WHEEL_INDEX_TEMPLATE",
    "best_cuda_tier_for_driver",
    "cuda_tier_report",
    "cuda_version_tuple",
    "detect_nvidia_driver_cuda",
    "normalize_cuda_profile",
    "preferred_unified_tier",
    "resolve_install_tier",
    "resolve_cuda_tier",
    "torch_specs_for_tier",
    "torch_wheel_index_url",
    "unified_env_enabled",
    "unified_env_exists",
    "unified_env_name",
]


if __name__ == "__main__":
    raise SystemExit(_main())
