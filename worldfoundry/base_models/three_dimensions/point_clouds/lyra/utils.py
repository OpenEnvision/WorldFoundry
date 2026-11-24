"""Lyra reconstruction paths, checkpoints, and runtime environment helpers."""

import os
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import torch

from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    package_root,
)

DEFAULT_LYRA2_ALIAS = "lyra-2"
DEFAULT_LYRA1_ALIAS = "lyra-1"


def lyra_runtime_root(name: str) -> Path:
    visual_generation_root = package_root() / "synthesis" / "visual_generation"
    if name == "lyra2_runtime":
        return visual_generation_root / "lyra_2"
    if name == "lyra1_runtime":
        return visual_generation_root / "lyra_1" / "lyra1_runtime"
    raise ValueError(f"Unsupported Lyra runtime name: {name}")


def _candidate_subdir(
    path_value: Path,
    repo_dir_name: str,
    sentinel_parts: Iterable[str],
) -> Path | None:
    sentinel_parts = tuple(sentinel_parts)
    if path_value.joinpath(*sentinel_parts).is_file():
        return path_value
    nested = path_value / repo_dir_name
    if nested.joinpath(*sentinel_parts).is_file():
        return nested
    return None


def _candidate_lyra2_subdir(path_value: Path) -> Path | None:
    return _candidate_subdir(
        path_value,
        repo_dir_name="Lyra-2",
        sentinel_parts=("lyra_2", "__init__.py"),
    )


def _candidate_lyra2_checkpoint_subdir(path_value: Path) -> Path | None:
    candidates = [
        path_value,
        path_value / "Lyra-2.0",
        path_value / "Lyra-2",
    ]
    for candidate in candidates:
        checkpoint_dir = candidate / "checkpoints" / "model"
        negative_prompt_path = candidate / "checkpoints" / "text_encoder" / "negative_prompt.pt"
        da3_model_path = candidate / "checkpoints" / "recon" / "model.pt"
        if checkpoint_dir.is_dir() and negative_prompt_path.is_file() and da3_model_path.exists():
            return candidate
    return None


def _candidate_lyra1_subdir(path_value: Path) -> Path | None:
    return _candidate_subdir(
        path_value,
        repo_dir_name="Lyra-1",
        sentinel_parts=("sample.py",),
    )


def _repo_candidates(repo_dir_name: str) -> list[Path]:
    runtime_name = "lyra2_runtime" if repo_dir_name == "Lyra-2" else "lyra1_runtime"
    return [
        lyra_runtime_root(runtime_name),
    ]


def local_repo_candidates() -> list[Path]:
    return _repo_candidates("Lyra-2")


def local_lyra2_checkpoint_candidates() -> list[Path]:
    ckpt_root = checkpoint_root_path()
    hfd_root = hfd_root_path()
    return [
        hfd_root / "Lyra-2.0",
        hfd_root / "nvidia--Lyra-2.0",
        ckpt_root / "Lyra-2.0",
        ckpt_root / "Lyra-2",
    ]


def default_local_repo() -> Path | None:
    for candidate in local_repo_candidates():
        resolved = _candidate_lyra2_subdir(candidate)
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def local_lyra1_repo_candidates() -> list[Path]:
    return _repo_candidates("Lyra-1")


def default_local_lyra2_checkpoint_root() -> Path | None:
    for candidate in local_lyra2_checkpoint_candidates():
        resolved = _candidate_lyra2_checkpoint_subdir(candidate)
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def default_local_lyra1_repo() -> Path | None:
    for candidate in local_lyra1_repo_candidates():
        resolved = _candidate_lyra1_subdir(candidate)
        if resolved is not None and resolved.is_dir():
            return resolved.resolve()
    return None


def local_lyra1_checkpoint_candidates() -> list[Path]:
    ckpt_root = checkpoint_root_path()
    hfd_root = hfd_root_path()
    return [
        hfd_root / "Lyra",
        hfd_root / "lyra-1",
        hfd_root / "nvidia--Lyra",
        ckpt_root / "Lyra",
        ckpt_root / "lyra-1",
        ckpt_root / "Lyra-1" / "checkpoints",
    ]


def default_local_lyra1_checkpoint_root() -> Path | None:
    for candidate in local_lyra1_checkpoint_candidates():
        if _candidate_lyra1_weights_subdir(candidate) is not None:
            return candidate.resolve()
    return None


def resolve_repo_root(pretrained_model_path: str | None) -> str:
    default_runtime = default_local_repo()
    if pretrained_model_path is None or str(pretrained_model_path).strip() in {
        "",
        DEFAULT_LYRA2_ALIAS,
        "lyra",
        "lyra2",
        "Lyra-2",
        "Lyra",
        "nvidia/Lyra-2.0",
    }:
        if default_runtime is None:
            raise FileNotFoundError("Unable to locate the vendored Lyra-2 runtime under WorldFoundry.")
        return str(default_runtime)

    candidate = Path(str(pretrained_model_path)).expanduser()
    checkpoint_root = _candidate_lyra2_checkpoint_subdir(candidate)
    if checkpoint_root is not None:
        if default_runtime is None:
            raise FileNotFoundError("Lyra-2 checkpoints were found, but the vendored Lyra-2 runtime is missing.")
        return str(default_runtime)

    raise FileNotFoundError(
        f"Lyra-2 checkpoint directory not found at '{pretrained_model_path}'. "
        "External Lyra-2 source repositories are not accepted at runtime."
    )


def resolve_lyra1_repo_root(pretrained_model_path: str | None) -> str:
    default_runtime = default_local_lyra1_repo()
    if pretrained_model_path is None or str(pretrained_model_path).strip() in {
        "",
        DEFAULT_LYRA1_ALIAS,
        "lyra1",
        "lyra-1",
        "Lyra-1",
        "Lyra",
        "nvidia/Lyra",
    }:
        if default_runtime is None:
            raise FileNotFoundError("Unable to locate the vendored Lyra-1 runtime under WorldFoundry.")
        return str(default_runtime)

    candidate = Path(str(pretrained_model_path)).expanduser()
    if _candidate_lyra1_weights_subdir(candidate) is not None:
        if default_runtime is None:
            raise FileNotFoundError("Lyra-1 checkpoints were found, but the vendored Lyra-1 runtime is missing.")
        return str(default_runtime)

    raise FileNotFoundError(
        f"Lyra-1 checkpoint directory not found at '{pretrained_model_path}'. "
        "External Lyra-1 source repositories are not accepted at runtime."
    )


def _candidate_lyra1_weights_subdir(path_value: Path) -> Path | None:
    candidates = [
        path_value,
        path_value / "Lyra",
        path_value / "lyra-1",
        path_value / "Lyra-1",
        path_value / "nvidia--Lyra",
        path_value / "checkpoints" / "Lyra",
        path_value / "checkpoints" / "lyra-1",
    ]
    for candidate in candidates:
        if (candidate / "lyra_static.pt").is_file() and (candidate / "lyra_dynamic.pt").is_file():
            return candidate.resolve()
    return None


def configure_lyra_runtime_env():
    # TransformerEngine fused attention currently crashes in this Lyra-2 setup on H100.
    # Keep FlashAttention/unfused paths enabled, but disable the fused backend by default.
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")


def resolve_path(path_value: str | None, repo_root: str) -> str | None:
    if path_value is None:
        return None
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(repo_root) / candidate).resolve())


def resolve_checkpoint_root(
    pretrained_model_path: str | None,
    repo_root: str | None = None,
) -> str | None:
    candidates = []
    if pretrained_model_path is not None and str(pretrained_model_path).strip():
        candidates.append(Path(str(pretrained_model_path)).expanduser())
    if repo_root is not None and str(repo_root).strip():
        candidates.append(Path(str(repo_root)).expanduser())

    for candidate in candidates:
        resolved = _candidate_lyra2_checkpoint_subdir(candidate)
        if resolved is not None:
            return str(resolved.resolve())

    default_root = default_local_lyra2_checkpoint_root()
    if default_root is not None:
        return str(default_root)
    return None


def resolve_required_paths(
    repo_root: str,
    checkpoint_dir: str | None = None,
    negative_prompt_path: str | None = None,
    da3_model_path_custom: str | None = None,
    weights_root: str | None = None,
) -> dict[str, str]:
    base_root = weights_root or repo_root
    paths = {
        "checkpoint_dir": resolve_path(checkpoint_dir or "checkpoints/model", base_root),
        "negative_prompt_path": resolve_path(
            negative_prompt_path or "checkpoints/text_encoder/negative_prompt.pt",
            base_root,
        ),
        "da3_model_path_custom": resolve_path(
            da3_model_path_custom or "checkpoints/recon/model.pt",
            base_root,
        ),
    }
    return paths


def ensure_path_exists(path_value: str | None, name: str):
    if path_value is None:
        raise FileNotFoundError(f"{name} is not configured.")
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{name} not found: {path_value}")


def device_index(device: str | None) -> int:
    if not device or not str(device).startswith("cuda"):
        return 0
    if ":" in str(device):
        return int(str(device).split(":", maxsplit=1)[1])
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def maybe_set_cuda_device(device: str | None):
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device_index(device))


@contextmanager
def working_directory(path: str):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
