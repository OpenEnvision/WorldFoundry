from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_WORLDFM_REPO = "inspatio/worldfm"
EXPECTED_WORLDFM_FILES = (
    "worldfm_1-step.pth",
    "worldfm_2-step.pth",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


@dataclass(frozen=True)
class WorldFMAssetResolution:
    """Resolved local WorldFM synthesis assets.

    Args:
        checkpoint_path: Local WorldFM checkpoint file used by the diffusion model.
        vae_path: Local Diffusers AutoencoderKL directory.
    """

    checkpoint_path: Path
    vae_path: Path


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).resolve().parents[7]


def _is_vae_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (
            (path / "diffusion_pytorch_model.safetensors").is_file() or (path / "diffusion_pytorch_model.bin").is_file()
        )
    )


def _candidate_roots(source: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if source and str(source) not in {DEFAULT_WORLDFM_REPO, "worldfm"}:
        roots.append(Path(source).expanduser())
    roots.append(_project_root() / "cache" / "hfd" / "worldfm")
    return roots


def _find_checkpoint(root: Path, expected_checkpoint: str) -> Path | None:
    if root.is_file() and root.name == expected_checkpoint:
        return root.resolve()
    if not root.exists():
        return None
    for candidate_root in (root, root / "weights"):
        candidate = candidate_root / expected_checkpoint
        if candidate.is_file():
            return candidate.resolve()
    return None


def _find_vae(root: Path) -> Path | None:
    if _is_vae_dir(root):
        return root.resolve()
    candidate = root / "vae"
    if _is_vae_dir(candidate):
        return candidate.resolve()
    return None


def missing_worldfm_asset_message(
    checkpoint_source: str | Path | None,
    *,
    step: int,
    checkpoint_filename: str | None = None,
) -> str:
    """Build the explicit local-asset requirement message.

    Args:
        checkpoint_source: Caller-provided checkpoint file or model directory.
        step: DMD step count used to choose the default checkpoint filename.
        checkpoint_filename: Optional checkpoint filename override.
    """

    expected_checkpoint = checkpoint_filename or f"worldfm_{int(step)}-step.pth"
    searched_roots = [str(path) for path in _candidate_roots(checkpoint_source)]
    return (
        "WorldFM synthesis is configured for strict in-tree execution and cannot download "
        f"`{DEFAULT_WORLDFM_REPO}` at runtime. Provide a local directory containing "
        f"`{expected_checkpoint}` plus `vae/config.json` and a VAE weight file, or pass "
        "`checkpoint_filename`/`vae_path` explicitly. Searched: " + ", ".join(searched_roots)
    )


def resolve_worldfm_assets(
    checkpoint_source: str | Path | None,
    vae_source: str | Path | None,
    *,
    step: int,
    checkpoint_filename: str | None = None,
    allow_missing: bool = False,
) -> WorldFMAssetResolution | None:
    """Resolve WorldFM assets from local paths only.

    Args:
        checkpoint_source: Local checkpoint file or asset directory.
        vae_source: Optional local VAE directory or parent asset directory.
        step: DMD step count used to choose the default checkpoint filename.
        checkpoint_filename: Optional checkpoint filename override.
        allow_missing: Return ``None`` instead of raising when assets are absent.
    """

    expected_checkpoint = checkpoint_filename or f"worldfm_{int(step)}-step.pth"
    checkpoint_path: Path | None = None
    checkpoint_roots = _candidate_roots(checkpoint_source)
    for root in checkpoint_roots:
        checkpoint_path = _find_checkpoint(root, expected_checkpoint)
        if checkpoint_path is not None:
            break

    vae_roots = [Path(vae_source).expanduser()] if vae_source else []
    vae_roots.extend(checkpoint_roots)
    vae_path: Path | None = None
    for root in vae_roots:
        vae_path = _find_vae(root)
        if vae_path is not None:
            break

    if checkpoint_path is not None and vae_path is not None:
        return WorldFMAssetResolution(checkpoint_path=checkpoint_path, vae_path=vae_path)
    if allow_missing:
        return None
    raise FileNotFoundError(
        missing_worldfm_asset_message(
            checkpoint_source,
            step=step,
            checkpoint_filename=checkpoint_filename,
        )
    )


def load_worldfm_checkpoint(model_path: str | Path) -> Any:
    """Load a local WorldFM checkpoint file.

    Args:
        model_path: Local ``.pth`` checkpoint path.
    """

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WorldFM checkpoint file does not exist: {path}")
    return torch.load(path, map_location=lambda storage, loc: storage)
