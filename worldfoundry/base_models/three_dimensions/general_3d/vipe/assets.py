"""Checkpoint resolution and explicit download preparation for ViPE."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock

from worldfoundry.core.io.assets.download import download_to_cache
from worldfoundry.core.io.paths import checkpoint_root_path

DROID_GOOGLE_DRIVE_ID = "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"
GEOCALIB_PINHOLE_URL = "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar"


@dataclass(frozen=True, slots=True)
class VipeAsset:
    """One required ViPE pose-inference checkpoint."""

    name: str
    env_var: str
    path: Path
    source: str

    @property
    def available(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["available"] = self.available
        return result


def _legacy_torch_hub_path(*parts: str) -> Path:
    try:
        import torch

        return Path(torch.hub.get_dir()).joinpath(*parts)
    except Exception:
        return Path.home() / ".cache" / "torch" / "hub" / Path(*parts)


def _resolve_asset(env_var: str, standard_name: str, legacy_path: Path, source: str) -> VipeAsset:
    explicit = os.environ.get(env_var)
    standard_path = checkpoint_root_path("vipe", standard_name)
    if explicit:
        path = Path(explicit).expanduser()
    elif standard_path.is_file():
        path = standard_path
    elif legacy_path.is_file():
        path = legacy_path
    else:
        path = standard_path
    return VipeAsset(name=standard_name, env_var=env_var, path=path, source=source)


def droid_checkpoint() -> VipeAsset:
    return _resolve_asset(
        "WORLDFOUNDRY_VIPE_DROID_CHECKPOINT",
        "droid.pth",
        _legacy_torch_hub_path("droid_slam", "droid.pth"),
        f"Google Drive file id {DROID_GOOGLE_DRIVE_ID}",
    )


def geocalib_checkpoint() -> VipeAsset:
    return _resolve_asset(
        "WORLDFOUNDRY_VIPE_GEOCALIB_CHECKPOINT",
        "geocalib-pinhole.tar",
        _legacy_torch_hub_path("geocalib", "pinhole.tar"),
        GEOCALIB_PINHOLE_URL,
    )


def required_pose_assets() -> tuple[VipeAsset, VipeAsset]:
    return droid_checkpoint(), geocalib_checkpoint()


def require_asset(asset: VipeAsset) -> Path:
    """Return an existing asset path or fail with its exact preparation contract."""
    if not asset.available:
        raise FileNotFoundError(
            f"Missing ViPE checkpoint {asset.name}. Set {asset.env_var}, place the file at {asset.path}, "
            f"or run `python -m worldfoundry.base_models.three_dimensions.general_3d.vipe "
            f"prepare-assets --download`. Official source: {asset.source}."
        )
    _validate_torch_checkpoint(asset.path)
    return asset.path


def _validate_torch_checkpoint(path: Path) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Checkpoint {path} is not a non-empty state-dict container.")


def _download_droid(target: Path) -> Path:
    import gdown

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target) + ".lock")
    with lock:
        if target.is_file():
            _validate_torch_checkpoint(target)
            return target
        temporary = target.with_name(f".{target.name}.part")
        temporary.unlink(missing_ok=True)
        try:
            result = gdown.download(id=DROID_GOOGLE_DRIVE_ID, output=str(temporary), quiet=False)
            if result is None or not temporary.is_file():
                raise RuntimeError(f"gdown did not produce {temporary}")
            _validate_torch_checkpoint(temporary)
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return target


def prepare_pose_assets(*, download: bool = False) -> tuple[VipeAsset, VipeAsset]:
    """Validate required checkpoints and optionally download missing public assets."""
    assets = required_pose_assets()
    if download:
        droid, geocalib = assets
        if not droid.available:
            _download_droid(droid.path)
        if not geocalib.available:
            download_to_cache(
                GEOCALIB_PINHOLE_URL,
                cache_dir=geocalib.path.parent,
                filename=geocalib.path.name,
                validator=_validate_torch_checkpoint,
                timeout=120.0,
            )
        assets = required_pose_assets()

    missing = [asset for asset in assets if not asset.available]
    if missing:
        details = "; ".join(
            f"{asset.name}: set {asset.env_var} or place it at {asset.path} (source: {asset.source})"
            for asset in missing
        )
        raise FileNotFoundError(
            "ViPE pose inference checkpoints are missing. "
            f"{details}. To fetch the public assets explicitly, run "
            "`python -m worldfoundry.base_models.three_dimensions.general_3d.vipe prepare-assets --download`."
        )

    for asset in assets:
        _validate_torch_checkpoint(asset.path)
    return assets


__all__ = [
    "DROID_GOOGLE_DRIVE_ID",
    "GEOCALIB_PINHOLE_URL",
    "VipeAsset",
    "droid_checkpoint",
    "geocalib_checkpoint",
    "prepare_pose_assets",
    "require_asset",
    "required_pose_assets",
]
