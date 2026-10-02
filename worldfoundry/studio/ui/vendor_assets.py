"""Provision pinned browser modules used by WorldFoundry Studio.

The modules are downloaded only when this module is explicitly executed.  In
particular, importing Studio and starting a frontend never perform network I/O.
The pinned ``@sparkjsdev/spark`` and ``three`` packages are MIT licensed; see
their exact-version npm package pages for the corresponding license texts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "assets" / "vendor"
VENDOR_ASSET_INSTALL_COMMAND = "python -m worldfoundry.studio.ui.vendor_assets"
# Backward-compatible names used by early source-checkout integrations.
VENDOR_DIR = VENDOR_ROOT
PROVISION_COMMAND = VENDOR_ASSET_INSTALL_COMMAND
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SPARK_MODULE_PATH = VENDOR_ROOT / "spark" / "spark.module.min.js"
THREE_MODULE_PATH = VENDOR_ROOT / "three" / "three.module.js"
THREE_CORE_MODULE_PATH = VENDOR_ROOT / "three" / "three.core.js"


@dataclass(frozen=True)
class VendorAsset:
    """A browser module pinned by URL and content digest."""

    name: str
    package: str
    version: str
    url: str
    sha256: str
    relative_path: Path

    def __post_init__(self) -> None:
        relative_path = Path(self.relative_path)
        if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
            raise ValueError(f"vendor asset path must stay below the vendor root: {relative_path}")
        object.__setattr__(self, "relative_path", relative_path)

    def path_under(self, root: Path) -> Path:
        return Path(root) / self.relative_path


@dataclass(frozen=True)
class VendorAssetStatus:
    """The result of an offline integrity check for one browser module."""

    asset: VendorAsset
    path: Path
    state: str
    actual_sha256: str | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.state == "valid"


class VendorAssetError(RuntimeError):
    """Raised when required browser modules are unavailable or invalid."""


# Spark 0.1.10 imports only the bare ``three`` specifier exposed by Studio's
# import map. Newer Spark builds also require ``three/addons/...`` specifiers.
VENDOR_ASSETS = (
    VendorAsset(
        name="Spark",
        package="@sparkjsdev/spark",
        version="0.1.10",
        url="https://cdn.jsdelivr.net/npm/@sparkjsdev/spark@0.1.10/dist/spark.module.js",
        sha256="e2841904c3facdf2ab5177b13b4827cdc72118cb8b613673ca08d8e983c5bf9d",
        relative_path=SPARK_MODULE_PATH.relative_to(VENDOR_ROOT),
    ),
    VendorAsset(
        name="Three.js module",
        package="three",
        version="0.178.0",
        url="https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.module.js",
        sha256="bc0d236927f5163414e7c59a5567257dfe925f1929ce0a151ac4185dc45ca5a2",
        relative_path=THREE_MODULE_PATH.relative_to(VENDOR_ROOT),
    ),
    VendorAsset(
        name="Three.js core",
        package="three",
        version="0.178.0",
        url="https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.core.js",
        sha256="562b72799ef1145f77997ece49a34f578422873757b0a13e41d76dcbfb776f06",
        relative_path=THREE_CORE_MODULE_PATH.relative_to(VENDOR_ROOT),
    ),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def check_assets(
    root: Path = VENDOR_ROOT,
    assets: Sequence[VendorAsset] = VENDOR_ASSETS,
) -> tuple[VendorAssetStatus, ...]:
    """Check local browser modules without performing network I/O."""

    statuses: list[VendorAssetStatus] = []
    for asset in assets:
        path = asset.path_under(Path(root))
        if not path.is_file():
            statuses.append(VendorAssetStatus(asset=asset, path=path, state="missing"))
            continue
        try:
            actual_sha256 = _file_sha256(path)
        except OSError as exc:
            statuses.append(
                VendorAssetStatus(
                    asset=asset,
                    path=path,
                    state="unreadable",
                    error=str(exc),
                )
            )
            continue
        state = "valid" if actual_sha256 == asset.sha256 else "hash-mismatch"
        statuses.append(
            VendorAssetStatus(
                asset=asset,
                path=path,
                state=state,
                actual_sha256=actual_sha256,
            )
        )
    return tuple(statuses)


def _failure_message(statuses: Sequence[VendorAssetStatus]) -> str:
    details = []
    for status in statuses:
        if status.valid:
            continue
        if status.state == "missing":
            details.append(f"- {status.asset.name}: missing at {status.path}")
        elif status.state == "unreadable":
            details.append(f"- {status.asset.name}: unreadable at {status.path} ({status.error})")
        else:
            details.append(
                f"- {status.asset.name}: SHA-256 mismatch at {status.path} "
                f"(expected {status.asset.sha256}, got {status.actual_sha256})"
            )
    return (
        "Studio browser modules are missing or invalid:\n"
        + "\n".join(details)
        + f"\nRun `{VENDOR_ASSET_INSTALL_COMMAND}` to install the pinned modules."
    )


def require_vendor_assets(
    root: Path = VENDOR_ROOT,
    assets: Sequence[VendorAsset] = VENDOR_ASSETS,
) -> tuple[VendorAssetStatus, ...]:
    """Require valid local modules, without downloading them."""

    statuses = check_assets(root=root, assets=assets)
    if not all(status.valid for status in statuses):
        raise VendorAssetError(_failure_message(statuses))
    return statuses


def _validated_timeout(timeout: float) -> float:
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a finite number greater than zero") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be a finite number greater than zero")
    return value


def _download_asset(
    asset: VendorAsset,
    *,
    root: Path,
    timeout: float,
    opener: Callable[..., Any],
) -> None:
    target = asset.path_under(root)
    temp_path: Path | None = None
    file_descriptor = -1
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
        )
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        downloaded_bytes = 0
        with os.fdopen(file_descriptor, "wb") as output:
            file_descriptor = -1
            with opener(asset.url, timeout=timeout) as response:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_DOWNLOAD_BYTES:
                        raise VendorAssetError(
                            f"Downloaded {asset.package}@{asset.version} exceeded the "
                            f"{MAX_DOWNLOAD_BYTES}-byte safety limit. The target was not replaced."
                        )
                    digest.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != asset.sha256:
            raise VendorAssetError(
                f"Downloaded {asset.package}@{asset.version} failed SHA-256 verification "
                f"(expected {asset.sha256}, got {actual_sha256}). The target was not replaced."
            )
        os.replace(temp_path, target)
        temp_path = None
    except VendorAssetError:
        raise
    except Exception as exc:
        raise VendorAssetError(
            f"Could not download {asset.package}@{asset.version} from {asset.url}: {exc}. "
            "The target was not replaced."
        ) from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def provision_assets(
    root: Path = VENDOR_ROOT,
    assets: Sequence[VendorAsset] = VENDOR_ASSETS,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] | None = None,
) -> tuple[VendorAssetStatus, ...]:
    """Download missing or invalid modules and atomically publish valid bytes."""

    validated_timeout = _validated_timeout(timeout)
    root = Path(root)
    download = opener or request.urlopen
    statuses = check_assets(root=root, assets=assets)
    for status in statuses:
        if status.valid:
            continue
        _download_asset(
            status.asset,
            root=root,
            timeout=validated_timeout,
            opener=download,
        )
    return require_vendor_assets(root=root, assets=assets)


def _print_statuses(statuses: Sequence[VendorAssetStatus]) -> None:
    for status in statuses:
        print(f"{status.state:13} {status.asset.package}@{status.asset.version} -> {status.path}")


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = VENDOR_ROOT,
    opener: Callable[..., Any] | None = None,
) -> int:
    """Command-line entry point for explicit provisioning and offline checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check local files and exit without network access",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=f"vendor directory (default: {VENDOR_ROOT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    args = parser.parse_args(argv)
    destination = args.destination or Path(root)

    try:
        if args.check:
            statuses = check_assets(root=destination)
            _print_statuses(statuses)
            if all(status.valid for status in statuses):
                return 0
            print(_failure_message(statuses), file=sys.stderr)
            return 1

        statuses = provision_assets(root=destination, timeout=args.timeout, opener=opener)
        _print_statuses(statuses)
        return 0
    except (ValueError, VendorAssetError) as exc:
        message = str(exc)
        if VENDOR_ASSET_INSTALL_COMMAND not in message:
            message += f"\nRun `{VENDOR_ASSET_INSTALL_COMMAND}` to retry."
        print(message, file=sys.stderr)
        return 1


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_DOWNLOAD_BYTES",
    "PROVISION_COMMAND",
    "SPARK_MODULE_PATH",
    "THREE_CORE_MODULE_PATH",
    "THREE_MODULE_PATH",
    "VENDOR_ASSETS",
    "VENDOR_ASSET_INSTALL_COMMAND",
    "VENDOR_DIR",
    "VENDOR_ROOT",
    "VendorAsset",
    "VendorAssetError",
    "VendorAssetStatus",
    "check_assets",
    "main",
    "provision_assets",
    "require_vendor_assets",
]


if __name__ == "__main__":
    raise SystemExit(main())
