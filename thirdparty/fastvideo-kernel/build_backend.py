"""Keep standalone kernel sdists independent of the central license symlink."""

from __future__ import annotations

import copy
import gzip
import io
import os
import tarfile
import tempfile
from pathlib import Path

from scikit_build_core import build as _backend

build_wheel = _backend.build_wheel
build_editable = _backend.build_editable
get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
get_requires_for_build_editable = _backend.get_requires_for_build_editable
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _backend.prepare_metadata_for_build_editable


def build_sdist(sdist_directory: str, config_settings: dict | None = None) -> str:
    """Embed the same notice bytes as a regular file, without changing sources."""
    filename = _backend.build_sdist(sdist_directory, config_settings)
    archive_path = Path(sdist_directory) / filename
    notice = Path(__file__).with_name("LICENSE").read_bytes()
    descriptor, temporary = tempfile.mkstemp(dir=archive_path.parent, suffix=".tar.gz")
    try:
        with os.fdopen(descriptor, "wb") as stream, tarfile.open(archive_path, "r:gz") as source:
            epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "1667997441"))
            with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                    found = False
                    for entry in source:
                        if entry.name.partition("/")[2] == "LICENSE":
                            entry = copy.copy(entry)
                            entry.type = tarfile.REGTYPE
                            entry.linkname = ""
                            entry.size = len(notice)
                            target.addfile(entry, io.BytesIO(notice))
                            found = True
                        else:
                            target.addfile(entry, source.extractfile(entry) if entry.isfile() else None)
                    if not found:
                        raise RuntimeError("native kernel sdist omitted the central license link")
        os.replace(temporary, archive_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return filename
