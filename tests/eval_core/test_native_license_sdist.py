"""Exercise native source packaging without Torch, compilers, or scikit-build."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("source_link", [True, False])
def test_native_sdist_license_survives_standalone_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_link: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    backend_source = repo_root / "thirdparty/fastvideo-kernel/build_backend.py"
    project = tmp_path / "repo/thirdparty/fastvideo-kernel"
    project.mkdir(parents=True)
    notice = b"Complete copyright and license terms\n"
    (tmp_path / "repo/THIRD-PARTY-NOTICES").write_bytes(notice)
    license_path = project / "LICENSE"
    license_path.symlink_to("../../THIRD-PARTY-NOTICES")
    backend_path = project / "build_backend.py"
    backend_path.write_bytes(backend_source.read_bytes())

    def build_sdist(directory: str, config: dict | None) -> str:
        filename = "fastvideo_kernel-0.3.2.tar.gz"
        with tarfile.open(Path(directory) / filename, "w:gz") as archive:
            entry = tarfile.TarInfo("fastvideo_kernel-0.3.2/LICENSE")
            if source_link:
                entry.type = tarfile.SYMTYPE
                entry.linkname = "../../THIRD-PARTY-NOTICES"
                archive.addfile(entry)
            else:
                entry.size = len(notice)
                archive.addfile(entry, io.BytesIO(notice))
            code = b"package source\n"
            source = tarfile.TarInfo("fastvideo_kernel-0.3.2/source.py")
            source.size = len(code)
            archive.addfile(source, io.BytesIO(code))
        return filename

    backend = types.ModuleType("scikit_build_core.build")
    for name in (
        "build_wheel", "build_editable", "get_requires_for_build_wheel",
        "get_requires_for_build_editable", "get_requires_for_build_sdist",
        "prepare_metadata_for_build_wheel", "prepare_metadata_for_build_editable",
    ):
        setattr(backend, name, lambda *args, **kwargs: None)
    backend.build_sdist = build_sdist
    package = types.ModuleType("scikit_build_core")
    package.build = backend
    monkeypatch.setitem(sys.modules, "scikit_build_core", package)
    monkeypatch.setitem(sys.modules, "scikit_build_core.build", backend)
    spec = importlib.util.spec_from_file_location("native_license_backend", backend_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filename = module.build_sdist(str(tmp_path))
    with tarfile.open(tmp_path / filename, "r:gz") as archive:
        member = archive.getmember("fastvideo_kernel-0.3.2/LICENSE")
        assert member.isfile()
        assert not member.issym()
        assert archive.extractfile(member).read() == notice
        assert archive.extractfile("fastvideo_kernel-0.3.2/source.py").read() == b"package source\n"
        archive.extractall(tmp_path / "standalone")
    assert (tmp_path / "standalone/fastvideo_kernel-0.3.2/LICENSE").read_bytes() == notice
    assert license_path.is_symlink()
