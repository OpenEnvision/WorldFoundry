from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.setup.check_packaging_license_gate import (
    REQUIRED_RUNTIME_RESOURCES,
    REQUIRED_WHEEL_PACKAGES,
    audit_license_bundle,
    audit_runtime_resources,
    audit_sdist,
    audit_wheel,
    dead_exclude_patterns,
    discover_packages,
    leaked_packages,
    license_bundle_errors,
    license_gated_paths,
    load_find_config,
    main,
    misplaced_license_files,
    missing_core_sources,
    missing_required_wheel_packages,
    package_data_exclusion_gaps,
    private_file,
    selected_packages,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_discovery_honors_namespace_mode_without_walking_sibling_trees(tmp_path: Path) -> None:
    package_root = tmp_path / "worldfoundry"
    (package_root / "namespace_only" / "child").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    config = {"where": ["."], "namespaces": True, "include": ["worldfoundry*"]}

    assert discover_packages(config, repo_root=tmp_path) == [
        "worldfoundry",
        "worldfoundry.namespace_only",
        "worldfoundry.namespace_only.child",
    ]

    config["namespaces"] = False
    assert discover_packages(config, repo_root=tmp_path) == ["worldfoundry"]


def test_package_set_helpers_detect_dead_rules_and_gated_leaks() -> None:
    packages = ["worldfoundry", "worldfoundry.wrapper", "worldfoundry.wrapper.gated"]
    excludes = ["worldfoundry.wrapper.gated*"]

    assert dead_exclude_patterns(packages, excludes) == []
    assert dead_exclude_patterns(packages, ["worldfoundry.missing*"]) == [
        "worldfoundry.missing*"
    ]
    assert selected_packages(packages, excludes) == ["worldfoundry", "worldfoundry.wrapper"]
    assert leaked_packages(packages, ["worldfoundry.wrapper.gated"]) == [
        "worldfoundry.wrapper.gated"
    ]


def test_data_channel_gate_requires_an_exact_retained_parent() -> None:
    gated = ["worldfoundry/wrapper/gated/runtime"]
    kept = ["worldfoundry", "worldfoundry.wrapper"]

    assert package_data_exclusion_gaps(gated, kept, {}) == [
        "worldfoundry/wrapper/gated/runtime (parent 'worldfoundry.wrapper', "
        "relative path 'gated/runtime')"
    ]
    assert package_data_exclusion_gaps(
        gated,
        kept,
        {"worldfoundry.wrapper": ["gated/**/*"]},
    ) == []


def _write_wheel(path: Path, entries: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for entry in entries:
            wheel.writestr(entry, "")


def test_built_wheel_audit_checks_gated_files_and_first_party_wrapper(tmp_path: Path) -> None:
    wrapper_entry = REQUIRED_WHEEL_PACKAGES[0].replace(".", "/") + "/__init__.py"
    clean_wheel = tmp_path / "clean.whl"
    _write_wheel(clean_wheel, [wrapper_entry, "worldfoundry/__init__.py"])

    assert audit_wheel(clean_wheel, ["worldfoundry/wrapper/gated"]) == []
    assert missing_required_wheel_packages(clean_wheel) == []

    leaking_wheel = tmp_path / "leaking.whl"
    gated_entry = "worldfoundry/wrapper/gated/runtime.py"
    _write_wheel(leaking_wheel, [gated_entry])

    assert audit_wheel(leaking_wheel, ["worldfoundry/wrapper/gated"]) == [gated_entry]
    assert missing_required_wheel_packages(leaking_wheel) == list(REQUIRED_WHEEL_PACKAGES)


@pytest.mark.parametrize("artifact_name", ["worldfoundry.whl", "worldfoundry.tar.gz"])
def test_distribution_audit_rejects_missing_core_implementation(tmp_path: Path, artifact_name: str) -> None:
    source_root = tmp_path / "source"
    package = "worldfoundry/core/model_loading/checkpoints"
    core_dir = source_root / package
    core_dir.mkdir(parents=True)
    (core_dir / "__init__.py").write_text("", encoding="utf-8")
    (core_dir / "file.py").write_text("", encoding="utf-8")
    (core_dir / "__pycache__").mkdir()
    (core_dir / "__pycache__/ignored.py").write_text("", encoding="utf-8")
    artifact = tmp_path / artifact_name

    def write_artifact(entries: list[str]) -> None:
        if artifact.suffix == ".whl":
            _write_wheel(artifact, entries)
        else:
            with tarfile.open(artifact, "w:gz") as archive:
                for entry in entries:
                    archive.addfile(tarfile.TarInfo("worldfoundry-0.0.0/" + entry), io.BytesIO())

    entries = [f"{package}/__init__.py"]
    write_artifact(entries)
    assert missing_core_sources(artifact, repo_root=source_root) == [f"{package}/file.py"]

    write_artifact([*entries, f"{package}/file.py"])
    assert missing_core_sources(artifact, repo_root=source_root) == []


def test_repository_license_gate_uses_narrow_hunyuan_excludes() -> None:
    find_config = load_find_config()
    excludes = set(find_config["exclude"])
    gated_paths = set(license_gated_paths())

    assert find_config["namespaces"] is True
    assert "worldfoundry.synthesis.visual_generation.hunyuan_world" not in excludes
    assert "worldfoundry.synthesis.visual_generation.hunyuan_world.*" not in excludes
    for subtree in ("hunyuan_game_craft", "hunyuan_world_voyager",
                    "hy_world_2p0_panogen_runtime", "hy_world_2p0_worldgen_runtime"):
        package = f"worldfoundry.synthesis.visual_generation.hunyuan_world.{subtree}"
        assert selected_packages([package, f"{package}.child"], sorted(excludes)) == []
        assert package.replace(".", "/") in gated_paths


def test_repository_static_packaging_license_gate() -> None:
    assert main([]) == 0


def test_ci_builds_and_audits_clean_distributions() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "packaging-license-gate:" in workflow
    assert 'python -m build --outdir "$audit_dir"' in workflow
    assert 'make packaging-check WHEEL="$wheel_path" SDIST="$1"' in workflow


@pytest.mark.parametrize("path", [
    "tests/conftest.py", "tests/README.md", "tests/run_tests_docker.sh",
    "tests/training/test_checkpoint.py", "tests/fixtures/request.json",
    "tests/fixtures/catalog.yaml", "tests/fixtures/results.jsonl",
])
def test_first_party_test_sources_are_public(path: str) -> None:
    assert not private_file(path)


@pytest.mark.parametrize("path", [
    "worldfoundry/training/runner.py", "scripts/training/run.py",
    "worldfoundry/runtime/tests/test_local.py", "test/test_old.py",
    "tests/__pycache__/test_jobs.pyc", "tests/fixtures/weights.safetensors",
    "tests/fixtures/private.key", "tests/logs/events.jsonl", "tests/.env.json",
])
def test_public_tests_do_not_open_private_sources_or_artifacts(path: str) -> None:
    assert private_file(path)


def test_public_test_sources_remain_outside_wheels(tmp_path: Path) -> None:
    wheel = tmp_path / "tests.whl"
    _write_wheel(wheel, ["tests/test_public.py", "worldfoundry/__init__.py"])
    assert audit_wheel(wheel, []) == ["tests/test_public.py"]


def test_source_artifact_accepts_tests_and_rejects_private_code(tmp_path: Path) -> None:
    sdist = tmp_path / "source.tar.gz"
    entries = [
        "tests/test_public.py", "tests/training/test_checkpoint.py", "tests/fixtures/request.json",
        "worldfoundry/training/runner.py", "worldfoundry/wrapper/gated/runtime.py",
        "tests/__pycache__/test_public.pyc", "tests/fixtures/weights.pt",
    ]
    with tarfile.open(sdist, "w:gz") as archive:
        for entry in entries:
            data = b"fixture"
            info = tarfile.TarInfo("worldfoundry-0.0.0/" + entry)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    assert audit_sdist(sdist, ["worldfoundry/wrapper/gated"]) == sorted(entries[3:])


@pytest.mark.parametrize("damage", ["changed_terms", "truncated_text", "missing_records"])
def test_license_bundle_rejects_lost_or_modified_terms(damage: str) -> None:
    notice = (REPO_ROOT / "THIRD-PARTY-NOTICES").read_bytes()
    start = notice.index(b"----- BEGIN RETAINED LICENSE TEXT -----\n")
    if damage == "changed_terms":
        notice = notice[:start] + notice[start:].replace(b"Grant of Copyright License", b"Grant of Copyright Licenxe", 1)
    elif damage == "truncated_text":
        notice = notice[:start + 100]
    else:
        notice = notice.replace(b"----- BEGIN RETAINED LICENSE TEXT -----\n", b"", 1)
    assert license_bundle_errors(notice)


@pytest.mark.parametrize("format_name", ["wheel", "sdist"])
@pytest.mark.parametrize("contents", [None, b"Incomplete license", b"Complete source notice"])
def test_distribution_requires_exact_central_license_bundle(
    tmp_path: Path, format_name: str, contents: bytes | None,
) -> None:
    expected = b"Complete source notice"
    suffix = ".whl" if format_name == "wheel" else ".tar.gz"
    artifact = tmp_path / ("worldfoundry" + suffix)
    if format_name == "wheel":
        with zipfile.ZipFile(artifact, "w") as archive:
            if contents is not None:
                archive.writestr("worldfoundry-0.0.0.dist-info/licenses/THIRD-PARTY-NOTICES", contents)
    else:
        with tarfile.open(artifact, "w:gz") as archive:
            if contents is not None:
                member = tarfile.TarInfo("worldfoundry-0.0.0/THIRD-PARTY-NOTICES")
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
    assert bool(audit_license_bundle(artifact, expected)) == (contents != expected)


@pytest.mark.parametrize("format_name", ["wheel", "sdist"])
@pytest.mark.parametrize("damage", ["none", "missing", "changed"])
def test_model_runtime_resource_gate_detects_missing_or_changed_statistics(
    tmp_path: Path, format_name: str, damage: str,
) -> None:
    payloads = {name: (REPO_ROOT / name).read_bytes() for name in REQUIRED_RUNTIME_RESOURCES}
    statistics = next(name for name in payloads if name.endswith("nav_25dof_stats.json"))
    if damage == "missing":
        payloads.pop(statistics)
    elif damage == "changed":
        changed = json.loads(payloads[statistics])
        changed["state_stats"]["min"][0] += 0.01
        payloads[statistics] = json.dumps(changed).encode()
    artifact = tmp_path / ("worldfoundry.whl" if format_name == "wheel" else "worldfoundry.tar.gz")
    if format_name == "wheel":
        with zipfile.ZipFile(artifact, "w") as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
    else:
        with tarfile.open(artifact, "w:gz") as archive:
            for name, data in payloads.items():
                member = tarfile.TarInfo("worldfoundry-0.0.0/" + name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    assert bool(audit_runtime_resources(artifact)) == (damage != "none")


def test_native_build_license_is_a_link_to_the_single_bundle(tmp_path: Path) -> None:
    (tmp_path / "THIRD-PARTY-NOTICES").write_bytes(b"Full notice")
    native = tmp_path / "thirdparty/fastvideo-kernel/LICENSE"
    native.parent.mkdir(parents=True)
    native.symlink_to("../../THIRD-PARTY-NOTICES")
    misplaced = "worldfoundry/component/COPYING.MPL2"
    other = tmp_path / misplaced
    other.parent.mkdir(parents=True)
    other.write_bytes(b"Separate license")
    assert misplaced_license_files([str(native.relative_to(tmp_path)), misplaced], repo_root=tmp_path) == [misplaced]
    native.unlink()
    native.symlink_to("missing-license")
    assert misplaced_license_files([str(native.relative_to(tmp_path))], repo_root=tmp_path) == [
        "thirdparty/fastvideo-kernel/LICENSE"
    ]


# Original texts audited at 36ab841d, before consolidation. These independent
# digests catch removal or alteration even if a bundle record is rewritten.
@pytest.mark.parametrize("original_path, expected_sha256", [
    ('LICENSE', 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'),
    ('thirdparty/THIRD_PARTY_LICENSES.md', 'a492fa1695bbfa37d9ad334154be36555a063f1dfd1408eb9fd4fa2ab0b20e32'),
    ('thirdparty/fastvideo-kernel/LICENSE', '5c7f173199fd7fb3cc83d86d24f3541e8ae0cb8c16e912ca519ed6a1435bd8f3'),
    ('worldfoundry/base_models/diffusion_model/models/networks/hunyuan_video/gamecraft/LICENSE', '4d6124d3c683dcc0c9e5ec10bb6c85e66ed42983a2988e457256efbfda060104'),
    ('worldfoundry/base_models/diffusion_model/models/networks/hunyuan_video/h15/LICENSE-HUNYUAN', '343d271e1c78188d5ed982c1ec8b237e982f4a92dcb65573a6c425fcab5037e9'),
    ('worldfoundry/base_models/diffusion_model/models/networks/hunyuan_video/h15/LICENSE-MINWM', 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'),
    ('worldfoundry/base_models/three_dimensions/depth/unidepth/LICENSE', 'db2e35513dbadcdc67f5819a3bfee2777786538dd3531620cd5fbd4b6ed6e538'),
    ('worldfoundry/base_models/three_dimensions/general_3d/mapanything/CHECKPOINTS_NOTICE.md', 'fdcf5948059d6524414d71edf53f1454a3fe007f9369fc48b1db1241797b2c13'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/LICENSE', 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/LICENSE', 'bf7bf417f7ecc32b5cc0b479cd39e627dde499ff4d47ed8796aee44556cbd5b6'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.APACHE', '03379001a7b12a2ec997a25554247d985270b353c10d5bafee9ac8d6519820b7'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.BSD', '51928dce36213c5333ba3172e847d735d4c6e9b7ff2722a326c49067155b82eb'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.GPL', '8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.LGPL', 'dc626520dcd53a22f727af3ee42c770e56c97a64fe3adb063799d8ab032fe551'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.MINPACK', 'c87b7f8ee88f6195e91743820c00354833583aef091b72e2d4a49c8e28e798a0'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.MPL2', 'fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/eigen/COPYING.README', 'c83230b770f17ef1386ea1fd3681271dd98aa93646bdbfb5bff3a1b7050fff9d'),
    ('worldfoundry/base_models/three_dimensions/slam/mega_sam_runtime/base/thirdparty/lietorch/LICENSE', '4261dca112b565843598e0db554ff90e578701fca983262bd4f5a7ce67d81f89'),
    ('worldfoundry/synthesis/visual_generation/hunyuan_world/gamecraft_inference/LICENSE', '4d6124d3c683dcc0c9e5ec10bb6c85e66ed42983a2988e457256efbfda060104'),
    ('worldfoundry/synthesis/visual_generation/inspatio_world/inspatio_world_runtime/inference_inputs/LICENSE-InSpatio', 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'),
    ('worldfoundry/synthesis/visual_generation/open_sora/open_sora_runtime/LICENSE', '9c6a5e20e18f634012e402cf0c801c9620075ca65ca96dbb16f46433a81beca9'),
    ('worldfoundry/synthesis/visual_generation/world_model/dino_wm/LICENSE.upstream', '31ee2e6efe0d638e9685d06f6267ae69c9456bbaf1fea7b3c716dee6b79c7ab5'),
])
def test_consolidation_preserves_original_license_bytes(
    original_path: str, expected_sha256: str,
) -> None:
    notice = (REPO_ROOT / "THIRD-PARTY-NOTICES").read_bytes()
    tail = notice.split(b"Original-File: " + original_path.encode() + b"\n", 1)[1]
    record = re.search(rb"SHA256: [0-9a-f]{64}\nBytes: ([0-9]+)\n----- BEGIN RETAINED LICENSE TEXT -----\n", tail)
    assert record is not None
    original = tail[record.end():record.end() + int(record.group(1))]
    assert hashlib.sha256(original).hexdigest() == expected_sha256
