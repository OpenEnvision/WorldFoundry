from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.setup.check_packaging_license_gate import (
    REQUIRED_WHEEL_PACKAGES,
    audit_sdist,
    audit_wheel,
    dead_exclude_patterns,
    discover_packages,
    leaked_packages,
    license_gated_paths,
    load_find_config,
    main,
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
