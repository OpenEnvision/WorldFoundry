from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# NOTE: the sanitized-sdist builder tests that used to live in this module were
# removed together with the tools/packaging toolchain (build_sanitized_sdist,
# check_release_worktree, check_sdist_hygiene no longer exist in the repo).
# Only the packaging-metadata contract tests below remain.

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 only.
    tomllib = pytest.importorskip("tomli")


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _optional_dependencies() -> dict[str, list[str]]:
    return _pyproject()["project"]["optional-dependencies"]


def _package_data() -> dict[str, list[str]]:
    return _pyproject()["tool"]["setuptools"]["package-data"]


def _requirement_names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(requirement).name) for requirement in requirements}


def test_data_gpu_probe_dependencies_are_packaged() -> None:
    optional = _optional_dependencies()

    for extra in ("video", "metrics", "all"):
        assert "pyarrow" in optional[extra]
        assert "h5py" in optional[extra]


def test_ui_extra_covers_realtime_dependencies_and_pins_gradio_5() -> None:
    optional = _optional_dependencies()

    assert "gradio>=5.50,<6" in optional["ui"]
    assert "gradio>=5.50,<6" in optional["all"]
    assert _requirement_names(optional["studio_realtime"]) <= _requirement_names(optional["ui"])


def test_lint_tooling_uses_one_exact_ruff_version() -> None:
    payload = _pyproject()
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ruff==0.12.7" in payload["project"]["optional-dependencies"]["dev"]
    assert "ruff==0.12.7" in payload["dependency-groups"]["lint"]
    assert 'python -m pip install PyYAML "ruff==0.12.7"' in workflow
    assert "pytest>=8" in payload["project"]["optional-dependencies"]["dev"]


def test_make_install_dev_uses_the_packaging_extra() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    target = re.search(
        r"(?m)^install-dev\s*:[^\n]*\n(?P<recipe>(?:\t[^\n]*(?:\n|$))+)",
        makefile,
    )

    assert target is not None, "Makefile must define an install-dev recipe"
    recipe = [line.removeprefix("\t") for line in target.group("recipe").splitlines()]
    assert recipe == ['$(PIP) install -e ".[dev]"']


def test_public_cpu_ci_uses_test_extra_without_torch() -> None:
    payload = _pyproject()
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    eval_core = payload["project"]["optional-dependencies"]["eval_core"]
    test = payload["project"]["optional-dependencies"]["test"]

    assert {"numpy", "pytest", "setuptools", "tomli", "torch", "torchvision"} <= _requirement_names(eval_core)
    assert "tomli>=2; python_version < '3.11'" in eval_core
    assert {"numpy", "pytest", "setuptools", "tomli"} <= _requirement_names(test)
    assert not {"torch", "torchvision"} & _requirement_names(test)
    assert 'python -m pip install -e ".[test]"' in workflow
    assert "make test" in workflow
    assert 'python -m pip install "${wheel_path}[test]"' in workflow


def test_environment_reference_documents_all_dependency_surfaces() -> None:
    payload = _pyproject()
    reference_paths = (
        REPO_ROOT / "docs/fumadocs/content/docs/reference/environments.mdx",
        REPO_ROOT / "docs/fumadocs/content/docs/reference/environments.zh.mdx",
    )

    for path in reference_paths:
        reference = path.read_text(encoding="utf-8")
        for name in payload["project"]["optional-dependencies"]:
            assert f"<code>{name}</code>" in reference, f"{path.name} does not document extra {name!r}"
        for name in payload["dependency-groups"]:
            assert f"- `{name}`" in reference, f"{path.name} does not document group {name!r}"


def test_studio_static_assets_are_packaged() -> None:
    package_data = _package_data()
    excluded_package_data = _pyproject()["tool"]["setuptools"]["exclude-package-data"]

    assert "assets/**/*" in package_data["worldfoundry.studio"]
    assert (REPO_ROOT / "worldfoundry/studio/assets").is_dir()
    assert "ui/*.html" in package_data["worldfoundry.studio"]
    assert "assets/vendor/**/*" in excluded_package_data["*"]
    assert (REPO_ROOT / "worldfoundry/studio/ui/vendor_assets.py").is_file()


def test_unified_requirements_install_data_gpu_probe_dependencies() -> None:
    requirements = (REPO_ROOT / "requirements" / "worldfoundry-unified.txt").read_text(encoding="utf-8")
    install_script = (REPO_ROOT / "scripts" / "setup" / "conda_install.sh").read_text(encoding="utf-8")

    assert "build" in requirements
    assert (
        "-e .[tui,optimized_core,video,hf,api,ui,metrics,studio_pointcloud,studio_rerun]"
        in requirements
    )
    assert "stable-worldmodel[train]" in requirements
    for package in ("h5py", "numpy", "opencv-python", "pillow", "pyarrow"):
        assert package in requirements
    for module in ('"cv2"', '"h5py"', '"numpy"', '"PIL"', '"pyarrow"'):
        assert module in install_script


def test_optimized_core_extra_declares_flashdreams_runtime_dependencies() -> None:
    optional = _optional_dependencies()

    assert "optimized_core" in optional
    assert "boto3" in optional["optimized_core"]
    assert "einops>=0.8.0,<0.9.0" in optional["optimized_core"]
    assert "filelock>=3.24.2,<4.0.0" in optional["optimized_core"]
    assert "huggingface-hub" in optional["optimized_core"]
    assert "loguru" in optional["optimized_core"]
    assert "safetensors>=0.4" in optional["optimized_core"]
    assert "torch>=2.7,<2.12.0" in optional["optimized_core"]


def test_sdist_manifest_excludes_generated_downloaded_and_large_artifacts() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    # Model runtime profiles moved from data/models/runtime_profiles to
    # data/models/runtime (+ environments/), and the cosmos vendored trees are
    # covered by the global binary/notebook excludes instead of per-path prunes.
    required_snippets = (
        "include worldfoundry/data/benchmarks/*.md",
        "recursive-include worldfoundry/data/benchmarks/catalog *.yaml",
        "recursive-include worldfoundry/data/benchmarks/runtime_profiles *.yaml",
        "recursive-include worldfoundry/data/models/catalog *.yaml",
        "recursive-include worldfoundry/data/models/runtime *.yaml",
        "recursive-include worldfoundry/data/models/bindings *.yaml",
        "recursive-include worldfoundry/data/models/runtime/configs *.yaml",
        "recursive-include worldfoundry/data/models/runtime/configs *.yml",
        "recursive-include worldfoundry/data/models/runtime/configs *.json",
        "recursive-include worldfoundry/data/models/runtime/environments *.yaml",
        "prune tmp",
        "prune cache",
        "prune data/hfd_datasets",
        "prune worldfoundry/data/test_cases",
        "prune docs/fumadocs/.next",
        "prune docs/fumadocs/node_modules",
        "prune worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/eval",
        "prune worldfoundry/synthesis/visual_generation/pandora/pandora_runtime/ChatUniVi/train",
        "prune worldfoundry/synthesis/visual_generation/dynamicrafter_pandora/DynamiCrafter/assets",
        "prune worldfoundry/synthesis/visual_generation/dynamicrafter_pandora/DynamiCrafter/prompts",
        "global-exclude *.py[cod]",
    )
    for snippet in required_snippets:
        assert snippet in manifest
    for pattern in (
        "*.gif",
        "*.ipynb",
        "*.pdf",
        "*.mp4",
        "*.safetensors",
        "*.npy",
        "*.pt",
        "*.pth",
        "*.onnx",
    ):
        assert pattern in manifest
