"""The YAML capability registry loads strictly and expands templates and nested stacks."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.base_models import capabilities
from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES, CapabilityRegistryError


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def registry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(capabilities, "REGISTRY_DIR", tmp_path)
    return tmp_path


@pytest.mark.unit
def test_plain_capability_defaults_owner_env_and_packages(registry_dir: Path) -> None:
    _write(
        registry_dir / "capabilities" / "models.yaml",
        """
demo:
  family: depth
  canonical_path: worldfoundry/base_models/three_dimensions/depth/vigeo
  package_imports: [torch, cv2]
  assets:
    - {id: demo_ckpt, kind: file, role: checkpoint, env: [DEMO_CKPT, DEMO_CKPT_ALT], local_path: x.pt}
""",
    )
    capability = capabilities._load_capabilities(registry_dir / "capabilities")["demo"]

    assert capability.canonical_owner == "worldfoundry.base_models.three_dimensions.depth.vigeo"
    assert capability.asset_env == ("DEMO_CKPT", "DEMO_CKPT_ALT")
    assert capability.pip_package_by_import() == {"torch": "torch", "cv2": "cv2"}
    assert capability.assets[0].env == ("DEMO_CKPT", "DEMO_CKPT_ALT")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("demo:\n  family: depth\n  colour: red\n", "unknown field(s) colour"),
        (
            "demo:\n  family: depth\n  assets:\n    - {id: a, kind: dir, role: r, env: [], local_path: p, size: 1}\n",
            "size",
        ),
        ("demo:\n  template: nope\n", "unknown template 'nope'"),
        ("demo:\n  template: in_tree_source\n  path: p\n  upstream: u\n  branch: main\n", "branch"),
    ],
)
def test_invalid_declarations_fail_with_their_location(registry_dir: Path, text: str, message: str) -> None:
    _write(registry_dir / "capabilities" / "bad.yaml", text)

    with pytest.raises(CapabilityRegistryError, match="bad.yaml: demo") as excinfo:
        capabilities._load_capabilities(registry_dir / "capabilities")
    assert message in str(excinfo.value)


@pytest.mark.unit
def test_duplicate_ids_across_files_are_rejected(registry_dir: Path) -> None:
    _write(registry_dir / "capabilities" / "a.yaml", "demo:\n  family: depth\n")
    _write(registry_dir / "capabilities" / "b.yaml", "demo:\n  family: slam\n")

    with pytest.raises(CapabilityRegistryError, match="duplicate capability id"):
        capabilities._load_capabilities(registry_dir / "capabilities")


@pytest.mark.unit
def test_stacks_expand_nested_members_in_order_and_reject_unknown_ids(registry_dir: Path) -> None:
    known = {name: BASE_MODEL_CAPABILITIES[name] for name in ("raft", "sea_raft", "droid_slam")}
    path = _write(
        registry_dir / "stacks.yaml",
        """
motion: {family: m, capabilities: [raft, sea_raft], note: n}
all: {family: m, capabilities: [droid_slam, motion, raft], note: n}
""",
    )
    stacks = capabilities._load_stacks(path, known)
    assert stacks["all"].capability_ids == ("droid_slam", "raft", "sea_raft")

    _write(path, "broken: {family: m, capabilities: [missing], note: n}\n")
    with pytest.raises(CapabilityRegistryError, match="unknown capability or stack 'missing'"):
        capabilities._load_stacks(path, known)

    _write(path, "a: {family: m, capabilities: [b], note: n}\nb: {family: m, capabilities: [a], note: n}\n")
    with pytest.raises(CapabilityRegistryError, match="cycle"):
        capabilities._load_stacks(path, known)


@pytest.mark.unit
def test_benchmark_assets_must_name_registered_capabilities(registry_dir: Path) -> None:
    path = _write(registry_dir / "benchmark_assets.yaml", "demo-bench: [raft, missing]\n")

    with pytest.raises(CapabilityRegistryError, match="demo-bench"):
        capabilities._load_benchmark_assets(path, BASE_MODEL_CAPABILITIES)


@pytest.mark.unit
def test_templates_expand_to_dataset_and_in_tree_assets() -> None:
    dataset = BASE_MODEL_CAPABILITIES["aigcbench_dataset_assets"]
    assert dataset.family == "benchmark_data_asset"
    assert dataset.assets[0].hf_repo_type == "dataset"
    assert dataset.assets[0].local_path.startswith("${WORLDFOUNDRY_HFD_DATASET_ROOT}/")

    source = BASE_MODEL_CAPABILITIES["calvin_source_assets"].assets[0]
    assert source.id == "calvin_source_assets_in_tree"
    assert source.role == "benchmark_in_tree_source"
    assert source.download_command() is None


@pytest.mark.unit
def test_external_official_runtime_template_requires_a_manually_staged_checkout() -> None:
    capability = BASE_MODEL_CAPABILITIES["behavior1k_official_runtime_assets"]
    checkout, dataset = capability.assets

    assert checkout.env == ("WORLDFOUNDRY_BEHAVIOR1K_ROOT",)
    assert checkout.download_command() is None
    assert "does not clone it" in checkout.manual_action
    assert dataset.local_path == "${WORLDFOUNDRY_DATA_DIR}/datasets/behavior1k"


@pytest.mark.unit
def test_in_tree_source_honours_the_runner_root_override(tmp_path: Path) -> None:
    asset = BASE_MODEL_CAPABILITIES["mind_source_assets"].assets[0]
    override = tmp_path / "mind"
    override.mkdir()
    (override / "process.py").write_text("", encoding="utf-8")

    assert asset.env == ("WORLDFOUNDRY_MIND_ROOT",)
    assert asset.check({"WORLDFOUNDRY_MIND_ROOT": str(override)})["matched_path"] == str(override)


@pytest.mark.unit
def test_dotted_import_with_missing_parent_package_is_unavailable() -> None:
    assert capabilities._import_available("wf_missing_parent_package.submodule") is False
