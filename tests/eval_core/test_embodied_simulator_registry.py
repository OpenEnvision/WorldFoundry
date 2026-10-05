from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from worldfoundry.evaluation.tasks.catalog.zoo_registry import (
    clear_benchmark_zoo_registry_cache,
    load_benchmark_zoo_registry,
)
from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.embodied import EmbodiedClosedLoopRunner
from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
from worldfoundry.evaluation.tasks.embodied.docker_runner import build_docker_run_command
from worldfoundry.evaluation.tasks.embodied.simulators import (
    SIMULATOR_ENTRIES,
    get_simulator_entry,
    list_simulator_ids,
    resolve_simulator_class,
)
from worldfoundry.evaluation.tasks.embodied.simulators.base import BaseSimulator
from worldfoundry.evaluation.tasks.embodied.simulators.specs import (
    GRIPPER_CLOSE_NEG,
    GRIPPER_CLOSE_POS,
    POSITION_DELTA,
    ROTATION_AA,
    ROTATION_EULER,
    DimSpec,
    check_specs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_EMBODIED_BENCHMARK_IDS = (
    "behavior1k",
    "kinetix",
    "libero-mem",
    "libero-plus",
    "libero-pro",
    "maniskill2",
    "mikasa",
    "molmospaces",
    "robocerebra",
    "robomme",
    "vlabench",
)

HARNESS_DOCKER_IMAGES = {
    "behavior1k": "ghcr.io/allenai/vla-evaluation-harness/behavior1k:latest",
    "calvin": "ghcr.io/allenai/vla-evaluation-harness/calvin:latest",
    "kinetix": "ghcr.io/allenai/vla-evaluation-harness/kinetix:latest",
    "libero": "ghcr.io/allenai/vla-evaluation-harness/libero:latest",
    "libero-mem": "ghcr.io/allenai/vla-evaluation-harness/libero-mem:latest",
    "libero-plus": "ghcr.io/allenai/vla-evaluation-harness/libero-plus:latest",
    "libero-pro": "ghcr.io/allenai/vla-evaluation-harness/libero-pro:latest",
    "maniskill2": "ghcr.io/allenai/vla-evaluation-harness/maniskill2:latest",
    "mikasa": "ghcr.io/allenai/vla-evaluation-harness/mikasa-robo:latest",
    "molmospaces": "ghcr.io/allenai/vla-evaluation-harness/molmospaces:latest",
    "rlbench": "ghcr.io/allenai/vla-evaluation-harness/rlbench:latest",
    "robocasa": "ghcr.io/allenai/vla-evaluation-harness/robocasa:latest",
    "robocerebra": "ghcr.io/allenai/vla-evaluation-harness/robocerebra:latest",
    "robomme": "ghcr.io/allenai/vla-evaluation-harness/robomme:latest",
    "robotwin": "ghcr.io/allenai/vla-evaluation-harness/robotwin:latest",
    "simpler-env": "ghcr.io/allenai/vla-evaluation-harness/simpler:latest",
    "vlabench": "ghcr.io/allenai/vla-evaluation-harness/vlabench:latest",
}


def test_native_simulator_registry_resolves_every_declared_class() -> None:
    assert len(SIMULATOR_ENTRIES) >= 17
    assert "libero" in list_simulator_ids()
    assert "vlabench" in list_simulator_ids()
    for entry in SIMULATOR_ENTRIES:
        cls = resolve_simulator_class(entry.benchmark_id)
        assert issubclass(cls, BaseSimulator), entry.import_path


def test_native_simulator_registry_official_config_dirs_exist() -> None:
    for entry in SIMULATOR_ENTRIES:
        assert entry.official_config_dir is not None
        config_dir = REPO_ROOT / entry.official_config_dir
        assert config_dir.is_dir(), (entry.benchmark_id, config_dir)
        assert tuple(config_dir.glob("*.yaml")), (entry.benchmark_id, config_dir)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("libero_10", "libero"),
        ("maniskill", "maniskill2"),
        ("simpler", "simpler-env"),
        ("robotwin-v2", "robotwin"),
        ("vla-bench", "vlabench"),
    ],
)
def test_native_simulator_registry_resolves_official_aliases(alias: str, expected: str) -> None:
    entry = get_simulator_entry(alias)
    assert entry is not None
    assert entry.benchmark_id == expected


def test_normalizer_only_benchmarks_are_not_closed_loop_simulators() -> None:
    for benchmark_id in ("libero-para", "bridgedata-v2", "metaworld"):
        assert get_simulator_entry(benchmark_id) is None


def test_harness_embodied_benchmarks_have_contracts_and_catalog_entries() -> None:
    clear_benchmark_zoo_registry_cache()
    registry = load_benchmark_zoo_registry(REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog")
    for benchmark_id in HARNESS_EMBODIED_BENCHMARK_IDS:
        contract = get_external_benchmark_contract(benchmark_id)
        assert contract.benchmark_id == benchmark_id
        entry = registry.get(benchmark_id)
        assert entry.runner_target is not None
        assert entry.verification_status == "normalizer_only"
        assert entry.official_benchmark_verified is False
        assert entry.leaderboard_valid is False


def test_harness_docker_images_match_official_runtime_profiles() -> None:
    profile_root = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "runtime_profiles" / "official"
    digest_map = json.loads((profile_root / "docker_image_digests.json").read_text(encoding="utf-8"))["images"]
    for profile_id, expected_image in HARNESS_DOCKER_IMAGES.items():
        payload = yaml.safe_load((profile_root / f"{profile_id}.yaml").read_text(encoding="utf-8"))
        docker = payload["docker"]
        resolved = digest_map.get(expected_image)
        if resolved is None:
            assert docker["image"] == expected_image
            assert docker["source_image"] == expected_image
            continue
        expected_source = f"{expected_image.rsplit(':', 1)[0]}@{resolved['digest']}"
        assert docker["source_image"] == expected_source
        assert docker["platform"] == resolved["platform"] == "linux/amd64"
        if profile_id == "libero":
            assert docker["image"] == "ghcr.io/openenvision/worldfoundry-embodied-libero:latest"
            assert "digest" not in docker
        else:
            assert docker["image"] == expected_source
            assert docker["digest"] == resolved["digest"]


def test_runtime_profiles_canonicalize_to_their_own_benchmark_ids() -> None:
    profile_root = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "runtime_profiles" / "official"
    for profile_id in HARNESS_DOCKER_IMAGES:
        config = load_canonical_embodied_config(profile_root / f"{profile_id}.yaml")
        assert config["benchmarks"][0]["benchmark_id"] == profile_id


def test_docker_runner_forwards_configured_runtime(tmp_path: Path) -> None:
    docker_config_path = tmp_path / "eval_config.yaml"
    docker_config_path.write_text("id: test\n", encoding="utf-8")

    cmd = build_docker_run_command(
        {"docker": {"image": "example/bench:latest", "runtime": "nvidia"}},
        docker_config_path=docker_config_path,
        output_dir=tmp_path / "out",
    )

    assert "--runtime" in cmd
    assert cmd[cmd.index("--runtime") + 1] == "nvidia"


def test_embodied_asset_scaffold_covers_harness_benchmarks(tmp_path: Path) -> None:
    script_path = REPO_ROOT / "scripts" / "setup" / "prepare_embodied_official_assets.py"
    spec = importlib.util.spec_from_file_location("prepare_embodied_official_assets", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.prepare(
        Namespace(
            source_manifest=module.DEFAULT_SOURCE_MANIFEST,
            output_root=tmp_path,
            benchmark_id=None,
            create_dirs=False,
        )
    )
    payload = yaml.safe_load(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    ids = {item["id"] for item in payload["benchmarks"]}
    assert set(HARNESS_EMBODIED_BENCHMARK_IDS).issubset(ids)
    env_text = Path(report["env_path"]).read_text(encoding="utf-8")
    assert "WORLDFOUNDRY_BEHAVIOR1K_RESULTS_PATH" in env_text
    assert "WORLDFOUNDRY_ROBOMME_RESULTS_PATH" in env_text
    assert "WORLDFOUNDRY_VLABENCH_RESULTS_PATH" in env_text


def test_closed_loop_runner_uses_native_registry_class_names() -> None:
    runner = EmbodiedClosedLoopRunner("zero-policy", "libero")
    assert runner._benchmark_class is resolve_simulator_class("libero")


def test_dim_spec_validates_values_and_round_trips() -> None:
    spec = DimSpec("position", 3, "delta_xyz", (-1, 1))
    assert spec.validate([0.0, 0.5, -0.5]) == []
    assert "expected at least 3D" in spec.validate([0.0])[0]
    assert "outside" in spec.validate([0.0, 2.0, 0.0])[0]
    assert DimSpec.from_dict(spec.to_dict()) == spec


def test_check_specs_reports_real_convention_mismatch() -> None:
    warnings = check_specs(
        producer_action={"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS},
        consumer_action={"position": POSITION_DELTA, "rotation": ROTATION_EULER, "gripper": GRIPPER_CLOSE_NEG},
        producer_observation={},
        consumer_observation={},
    )
    assert any("axis_angle vs euler_xyz" in warning for warning in warnings)
    assert any("binary_close_positive vs binary_close_negative" in warning for warning in warnings)
