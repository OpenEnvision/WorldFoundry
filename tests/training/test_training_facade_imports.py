from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _modules_after_fresh_import(module_name: str) -> set[str]:
    root = Path(__file__).resolve().parents[2]
    probe = f"""
import importlib
import json
import sys

importlib.import_module({module_name!r})
print(json.dumps(sorted(sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(result.stdout))


def _assert_public_exports_have_leaf_identity(facade: ModuleType) -> None:
    exports = facade._EXPORTS  # type: ignore[attr-defined]
    assert facade.__all__ == list(exports)  # type: ignore[attr-defined]
    for name, module_name in exports.items():
        leaf = importlib.import_module(module_name, facade.__name__)
        value = getattr(facade, name)
        assert value is getattr(leaf, name), name
        assert facade.__dict__[name] is value


def test_data_facade_import_does_not_load_implementation_modules() -> None:
    loaded = _modules_after_fresh_import("worldfoundry.training.data")

    assert "worldfoundry.training.data" in loaded
    assert not {
        name
        for name in loaded
        if name.startswith("worldfoundry.training.data.")
        or name == "worldfoundry.training.models"
        or name.startswith("worldfoundry.training.models.")
        or name == "torch"
        or name.startswith("torch.")
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "worldfoundry.training.data.ltx",
        "worldfoundry.training.data.cosmos",
    ],
)
def test_video_family_data_facades_do_not_load_implementations_or_torch(
    module_name: str,
) -> None:
    loaded = _modules_after_fresh_import(module_name)

    assert module_name in loaded
    assert not {
        name for name in loaded if name.startswith(f"{module_name}.") or name == "torch" or name.startswith("torch.")
    }


def test_models_facade_import_does_not_load_model_families_or_torch() -> None:
    loaded = _modules_after_fresh_import("worldfoundry.training.models")

    assert "worldfoundry.training.models" in loaded
    assert not {
        name
        for name in loaded
        if name.startswith("worldfoundry.training.models.") or name == "torch" or name.startswith("torch.")
    }


def test_checkpoint_facade_import_does_not_load_checkpoint_runtime_or_torch() -> None:
    loaded = _modules_after_fresh_import("worldfoundry.training.checkpoint")

    assert "worldfoundry.training.checkpoint" in loaded
    assert not {
        name
        for name in loaded
        if name.startswith("worldfoundry.training.checkpoint.") or name == "torch" or name.startswith("torch.")
    }


def test_flow_policy_facade_does_not_eager_load_algorithm_registry_or_torch() -> None:
    module_name = "worldfoundry.training.post_training.rl.algorithms.flow_policy"
    loaded = _modules_after_fresh_import(module_name)

    assert module_name in loaded
    assert not {
        name for name in loaded if name.startswith(f"{module_name}.") or name == "torch" or name.startswith("torch.")
    }


@pytest.mark.parametrize(
    "module_name",
    ["worldfoundry.training.distributed", "worldfoundry.training.tuning"],
)
def test_training_subsystem_facades_do_not_load_implementation_or_torch(
    module_name: str,
) -> None:
    loaded = _modules_after_fresh_import(module_name)

    assert module_name in loaded
    assert not {
        name for name in loaded if name.startswith(f"{module_name}.") or name == "torch" or name.startswith("torch.")
    }


def test_data_facade_preserves_every_public_object_identity() -> None:
    facade = importlib.import_module("worldfoundry.training.data")
    _assert_public_exports_have_leaf_identity(facade)


@pytest.mark.parametrize(
    "module_name",
    [
        "worldfoundry.training.data.ltx",
        "worldfoundry.training.data.cosmos",
    ],
)
def test_video_family_data_facades_preserve_public_object_identity(
    module_name: str,
) -> None:
    facade = importlib.import_module(module_name)
    _assert_public_exports_have_leaf_identity(facade)


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("VideoCachePreparationResult", "worldfoundry.training.data.video_precompute"),
        ("project_causal_video_mask_to_latent", "worldfoundry.training.data.video_masks"),
        ("LTXTextFeatureEncoder", "worldfoundry.training.data.ltx.encoding"),
        ("LTXVideoFeatureEncoder", "worldfoundry.training.data.ltx.encoding"),
        ("build_ltx_video_decoding_dataset", "worldfoundry.training.data.ltx.training_cache"),
        ("materialize_ltx_training_cache", "worldfoundry.training.data.ltx.training_cache"),
        ("prepare_ltx_training_cache_from_audits", "worldfoundry.training.data.ltx.training_cache"),
        ("CosmosTextFeatureEncoder", "worldfoundry.training.data.cosmos.encoding"),
        ("CosmosVideoFeatureEncoder", "worldfoundry.training.data.cosmos.encoding"),
        ("build_cosmos_video_decoding_dataset", "worldfoundry.training.data.cosmos.training_cache"),
        ("materialize_cosmos_training_cache", "worldfoundry.training.data.cosmos.training_cache"),
        ("prepare_cosmos_training_cache_from_audits", "worldfoundry.training.data.cosmos.training_cache"),
    ],
)
def test_video_family_cache_api_is_available_from_data_facade(
    name: str,
    module_name: str,
) -> None:
    facade = importlib.import_module("worldfoundry.training.data")
    leaf = importlib.import_module(module_name)

    assert name in facade.__all__  # type: ignore[attr-defined]
    assert getattr(facade, name) is getattr(leaf, name)


def test_models_facade_preserves_every_public_object_identity() -> None:
    facade = importlib.import_module("worldfoundry.training.models")
    _assert_public_exports_have_leaf_identity(facade)


def test_checkpoint_facade_preserves_every_public_object_identity() -> None:
    facade = importlib.import_module("worldfoundry.training.checkpoint")
    _assert_public_exports_have_leaf_identity(facade)


def test_flow_policy_facade_preserves_public_object_identity() -> None:
    facade = importlib.import_module("worldfoundry.training.post_training.rl.algorithms.flow_policy")
    _assert_public_exports_have_leaf_identity(facade)


@pytest.mark.parametrize(
    "name",
    [
        "NativeAgenticTrainingRun",
        "NativeAgenticTrainingSession",
        "materialize_agentic_training_run",
        "setup_ray_agentic_rollout",
    ],
)
def test_root_training_facade_exposes_agentic_entrypoints(name: str) -> None:
    facade = importlib.import_module("worldfoundry.training")
    post_training = importlib.import_module("worldfoundry.training.post_training")

    assert name in facade.__all__  # type: ignore[attr-defined]
    assert getattr(facade, name) is getattr(post_training, name)


@pytest.mark.parametrize(
    "module_name",
    ["worldfoundry.training.distributed", "worldfoundry.training.tuning"],
)
def test_training_subsystem_facades_preserve_public_object_identity(
    module_name: str,
) -> None:
    facade = importlib.import_module(module_name)
    _assert_public_exports_have_leaf_identity(facade)
