from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from worldfoundry.synthesis.visual_generation.runtime_manifest import (
    resolve_runtime_manifest,
    runtime_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_ROOT = REPO_ROOT / "worldfoundry/synthesis/visual_generation"
MODULE_ROOT = "worldfoundry.synthesis.visual_generation"
MODELS = (
    ("adaworld", "adaworld"),
    ("diamond", "diamond"),
    ("dino-wm", "dino_wm"),
    ("egowm", "egowm"),
    ("hma", "hma"),
    ("leworldmodel", "le_wm"),
    ("mineworld", "mineworld"),
    ("mira", "mira_wm"),
    ("nwm", "nwm"),
    ("open-dreamer", "open_dreamer"),
    ("oasis-500m", "open_oasis"),
    ("shotstream", "shotstream"),
    ("solarwm", "solarwm"),
    ("starwm", "starwm"),
    ("vid2world", "vid2world"),
    ("wilddet3d", "wilddet3d"),
)


@pytest.mark.parametrize(("model_id", "package"), MODELS)
def test_model_manifest_resolves_the_direct_package(model_id, package, tmp_path, monkeypatch):
    spec = runtime_spec(model_id)
    assert spec.runtime_module == f"{MODULE_ROOT}.{package}.worldfoundry_runtime"
    module = importlib.import_module(spec.runtime_module)
    model_root = VISUAL_ROOT / package
    assert Path(module.__file__).resolve().parent == model_root
    assert Path(module.RUNTIME_DIR).resolve() == model_root

    expected_root = model_root
    if spec.runtime_root_func:
        expected_root = tmp_path / "official-source" / package
        expected_root.mkdir(parents=True)
        monkeypatch.setattr(module, spec.runtime_root_func, lambda: expected_root)

    root, entrypoint, _ = resolve_runtime_manifest(spec)
    assert root == expected_root
    if spec.entrypoint_attr:
        assert entrypoint == Path(getattr(module, spec.entrypoint_attr)).resolve()
        assert entrypoint.is_file()
        assert entrypoint.is_relative_to(model_root)
    else:
        assert spec.entrypoint_relative
        assert entrypoint == expected_root / spec.entrypoint_relative


def test_model_discovery_remains_lightweight_in_a_fresh_process():
    modules = [f"{MODULE_ROOT}.{package}.worldfoundry_runtime" for _, package in MODELS]
    script = """
import importlib, json, sys
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
from worldfoundry.runtime.inference_catalog import list_model_inference_specs
list_model_inference_specs()
assert 'torch' not in sys.modules
assert 'diffusers' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(modules)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _config_targets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"target", "_target_"} and isinstance(item, str):
                yield item
            yield from _config_targets(item)
    elif isinstance(value, list):
        for item in value:
            yield from _config_targets(item)


def test_model_config_targets_resolve_to_packaged_modules():
    prefixes = tuple(f"{MODULE_ROOT}.{package}." for _, package in MODELS)
    config_root = REPO_ROOT / "worldfoundry/data/models/runtime/configs"
    checked = set()
    for _, package in MODELS:
        for path in (config_root / package).rglob("*.yaml"):
            targets = _config_targets(yaml.safe_load(path.read_text(encoding="utf-8")))
            for target in targets:
                if not target.startswith(prefixes):
                    continue
                module, _, name = target.rpartition(".")
                source = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
                if not source.is_file():
                    source = REPO_ROOT.joinpath(*module.split("."), "__init__.py")
                assert source.is_file(), f"{path}: {target}"
                tree = ast.parse(source.read_text(encoding="utf-8"))
                definitions = {
                    node.name for node in tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                }
                definitions.update(
                    alias.asname or alias.name
                    for node in tree.body if isinstance(node, ast.ImportFrom)
                    for alias in node.names
                )
                assert name in definitions, f"{path}: {target}"
                checked.add(target)
    assert len(checked) >= 15


def test_solarwm_keeps_both_registered_runtime_routes():
    from worldfoundry.synthesis.visual_generation.solarwm import SolarWMRuntime, worldfoundry_runtime

    assert Path(sys.modules[SolarWMRuntime.__module__].__file__).parent == VISUAL_ROOT / "solarwm"
    assert worldfoundry_runtime.OFFICIAL_ENTRYPOINT == VISUAL_ROOT / "solarwm/infer.py"


def test_removed_model_container_has_no_compatibility_package():
    assert not (VISUAL_ROOT / "world_model").exists()
    assert importlib.util.find_spec(f"{MODULE_ROOT}.world_model") is None
