"""Inference ownership follows the selected lazy backend and retains uncertain dispatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "lazy_dispatch_impact", _ROOT / "tests/manual/geometry_regression_impact.py"
)
impact = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(impact)


@pytest.fixture
def project(tmp_path):
    files = {
        "worldfoundry/backends/a.py": "",
        "worldfoundry/backends/b.py": "",
        "worldfoundry/registry.py": '''
from dataclasses import dataclass
import importlib
@dataclass
class WorldModelRuntimeSpec:
    runtime_module: str
WORLD_MODEL_RUNTIME_SPECS = {
    "a": WorldModelRuntimeSpec(runtime_module="worldfoundry.backends.a"),
    "b": WorldModelRuntimeSpec(runtime_module="worldfoundry.backends.b"),
}
def runtime_spec(model_id):
    return WORLD_MODEL_RUNTIME_SPECS[model_id]
class WorldModelRuntimeSynthesis:
    def generate(self, model_id):
        return importlib.import_module(runtime_spec(model_id).runtime_module)
''',
        "worldfoundry/pipeline.py": '''
from .registry import WorldModelRuntimeSynthesis
class A:
    MODEL_ID = "a"
class B:
    MODEL_ID = "b"
class Dynamic:
    MODEL_ID = resolve_identity()
''',
        "worldfoundry/metadata.py": '''
from .registry import WORLD_MODEL_RUNTIME_SPECS as records
class Catalog:
    def keys(self):
        return list(records)
''',
        "worldfoundry/loader.py": '''
import importlib
from .registry import WORLD_MODEL_RUNTIME_SPECS
class Loader:
    def generate(self, model_id):
        return importlib.import_module(WORLD_MODEL_RUNTIME_SPECS[model_id].runtime_module)
''',
        "worldfoundry/components.py": '''
def _component_pipeline_class(name, **targets):
    return type(name, (), targets)
A = _component_pipeline_class("A", operator_target="worldfoundry.backends.a:Operator")
B = _component_pipeline_class("B", operator_target="worldfoundry.backends.b:Operator")
''',
        "worldfoundry/component_user.py": "from .components import A\nclass User: pass\n",
    }
    for name, content in files.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    return tmp_path


@pytest.mark.parametrize("name", ["A", "B"])
def test_concrete_runtime_identity_owns_only_its_adapter(project, name):
    files = impact.ImportGraph(project).closure("worldfoundry.pipeline:" + name)
    assert "worldfoundry/backends/" + name.lower() + ".py" in files
    other = "b" if name == "A" else "a"
    assert "worldfoundry/backends/" + other + ".py" not in files
    assert "worldfoundry/registry.py" in files


def test_metadata_reader_does_not_execute_registered_backends(project):
    files = impact.ImportGraph(project).closure("worldfoundry.metadata:Catalog")
    assert "worldfoundry/registry.py" in files
    assert not {"worldfoundry/backends/a.py", "worldfoundry/backends/b.py"} & files


@pytest.mark.parametrize("target", ["worldfoundry.pipeline:Dynamic", "worldfoundry.loader:Loader"])
def test_unknown_runtime_dispatch_retains_every_adapter(project, target):
    files = impact.ImportGraph(project).closure(target)
    assert {"worldfoundry/backends/a.py", "worldfoundry/backends/b.py"} <= files


@pytest.mark.parametrize("target", ["worldfoundry.components:A", "worldfoundry.component_user:User"])
def test_component_factory_tracks_selected_export_through_named_import(project, target):
    files = impact.ImportGraph(project).closure(target)
    assert "worldfoundry/backends/a.py" in files
    assert "worldfoundry/backends/b.py" not in files


def test_unscoped_component_import_retains_every_factory_target(project):
    files = impact.ImportGraph(project).closure("worldfoundry.components")
    assert {"worldfoundry/backends/a.py", "worldfoundry/backends/b.py"} <= files


def test_eager_factory_keeps_every_executed_target(project):
    file = project / "worldfoundry/components.py"
    file.write_text(file.read_text().replace(
        "return type(name, (), targets)",
        "load_targets(targets)\n    return type(name, (), targets)",
    ))
    files = impact.ImportGraph(project).closure("worldfoundry.components:A")
    assert {"worldfoundry/backends/a.py", "worldfoundry/backends/b.py"} <= files


def test_record_constructor_with_side_effects_is_not_metadata_only(project):
    file = project / "worldfoundry/registry.py"
    file.write_text(file.read_text().replace(
        "    runtime_module: str",
        "    runtime_module: str\n    def __post_init__(self):\n        importlib.import_module(self.runtime_module)",
    ))
    files = impact.ImportGraph(project).closure("worldfoundry.metadata:Catalog")
    assert {"worldfoundry/backends/a.py", "worldfoundry/backends/b.py"} <= files


@pytest.mark.parametrize("target,adapter", [
    ("worldfoundry.pipelines.world_model.pipeline_runtime_manifest:DIAMONDPipeline", "diamond"),
    ("worldfoundry.pipelines.world_model.pipeline_runtime_manifest:DinoWMPipeline", "dino_wm"),
])
def test_catalogued_runtime_does_not_own_another_models_camera_attention(target, adapter):
    files = impact.ImportGraph(_ROOT).closure(target)
    assert f"worldfoundry/synthesis/visual_generation/{adapter}/worldfoundry_runtime.py" in files
    assert "worldfoundry/base_models/diffusion_model/models/networks/wan/variants/camera_attention.py" not in files
