import importlib
import importlib.util
import ast
from pathlib import Path

import pytest

import worldfoundry.operators as operators
from worldfoundry.operators.base_operator import BaseOperator


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_FRAMEWORK_GLOBS = (
    "worldfoundry/operators/*.py",
    "worldfoundry/pipelines/**/*.py",
    "worldfoundry/core/memory/**/*.py",
    "worldfoundry/synthesis/action_generation/memory.py",
    "worldfoundry/synthesis/visual_generation/memory/**/*.py",
    "worldfoundry/evaluation/**/*.py",
    "worldfoundry/studio/**/*.py",
)


def test_base_operator_saves_operation_types_and_uses_fresh_default():
    operation_types = ["visual_instruction"]
    operator = BaseOperator(operation_types=operation_types)

    assert operator.operation_types == operation_types
    assert not hasattr(operator, "op" + "ration_types")

    first = BaseOperator()
    second = BaseOperator()
    first.operation_types.append("textual_instruction")

    assert first.operation_types == ["textual_instruction"]
    assert second.operation_types == []


def test_hunyuan_world_voyager_has_no_historical_alias_module():
    old_module = "worldfoundry.operators.hunyuan_world_" + "vo" + "ager_operator"
    old_path = REPO_ROOT / "worldfoundry/operators" / ("hunyuan_world_" + "vo" + "ager_operator.py")

    assert importlib.util.find_spec("worldfoundry.operators.hunyuan_world_voyager_operator") is not None
    assert not old_path.exists()
    assert importlib.util.find_spec(old_module) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_module)


def test_visual_memories_use_canonical_stream_module():
    retired_modules = [
        "worldfoundry.memories.visual_synthesis.stream_memory",
        "worldfoundry.memories.visual_synthesis.wan.wan_2p2_memory",
        "worldfoundry.memories.visual_synthesis.worldcam.worldcam_memory",
        "worldfoundry.memories.visual_synthesis.matrix_game.matrix_game_2_memory",
        "worldfoundry.memories.visual_synthesis.lyra.lyra1_memory",
        "worldfoundry.memories.visual_synthesis.yume.yume_memory",
    ]

    assert importlib.util.find_spec("worldfoundry.synthesis.visual_generation.memory.stream") is not None
    assert importlib.util.find_spec("worldfoundry.core.memory") is not None
    for module_name in retired_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_operator_public_surface_exports_every_operator_class():
    operator_classes = set()

    for path in sorted((REPO_ROOT / "worldfoundry/operators").glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Operator") and not node.name.startswith("_"):
                operator_classes.add(node.name)

    assert set(operators.__all__) == operator_classes


def test_concrete_operator_classes_inherit_from_unified_base():
    operator_bases = {}
    violations = []

    for path in sorted((REPO_ROOT / "worldfoundry/operators").glob("*.py")):
        if path.name in {"__init__.py", "base_operator.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Operator"):
                continue
            operator_bases[node.name] = {
                base.id if isinstance(base, ast.Name) else base.attr
                for base in node.bases
                if isinstance(base, (ast.Name, ast.Attribute))
            }

    def inherits_unified_base(class_name: str, seen: set[str] | None = None) -> bool:
        """Return whether an operator class reaches the shared base class."""
        if class_name in {"BaseOperator", "EmbodiedActionOperator"}:
            return True
        seen = set() if seen is None else seen
        if class_name in seen:
            return False
        seen.add(class_name)
        return any(inherits_unified_base(base, seen) for base in operator_bases.get(class_name, set()))

    for path in sorted((REPO_ROOT / "worldfoundry/operators").glob("*.py")):
        if path.name in {"__init__.py", "base_operator.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Operator") and not inherits_unified_base(node.name):
                relative = path.relative_to(REPO_ROOT)
                violations.append(f"{relative}:{node.lineno}: {node.name}")

    assert violations == []


def test_public_framework_layers_do_not_use_mutable_defaults():
    violations = []

    for pattern in PUBLIC_FRAMEWORK_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                default_args = node.args.args[-len(node.args.defaults):] if node.args.defaults else []
                for arg, default in zip(default_args, node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        relative = path.relative_to(REPO_ROOT)
                        violations.append(f"{relative}:{node.lineno}: {node.name}({arg.arg}=mutable)")

    assert violations == []


def test_component_runtime_pipelines_use_pipelineabc_without_profilebacked_inheritance():
    component_pipelines = []
    duplicate_loaders = []
    profilebacked_references = []

    for path in sorted((REPO_ROOT / "worldfoundry/pipelines").glob("**/pipeline_*.py")):
        text = path.read_text(encoding="utf-8")
        if "load_runtime_profile" in text:
            duplicate_loaders.append(str(path.relative_to(REPO_ROOT)))
        if "ProfileBackedPipeline" in text:
            profilebacked_references.append(str(path.relative_to(REPO_ROOT)))

        tree = ast.parse(text, filename=str(path))
        for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
            base_names = {
                base.id if isinstance(base, ast.Name) else base.attr
                for base in class_node.bases
                if isinstance(base, (ast.Name, ast.Attribute))
            }
            if "PipelineABC" not in base_names:
                continue
            class_attrs = {node.targets[0].id for node in class_node.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
            if {"OPERATOR_CLS", "MEMORY_CLS", "SYNTHESIS_CLS"} <= class_attrs:
                component_pipelines.append(f"{path.relative_to(REPO_ROOT)}:{class_node.name}")

    from worldfoundry.pipelines import component_pipelines as component_pipeline_module
    from worldfoundry.pipelines.pipeline_utils import PipelineABC

    for name in component_pipeline_module.__all__:
        pipeline_cls = getattr(component_pipeline_module, name)
        if not isinstance(pipeline_cls, type):
            continue
        if pipeline_cls is component_pipeline_module.ComponentPipeline:
            continue
        if issubclass(pipeline_cls, PipelineABC) and pipeline_cls._uses_component_contract():
            component_pipelines.append(f"worldfoundry/pipelines/component_pipelines.py:{name}")

    assert duplicate_loaders == []
    assert profilebacked_references == []
    assert len(component_pipelines) >= 25


def test_pipelineabc_component_contract_runs_common_runtime_flow():
    from worldfoundry.pipelines.pipeline_utils import PipelineABC

    class FakeOperator:
        def __init__(self, input_schema=None):
            self.input_schema = input_schema or {}
            self.interactions = None
            self.deleted = False

        def get_interaction(self, interactions):
            self.interactions = interactions

        def process_interaction(self):
            return {"actions": self.interactions}

        def delete_last_interaction(self):
            self.deleted = True

        def process_prompt(self, prompt, **kwargs):
            return {"prompt": prompt, "extra_inputs": {"seed": kwargs.get("seed")}}

        def process_perception(self, images=None, video=None, ref_image_path=None, **kwargs):
            return {
                "images": images,
                "video": video,
                "ref_image_path": ref_image_path,
                "quality": kwargs.get("quality"),
            }

    class FakeSynthesis:
        def __init__(self):
            self.calls = []

        def predict(self, prompt, images, video, interactions, output_path=None, fps=None, **kwargs):
            call = {
                "prompt": prompt,
                "images": images,
                "video": video,
                "interactions": interactions,
                "output_path": output_path,
                "fps": fps,
                "kwargs": kwargs,
            }
            self.calls.append(call)
            return {"artifact_path": output_path, "call": call}

    class FakeMemory:
        def __init__(self):
            self.records = []
            self.previous = None

        def record(self, result, metadata=None):
            self.records.append((result, metadata))

        def select(self):
            return self.previous

    class FakePipeline(PipelineABC):
        MODEL_ID = "local-pipeline"
        OPERATOR_CLS = FakeOperator
        MEMORY_CLS = FakeMemory
        SYNTHESIS_CLS = FakeSynthesis

    operator = FakeOperator()
    synthesis = FakeSynthesis()
    memory = FakeMemory()
    pipeline = FakePipeline(
        model_id="local-pipeline-model",
        operator=operator,
        synthesis_model=synthesis,
        memory_module=memory,
        device="cpu",
    )

    result = pipeline(
        prompt="turn left",
        images=["frame-0"],
        interactions=["left"],
        output_path="out.mp4",
        fps=12,
        return_dict=True,
        ref_image_path="ref.png",
        operator_kwargs={"seed": 7, "quality": "high"},
        guidance_scale=1.5,
    )

    assert result["artifact_path"] == "out.mp4"
    assert result["call"] == {
        "prompt": "turn left",
        "images": ["frame-0"],
        "video": None,
        "interactions": ["left"],
        "output_path": "out.mp4",
        "fps": 12,
        "kwargs": {"seed": 7, "quality": "high", "guidance_scale": 1.5},
    }
    assert operator.deleted is True
    assert memory.records == [(result, {"type": "runtime_result", "model_id": "local-pipeline-model"})]

    memory.previous = {"artifact_path": "previous.mp4"}
    streamed = pipeline.stream(prompt="continue", return_dict=True)

    assert streamed["call"]["images"] == "previous.mp4"
