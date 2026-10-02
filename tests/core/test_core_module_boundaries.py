"""CPU-safe guards for lazy facades and the direction of core dependencies."""

from __future__ import annotations

import ast
import subprocess
import sys
import tokenize
from importlib.util import resolve_name
from pathlib import Path


def test_control_plane_imports_remain_lightweight() -> None:
    """Package boundaries must not load optional model stacks in a fresh process."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import worldfoundry.core
import worldfoundry.core.attention
import worldfoundry.core.configuration
import worldfoundry.core.io
import worldfoundry.core.nn
import worldfoundry.core.utils
from worldfoundry.core.execution import compile_cache, inference, process, realtime, runtime_cache
from worldfoundry.core.geometry import path
from worldfoundry.core.observability import logging_setup, nvtx, realtime_timing, time, torchprofile
from worldfoundry.core.media.processing import postprocess, rtx
from worldfoundry.core import PromptProcessor, configure_logging, VideoChunk, ModelInferenceSpec
assert not {'torch', 'numpy', 'hydra', 'diffusers', 'nvvfx'} & sys.modules.keys()
""",
        ],
        check=True,
        timeout=60,
    )


def test_lazy_targets_exist_without_optional_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    unresolved = []
    for path in (root / "worldfoundry/core").rglob("__init__.py"):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(name, ast.Name) and name.id in {"_EXPORT_MODULES", "_SUBMODULES"} for name in node.targets
            ):
                continue
            for target in ast.literal_eval(node.value).values():
                source = root.joinpath(*target.split("."))
                if not source.with_suffix(".py").is_file() and not (source / "__init__.py").is_file():
                    unresolved.append((str(path.relative_to(root)), target))
    assert not unresolved, f"lazy targets have no source: {unresolved}"


def test_core_imports_resolve_without_family_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    unresolved = []
    family_dependencies = []
    for path in (root / "worldfoundry/core").rglob("*.py"):
        package = ".".join(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                targets = [
                    resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
                ]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                if target.startswith(
                    (
                        "worldfoundry.base_models",
                        "worldfoundry.pipelines",
                        "worldfoundry.synthesis",
                        "worldfoundry.studio",
                        "worldfoundry.evaluation",
                        "worldfoundry.operators",
                        "worldfoundry.training",
                    )
                ):
                    family_dependencies.append((str(path.relative_to(root)), target))
                if not target.startswith("worldfoundry.core"):
                    continue
                source = root.joinpath(*target.split("."))
                if not source.with_suffix(".py").is_file() and not (source / "__init__.py").is_file():
                    unresolved.append((str(path.relative_to(root)), target))
    assert not unresolved, f"core imports have no source: {unresolved}"
    assert not family_dependencies, f"core depends on a model or application package: {family_dependencies}"


def test_model_consumers_reference_existing_core_modules() -> None:
    """Check model imports without initializing models, CUDA, or optional extensions."""
    root = Path(__file__).resolve().parents[2]
    unresolved = []
    for path in (root / "worldfoundry").rglob("*.py"):
        with tokenize.open(path) as stream:
            source = stream.read()
        if "worldfoundry.core" not in source:
            continue
        package = ".".join(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                targets = [
                    resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
                ]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                if not target.startswith("worldfoundry.core"):
                    continue
                module = root.joinpath(*target.split("."))
                if not module.with_suffix(".py").is_file() and not (module / "__init__.py").is_file():
                    unresolved.append((str(path.relative_to(root)), node.lineno, target))
    assert not unresolved, f"model consumers reference missing core modules: {unresolved}"
