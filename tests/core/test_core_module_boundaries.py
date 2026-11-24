"""CPU-safe guards for lazy facades and the direction of core dependencies."""

from __future__ import annotations

import ast
import subprocess
import sys
import tokenize
from graphlib import CycleError, TopologicalSorter
from importlib.util import resolve_name
from pathlib import Path

# Core units from the bottom layer up. A unit imports its own layer or lower ones,
# and units within one layer never import each other in a cycle.
_LAYERS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "action_normalization",
            "contracts",
            "env_registry",
            "execution.deadline",
            "json_contract",
            "model_loading.policy",
            "prompting",
            "registry",
            "utils",
            "world_model_manifest",
        }
    ),
    frozenset({"observability"}),
    frozenset({"io"}),
    frozenset({"configuration", "platform"}),
    frozenset({"distributed"}),
    frozenset({"geometry", "kernels", "media"}),
    frozenset({"attention"}),
    frozenset({"nn"}),
    frozenset({"model_loading.checkpoints"}),
    frozenset({"acceleration", "vram"}),
    frozenset({"model_loading"}),
    frozenset({"execution", "memory", "safety"}),
)


def _lazy_tables(init: Path) -> dict[str, str]:
    """Return the ``_EXPORT_MODULES`` and ``_SUBMODULES`` entries a package initializer declares."""
    entries: dict[str, str] = {}
    for node in ast.parse(init.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"_EXPORT_MODULES", "_SUBMODULES"} for target in node.targets
        ):
            entries.update(ast.literal_eval(node.value))
    return entries


def _binds_at_top_level(path: Path, name: str) -> bool:
    """Return whether ``path`` defines or assigns ``name`` at module level."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return True
    return False


def _core_unit(module: str) -> str | None:
    """Return the layer-table unit that owns ``module``; ``None`` is the ``worldfoundry.core`` facade."""
    relative = module.removeprefix("worldfoundry.core").lstrip(".")
    if not relative:
        return None
    matches = [unit for layer in _LAYERS for unit in layer if relative == unit or relative.startswith(f"{unit}.")]
    return max(matches, key=len) if matches else relative.split(".")[0]


def _core_import_edges(root: Path) -> list[tuple[str, str]]:
    """Return ``(importer, imported module)`` pairs inside core, including lazy and type-only imports."""
    edges = []
    for path in sorted((root / "worldfoundry/core").rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        package = module if parts[-1] == "__init__" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                targets = [
                    resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
                ]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            edges.extend((module, target) for target in targets if target.startswith("worldfoundry.core"))
    return edges


def test_control_plane_imports_remain_lightweight() -> None:
    """Package boundaries must not load optional model stacks in a fresh process."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import worldfoundry.core
import worldfoundry.core.acceleration
import worldfoundry.core.acceleration.profiles
import worldfoundry.core.acceleration.sol_engine
import worldfoundry.core.attention
import worldfoundry.core.configuration
import worldfoundry.core.io
import worldfoundry.core.nn
import worldfoundry.core.nn.transformer.contracts
import worldfoundry.core.utils
from worldfoundry.core.execution import inference, process, realtime, runtime_cache
from worldfoundry.core.platform import compile_cache
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
                        "worldfoundry.evals",
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


def test_core_layer_table_assigns_every_package_and_module() -> None:
    core = Path(__file__).resolve().parents[2] / "worldfoundry/core"
    entries = {path.stem for path in core.glob("*.py") if path.name != "__init__.py"}
    entries |= {path.name for path in core.iterdir() if (path / "__init__.py").is_file()}
    assigned = {unit.split(".")[0] for layer in _LAYERS for unit in layer}
    assert entries == assigned, f"unassigned: {sorted(entries - assigned)}; stale: {sorted(assigned - entries)}"


def test_core_imports_follow_the_layer_table() -> None:
    root = Path(__file__).resolve().parents[2]
    layer_of = {unit: index for index, layer in enumerate(_LAYERS) for unit in layer}
    upward = []
    graph: dict[str, set[str]] = {}
    for importer, imported in _core_import_edges(root):
        source, target = _core_unit(importer), _core_unit(imported)
        if source is None or source == target:
            continue
        if target is None or layer_of.get(target, len(_LAYERS)) > layer_of.get(source, len(_LAYERS)):
            upward.append((importer, imported))
            continue
        graph.setdefault(source, set()).add(target)
    assert not upward, f"core imports that point up the layer table: {upward}"
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise AssertionError(f"core units import each other: {error.args[1]}") from None


def test_subpackage_facades_export_only_owned_names() -> None:
    root = Path(__file__).resolve().parents[2]
    foreign = []
    for init in sorted((root / "worldfoundry/core").rglob("__init__.py")):
        package = ".".join(init.relative_to(root).parts[:-1])
        if package == "worldfoundry.core":
            continue
        targets = list(_lazy_tables(init).items())
        for node in ast.parse(init.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                module = node.module
                if module.startswith("worldfoundry."):
                    targets.extend((alias.name, module) for alias in node.names)
        foreign.extend(
            (package, name, target)
            for name, target in targets
            if target != package and not target.startswith(f"{package}.")
        )
    assert not foreign, f"package facades re-export names owned elsewhere: {foreign}"


def test_core_facade_names_resolve_to_their_defining_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    indirect = []
    for name, target in _lazy_tables(root / "worldfoundry/core/__init__.py").items():
        module = root.joinpath(*target.split("."))
        path = module.with_suffix(".py") if module.with_suffix(".py").is_file() else module / "__init__.py"
        if not _binds_at_top_level(path, name):
            indirect.append((name, target))
    assert not indirect, f"worldfoundry.core names that point at a re-exporting module: {indirect}"


def test_model_consumers_reference_existing_core_facade_exports() -> None:
    """Resolve lazy core names statically, including imports behind optional dependencies."""
    root = Path(__file__).resolve().parents[2]
    facades = {
        ".".join(path.relative_to(root).parts[:-1]): set(exports)
        for path in (root / "worldfoundry/core").rglob("__init__.py")
        if (exports := _lazy_tables(path))
    }
    unresolved = []
    for path in (root / "worldfoundry").rglob("*.py"):
        with tokenize.open(path) as stream:
            source = stream.read()
        if "worldfoundry.core" not in source:
            continue
        package = ".".join(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
            if module not in facades:
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in facades[module]:
                    continue
                target = root.joinpath(*module.split("."), alias.name)
                if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                    unresolved.append((str(path.relative_to(root)), node.lineno, module, alias.name))
    assert not unresolved, f"model consumers reference missing core facade exports: {unresolved}"
