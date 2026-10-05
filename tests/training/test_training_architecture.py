from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "worldfoundry" / "training"
_DEAD_GOVERNANCE_PATHS = (
    _TRAINING / "provenance.py",
    _TRAINING / "references",
    _TRAINING / "rl_scope.py",
    _ROOT / "configs" / "training" / "reference_traces",
    _ROOT / "worldfoundry" / "cli" / "training_commands" / "handlers" / "scope.py",
)
_RUNTIME_FACADES = {
    "worldfoundry.training.checkpoint",
    "worldfoundry.training.data",
    "worldfoundry.training.distributed",
    "worldfoundry.training.models",
    "worldfoundry.training.post_training",
    "worldfoundry.training.post_training.rewards",
    "worldfoundry.training.post_training.rl",
    "worldfoundry.training.recipes",
    "worldfoundry.training.tuning",
}


def _imports(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            result.append((node.lineno, node.module))
    return tuple(result)


def test_dead_governance_modules_do_not_return() -> None:
    assert [str(path.relative_to(_ROOT)) for path in _DEAD_GOVERNANCE_PATHS if path.exists()] == []


def test_runtime_contracts_do_not_require_governance_only_fields() -> None:
    from dataclasses import fields

    from worldfoundry.training.data.manifest import TrainingSample
    from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
    from worldfoundry.training.recipes.spec import TrainingRecipe

    assert {item.name for item in fields(TrainingSample)}.isdisjoint({"license", "provenance", "quality"})
    assert "metadata" not in {item.name for item in fields(TrainingRecipe)}
    assert "metadata" not in {item.name for item in fields(PostTrainingRecipe)}


def test_training_tree_has_no_module_package_name_collisions() -> None:
    collisions = [
        str(path.relative_to(_ROOT))
        for path in sorted(_TRAINING.rglob("*.py"))
        if path.name != "__init__.py" and path.with_suffix("").is_dir()
    ]
    assert collisions == []


def test_runtime_leaf_modules_do_not_import_public_facades() -> None:
    violations: list[str] = []
    for path in sorted(_TRAINING.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        for line, module_name in _imports(path):
            if module_name in _RUNTIME_FACADES:
                violations.append(f"{path.relative_to(_ROOT)}:{line}: {module_name}")
    assert violations == []


def test_generic_rewards_do_not_depend_on_rl_trajectory_types() -> None:
    rewards = _TRAINING / "post_training" / "rewards"
    violations = [
        f"{path.relative_to(_ROOT)}:{line}: {module_name}"
        for path in sorted(rewards.rglob("*.py"))
        for line, module_name in _imports(path)
        if module_name.startswith("worldfoundry.training.post_training.rl")
        or module_name == "rl"
        or module_name.startswith("rl.")
    ]
    assert violations == []


def test_shared_post_training_layer_does_not_depend_on_algorithms() -> None:
    shared = _TRAINING / "post_training" / "shared"
    violations = [
        f"{path.relative_to(_ROOT)}:{line}: {module_name}"
        for path in sorted(shared.rglob("*.py"))
        for line, module_name in _imports(path)
        if ".algorithms" in module_name
    ]
    assert violations == []


def test_training_code_does_not_use_numbered_version_names() -> None:
    pattern = re.compile(r"\bv1\b|version[ _-]?1", re.IGNORECASE)
    violations: list[str] = []
    roots = (
        _TRAINING,
        _ROOT / "worldfoundry" / "cli" / "training_commands",
        _ROOT / "configs" / "training",
        _ROOT / "configs" / "post_training",
    )
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".yaml", ".yml", ".json"}:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if pattern.search(line):
                    violations.append(f"{path.relative_to(_ROOT)}:{line_number}")
    assert violations == []


def test_training_runtime_does_not_add_content_hashing() -> None:
    violations: list[str] = []
    roots = (_TRAINING, _ROOT / "worldfoundry" / "cli" / "training_commands")
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8").lower()
            if any(fragment in source for fragment in ("hashlib", "sha256", "sha-256")):
                violations.append(str(path.relative_to(_ROOT)))
    assert violations == []
