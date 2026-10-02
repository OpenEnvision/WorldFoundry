from __future__ import annotations

import ast
import json
from pathlib import Path

_BANNED_UPSTREAM_MODULES = frozenset(
    {
        "cosmos_framework",
        "cosmos_predict2",
        "cosmos_rl",
        "dancegrpo",
        "diffusion_nft",
        "diffusiondpo",
        "dynamicrafter",
        "fastvideo",
        "flow_grpo",
        "ltx_trainer",
        "ltxv_trainer",
        "lvdm",
        "mixgrpo",
        "t2v_turbo",
        "unirl",
        "verl",
        "verl_omni",
    }
)


def _runtime_sources(root: Path) -> tuple[Path, ...]:
    training = root / "worldfoundry/training"
    training_sources = tuple(training.rglob("*.py"))
    cli_sources = tuple((root / "worldfoundry/cli/training_commands").rglob("*.py"))
    return tuple(
        sorted(
            (*training_sources, *cli_sources, root / "worldfoundry/cli/training.py"),
            key=lambda path: path.as_posix(),
        )
    )


def test_native_training_runtime_cannot_import_or_launch_reference_repositories() -> None:
    root = Path(__file__).resolve().parents[2]
    providers = root / "worldfoundry/training/providers"

    assert not providers.exists()
    for path in _runtime_sources(root):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                assert roots.isdisjoint(_BANNED_UPSTREAM_MODULES), path
                assert "subprocess" not in roots, path
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0:
                    imported_root = node.module.split(".", 1)[0]
                    assert imported_root not in _BANNED_UPSTREAM_MODULES, path
                    assert imported_root != "subprocess", path
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                    raise AssertionError(f"native training runtime cannot launch subprocesses: {path}")
        assert "external_launcher" not in source
        assert "sys.path" not in source


def test_native_training_runtime_has_no_source_control_or_trainer_disclaimer_hooks() -> None:
    root = Path(__file__).resolve().parents[2]
    banned_fragments = (
        "executes_model_author_trainer",
        "git rev-parse",
        "training-provenance",
        "worldfoundry_source_revision",
    )

    for path in _runtime_sources(root):
        source = path.read_text(encoding="utf-8").lower()
        for fragment in banned_fragments:
            assert fragment not in source, f"{fragment!r} returned in {path}"


def test_source_formula_fixtures_are_data_only_and_outside_runtime_import_graph() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture_root = root / "tests/training/fixtures/source_formulas"
    fixtures = tuple(sorted(fixture_root.glob("*.json")))

    assert fixtures
    test_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "tests/training").rglob("test_*.py"))
    )
    for path in fixtures:
        assert path.is_file() and not path.is_symlink()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"inputs", "expected", "atol", "rtol"}
        assert path.name in test_sources, f"orphan source-formula fixture: {path}"


def test_reference_trainers_are_not_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    dependency_files = (
        root / "pyproject.toml",
        root / "requirements/worldfoundry-unified.txt",
    )
    normalized = "\n".join(path.read_text(encoding="utf-8").lower().replace("-", "_") for path in dependency_files)

    for package in _BANNED_UPSTREAM_MODULES:
        assert package not in normalized, package
