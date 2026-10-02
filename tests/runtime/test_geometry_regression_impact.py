"""Changes must never silently escape the affected-model regression plan."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "manual"
spec = importlib.util.spec_from_file_location("impact_under_test", TOOLS / "geometry_regression_impact.py")
impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impact)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "worldfoundry/__init__.py": "",
        "worldfoundry/models/__init__.py": "from . import parent_side_effect\n",
        "worldfoundry/models/parent_side_effect.py": "",
        "worldfoundry/models/a.py": 'from . import conditional\nif False:\n    import worldfoundry.models.hidden\nBACKEND = "worldfoundry.models.lazy:Model"\n',
        "worldfoundry/models/b.py": "",
        "worldfoundry/models/conditional.py": "from ..utils import math\n",
        "worldfoundry/models/hidden.py": "",
        "worldfoundry/models/lazy.py": "",
        "worldfoundry/utils/math.py": "",
        "worldfoundry/dynamic/unused.py": "",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    matrix = root / "matrix.json"
    matrix.write_text(
        json.dumps({name: {"id": name, "target": f"worldfoundry.models.{name}:Model"} for name in ("a", "b")})
    )
    dependencies = root / "dependencies.json"
    dependencies.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shared_paths": ["worldfoundry/core/**", "matrix.json", "dependencies.json"],
                "ignored_paths": ["docs/**", "tests/**"],
                "components": {"a": ["worldfoundry/dynamic/**"], "b": ["worldfoundry/models/b.py"]},
                "cases": {"a": ["a"], "b": ["b"]},
            }
        )
    )
    git(root, "init", "-q")
    git(root, "config", "user.name", "Regression test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return root, matrix, dependencies


@pytest.mark.parametrize(
    "path",
    [
        "worldfoundry/models/a.py",
        "worldfoundry/models/conditional.py",
        "worldfoundry/models/hidden.py",
        "worldfoundry/models/lazy.py",
        "worldfoundry/utils/math.py",
        "worldfoundry/dynamic/unused.py",
    ],
)
def test_conditional_relative_lazy_and_declared_dynamic_dependencies_select_model(project, path):
    root, matrix, policy = project
    plan = impact.select_cases(matrix, policy, root, [path])
    assert plan["status"] == "planned"
    assert plan["selected_cases"] == ["a"]


@pytest.mark.parametrize(
    "path",
    [
        "worldfoundry/models/__init__.py",
        "worldfoundry/models/parent_side_effect.py",
        "worldfoundry/core/new_backend.py",
        "matrix.json",
    ],
)
def test_shared_and_parent_package_initializers_select_every_case(project, path):
    root, matrix, policy = project
    assert impact.select_cases(matrix, policy, root, [path])["selected_cases"] == ["a", "b"]


def test_docs_need_cpu_gates_without_claiming_inference_passed(project):
    root, matrix, policy = project
    plan = impact.select_cases(matrix, policy, root, ["docs/validation.md"])
    assert plan["status"] == "no_inference_changes"
    assert plan["selected_cases"] == []
    assert plan["required_checks"] == ["public-cpu", "inference-tensors"]


def test_unintegrated_code_selects_full_matrix_and_remains_uncovered(project):
    root, matrix, policy = project
    plan = impact.select_cases(matrix, policy, root, ["worldfoundry/new_model/runtime.py"])
    assert plan["status"] == "uncovered"
    assert plan["uncovered_paths"] == ["worldfoundry/new_model/runtime.py"]
    assert plan["selected_cases"] == ["a", "b"]


def test_bad_import_graph_is_not_green_even_for_docs_change(project):
    root, matrix, policy = project
    (root / "worldfoundry/models/a.py").write_text("invalid python: ?")
    plan = impact.select_cases(matrix, policy, root, ["docs/update.md"])
    assert plan["status"] == "uncovered"
    assert plan["selected_cases"] == ["a", "b"]
    assert "a" in plan["graph_errors"]


def test_recorded_imports_add_selection_without_replacing_declarations(project, tmp_path):
    root, matrix, policy = project
    evidence = tmp_path / "trace" / "b"
    evidence.mkdir(parents=True)
    (evidence / "manifest.json").write_text(
        json.dumps(
            {"status": "passed", "case": {"id": "b"}, "source_hashes": {"worldfoundry/dynamic/unused.py": "sha"}}
        )
    )
    plan = impact.select_cases(matrix, policy, root, ["worldfoundry/dynamic/unused.py"], evidence=evidence.parent)
    assert plan["selected_cases"] == ["a", "b"]


def test_rename_and_delete_keep_old_dependency_paths(project):
    root, _, _ = project
    base = git(root, "rev-parse", "HEAD")
    git(root, "mv", "worldfoundry/models/hidden.py", "worldfoundry/models/renamed module.py")
    git(root, "rm", "worldfoundry/models/b.py")
    git(root, "commit", "-qm", "rename and delete")
    assert impact.changed_paths(root, base, "HEAD") == [
        "worldfoundry/models/b.py",
        "worldfoundry/models/hidden.py",
        "worldfoundry/models/renamed module.py",
    ]


@pytest.mark.parametrize("value", ["", ".", "./module.py", "../module.py", "/tmp/module.py", "a\\b.py", "a//b.py"])
def test_invalid_paths_cannot_escape_repository(value):
    with pytest.raises(ValueError):
        impact.relative_path(value)


def test_new_case_requires_explicit_dependency_declaration(project):
    _, matrix, policy = project
    cases = json.loads(matrix.read_text())
    cases["new"] = {"id": "new", "target": "worldfoundry.models.a:Model"}
    matrix.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="explicit dependency"):
        impact.load_definitions(matrix, policy)


def test_snapshot_inside_other_git_checkout_uses_its_own_files(project):
    root, _, _ = project
    snapshot = root / "tmp" / "snapshot" / "worldfoundry"
    snapshot.mkdir(parents=True)
    (snapshot / "unique.py").write_text("")
    assert impact.ImportGraph(snapshot.parent).closure("worldfoundry.unique:Model") == {"worldfoundry/unique.py"}


def test_public_matrix_has_complete_declarations_and_known_hy2_path():
    root = TOOLS.parents[1]
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        ["worldfoundry/base_models/three_dimensions/point_clouds/hyworldmirror_2p0/runtime.py"],
    )
    assert not plan["graph_errors"]
    assert plan["status"] == "planned"
    assert "hyworldmirror-2" in plan["selected_cases"]
