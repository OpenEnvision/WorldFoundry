"""Changes must never silently escape the affected-model regression plan."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "manual"
spec = importlib.util.spec_from_file_location("impact_under_test", TOOLS / "geometry_regression_impact.py")
impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impact)


@pytest.mark.parametrize("path", [
    "tests/synthesis/test_matrix_game_2_checkpoint_conditioned_trajectory.py",
    "tests/synthesis/mg2_historical_reference.py",
])
def test_mg2_numerical_contract_changes_require_real_model_replay(path):
    root = TOOLS.parents[1]
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json", root, [path],
    )
    assert plan["status"] == "planned"
    assert plan["selected_cases"] == ["matrix-game2-controls-short"]
    assert not plan["uncovered_paths"]


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


@pytest.mark.parametrize("path,case_id", [
    ("worldfoundry/pipelines/cosmos/pipeline_cosmos_transfer2p5.py", "cosmos-transfer25-2b-edge-short"),
    ("worldfoundry/pipelines/gamma_world/pipeline_gamma_world.py", "gamma-world-causal-few-step-189f"),
    ("worldfoundry/synthesis/visual_generation/open_oasis/utils.py", "oasis500m-video-short"),
])
def test_public_world_model_changes_have_their_own_replay_case(path, case_id):
    root = TOOLS.parents[1]
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        [path],
    )
    assert plan["status"] == "planned"
    assert not plan["uncovered_paths"] and not plan["graph_errors"]
    assert case_id in plan["selected_cases"]
    assert any(reason["path"] == path and reason.get("components") for reason in plan["reasons"][case_id])


def test_public_matrix_still_rejects_an_unintegrated_world_model():
    root = TOOLS.parents[1]
    path = "worldfoundry/pipelines/unintegrated_world/pipeline_world.py"
    plan = impact.select_cases(
        TOOLS / "geometry_regression_cases.json",
        TOOLS / "geometry_regression_dependencies.json",
        root,
        [path],
    )
    assert plan["status"] == "uncovered"
    assert plan["uncovered_paths"] == [path]


VERIFICATION_SUITE = "tests/manual/geometry_regression_suite.py"
VERIFICATION_IMPACT = "tests/manual/geometry_regression_impact.py"
MG2_CASE = "matrix-game2-controls-short"
VERIFICATION_CONTRACTS = [
    "tests/runtime/test_geometry_regression_impact.py",
    "tests/runtime/test_geometry_regression_suite.py",
]


@pytest.fixture
def verification_project(project):
    root, _, _ = project
    files = {
        VERIFICATION_SUITE: "# Original replay orchestration\n",
        VERIFICATION_IMPACT: "# Original affected-case planning\n",
        "tests/manual/geometry_regression.py": "# Original numerical exporter\n",
        VERIFICATION_CONTRACTS[0]: "def test_complete_diff_proof(): pass\n",
        VERIFICATION_CONTRACTS[1]: "def test_replay_snapshot_safety(): pass\n",
        "docs/orchestration.md": "original guide\n",
        ".github/workflows/ci.yml": "name: validation\n",
        "Makefile": "test:\n\tpython -m pytest\n",
        ".gitignore": "worldfoundry/ignored_shadow.py\nignored_shadow.py\n",
        "MANIFEST.in": "include LICENSE\n",
        "LICENSE": "original project license\n",
        "requirements/model.txt": "original-runtime==1\n",
        "envs/model.yaml": "runtime: original\n",
        "assets/reference.bin": "original input bytes\n",
        "thirdparty/vendor/source.py": "# Original external runtime\n",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    matrix = root / "tests/manual/geometry_regression_cases.json"
    cases = {name: {"id": name, "target": f"worldfoundry.models.{name}:Model"} for name in ("a", "b")}
    cases[MG2_CASE] = {"id": MG2_CASE, "target": "worldfoundry.models.a:Model"}
    matrix.write_text(json.dumps(cases))
    dependencies = root / "tests/manual/geometry_regression_dependencies.json"
    dependencies.write_text(json.dumps({
        "schema_version": 1,
        "shared_paths": ["worldfoundry/core/**", "tests/manual/geometry_regression*.py",
                         str(matrix.relative_to(root)), str(dependencies.relative_to(root)),
                         "MANIFEST.in", "requirements/**", "envs/**", "assets/**", "thirdparty/**", "LICENSE"],
        "ignored_paths": ["tests/**", "docs/**", ".github/**", "Makefile"],
        "components": {"a": ["worldfoundry/models/a.py"], "b": ["worldfoundry/models/b.py"]},
        "cases": {"a": ["a"], "b": ["b"], MG2_CASE: ["a"]},
    }))
    git(root, "add", ".")
    git(root, "commit", "-qm", "complete committed verification fixture")
    return root, matrix, dependencies, git(root, "rev-parse", "HEAD")


def commit_verification_change(project, *, tools=(VERIFICATION_SUITE,), extra=None):
    root, _, _, base = project
    for name in tools:
        (root / name).write_text("# Changed verification orchestration\n")
    for name, value in (extra or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    git(root, "add", ".")
    git(root, "commit", "-qm", "verification changes")
    head = git(root, "rev-parse", "HEAD")
    return head, impact.changed_paths(root, base, head)


def verification_proofs(project, *, head, changes, base=None):
    root, matrix, dependencies, original = project
    return impact.verification_tooling_proofs(root, matrix, dependencies, changes,
                                              base=original if base is None else base, head=head)


def assert_conservative_verification_fallback(project, *, head, changes, base=None):
    root, matrix, dependencies, original = project
    reference = original if base is None else base
    assert impact.verification_tooling_proofs(root, matrix, dependencies, changes, base=reference, head=head) == []
    plan = impact.select_cases(matrix, dependencies, root, changes, base=reference, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", MG2_CASE}
    assert not plan["cpu_only_changes"]
    assert not plan["required_cpu_contracts"]
    return plan


@pytest.mark.parametrize("tool,contract", [(VERIFICATION_SUITE, VERIFICATION_CONTRACTS[1]),
                                          (VERIFICATION_IMPACT, VERIFICATION_CONTRACTS[0])])
def test_committed_verification_tool_change_requires_cpu_contract_and_real_mg2_replay(verification_project, tool, contract):
    root, matrix, dependencies, base = verification_project
    head, changes = commit_verification_change(verification_project, tools=(tool,))
    proofs = verification_proofs(verification_project, head=head, changes=changes)
    assert len(proofs) == 1
    proof = proofs[0]
    assert proof["path"] == tool
    assert proof["base_revision"] == base and proof["head_revision"] == head
    assert proof["source_sha256"] == hashlib.sha256((root / tool).read_bytes()).hexdigest()
    assert proof["required_cpu_contracts"] == [contract]
    assert len(proof["immutable_runtime_tree_sha256"]) == 64
    assert int(proof["immutable_runtime_tree_sha256"], 16) >= 0
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert plan["status"] == "planned" and plan["selected_cases"] == [MG2_CASE]
    assert plan["cpu_only_changes"] == proofs
    assert plan["required_cpu_contracts"] == [contract]
    assert plan["required_checks"] == ["public-cpu", "inference-tensors", "real-weight-replays"]
    assert plan["reasons"][MG2_CASE] == [{"kind": "verified_tooling_pipeline_replay", "paths": [tool]}]
    assert not plan["uncovered_paths"] and not plan["graph_errors"]


def test_two_changed_verification_tools_require_both_complete_cpu_contracts(verification_project):
    root, matrix, dependencies, base = verification_project
    tools = (VERIFICATION_IMPACT, VERIFICATION_SUITE)
    head, changes = commit_verification_change(verification_project, tools=tools)
    proofs = verification_proofs(verification_project, head=head, changes=changes)
    assert {item["path"] for item in proofs} == set(tools)
    assert len({item["immutable_runtime_tree_sha256"] for item in proofs}) == 1
    assert all(item["required_cpu_contracts"] == VERIFICATION_CONTRACTS for item in proofs)
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert plan["selected_cases"] == [MG2_CASE]
    assert plan["required_cpu_contracts"] == VERIFICATION_CONTRACTS


def test_allowed_documentation_tests_workflow_and_make_edits_preserve_runtime_proof(verification_project):
    root, matrix, dependencies, base = verification_project
    head, changes = commit_verification_change(verification_project)
    original = verification_proofs(verification_project, head=head, changes=changes)[0]
    extra = {"docs/orchestration.md": "updated guide\n", ".github/workflows/ci.yml": "name: changed validation\n",
             "Makefile": "test:\n\tpython -m pytest -q\n", "tests/runtime/additional_contract.py": "# Additional checks\n"}
    head, changes = commit_verification_change(verification_project, extra=extra)
    proof = verification_proofs(verification_project, head=head, changes=changes)[0]
    assert proof["immutable_runtime_tree_sha256"] == original["immutable_runtime_tree_sha256"]
    assert proof["source_sha256"] == original["source_sha256"]
    assert proof["head_revision"] == head and proof["base_revision"] == base
    assert impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)["selected_cases"] == [MG2_CASE]


@pytest.mark.parametrize("path", ["worldfoundry/models/b.py", "MANIFEST.in", "requirements/model.txt",
                                  "envs/model.yaml", "assets/reference.bin", "thirdparty/vendor/source.py", "LICENSE"])
def test_mixed_code_environment_packaging_or_asset_changes_retain_full_replay_selection(verification_project, path):
    head, changes = commit_verification_change(verification_project, extra={path: "# Changed production bytes\n"})
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("operation", ["mode", "delete", "rename", "new_asset", "symlink"])
def test_runtime_modes_membership_and_symlink_changes_cannot_use_cpu_proof(verification_project, operation):
    root, _, _, base = verification_project
    path = root / "assets/reference.bin"
    if operation == "mode":
        path.chmod(0o755)
    elif operation == "delete":
        path.unlink()
    elif operation == "rename":
        path.rename(root / "assets/renamed-reference.bin")
    elif operation == "new_asset":
        (root / "assets/new-reference.bin").write_text("additional input\n")
    else:
        path.unlink()
        path.symlink_to("../LICENSE")
    head, changes = commit_verification_change(verification_project)
    assert changes == impact.changed_paths(root, base, head)
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("definition", ["matrix", "dependencies"])
def test_committed_case_or_dependency_definition_change_cannot_be_exempted(verification_project, definition):
    _, matrix, dependencies, _ = verification_project
    path = matrix if definition == "matrix" else dependencies
    data = json.loads(path.read_text())
    if definition == "matrix":
        data["a"]["scope"] = "Changed replay definition"
    else:
        data["ignored_paths"].append("new-ignored-path/**")
    path.write_text(json.dumps(data))
    head, changes = commit_verification_change(verification_project)
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("path_kind", ["tool", "matrix", "dependencies", "contract"])
def test_uncommitted_inspected_tool_definition_or_contract_cannot_authorize_proof(verification_project, path_kind):
    root, matrix, dependencies, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    path = {"tool": root / VERIFICATION_SUITE, "matrix": matrix, "dependencies": dependencies,
            "contract": root / VERIFICATION_CONTRACTS[1]}[path_kind]
    if path_kind in {"matrix", "dependencies"}:
        path.write_text(json.dumps(json.loads(path.read_text()), indent=2) + "\n")
    else:
        path.write_text(path.read_text() + "# Edited after commit\n")
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("change", ["missing_tool", "missing_doc", "extra_path", "hidden_runtime"])
def test_incomplete_or_invented_changed_path_lists_cannot_authorize_proof(verification_project, change):
    extra = {"docs/orchestration.md": "updated guide\n"}
    if change == "hidden_runtime":
        extra["worldfoundry/models/b.py"] = "# Hidden production edit\n"
    head, changes = commit_verification_change(verification_project, extra=extra)
    if change == "missing_tool":
        supplied = [item for item in changes if item != VERIFICATION_SUITE]
    elif change == "missing_doc":
        supplied = [item for item in changes if item != "docs/orchestration.md"]
    elif change == "extra_path":
        supplied = [*changes, "docs/invented.md"]
    else:
        supplied = [item for item in changes if item != "worldfoundry/models/b.py"]
    assert verification_proofs(verification_project, head=head, changes=supplied) == []


@pytest.mark.parametrize("base", [None, "0" * 40, "missing-base-ref"])
def test_missing_or_unresolvable_history_cannot_turn_shared_tool_changes_into_cpu_only(verification_project, base):
    root, matrix, dependencies, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    assert impact.verification_tooling_proofs(root, matrix, dependencies, changes, base=base, head=head) == []
    plan = impact.select_cases(matrix, dependencies, root, changes, base=base, head=head)
    assert set(plan["selected_cases"]) == {"a", "b", MG2_CASE}
    assert not plan["cpu_only_changes"]


def test_replay_exporter_is_not_in_the_exact_verification_tool_allowance(verification_project):
    head, changes = commit_verification_change(verification_project, tools=("tests/manual/geometry_regression.py",))
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("path_kind", ["tool", "contract"])
@pytest.mark.parametrize("operation", ["missing", "symlink"])
def test_missing_or_symlinked_verification_implementation_cannot_be_exempted(verification_project, path_kind, operation):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    path = root / (VERIFICATION_SUITE if path_kind == "tool" else VERIFICATION_CONTRACTS[1])
    path.unlink()
    if operation == "symlink":
        path.symlink_to(root / "LICENSE")
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_fixture_without_mg2_has_only_cpu_proof_and_claims_no_inference_pass(verification_project):
    root, matrix, dependencies, _ = verification_project
    cases, policy = json.loads(matrix.read_text()), json.loads(dependencies.read_text())
    cases.pop(MG2_CASE)
    policy["cases"].pop(MG2_CASE)
    matrix.write_text(json.dumps(cases))
    dependencies.write_text(json.dumps(policy))
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture without MG2")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project)
    plan = impact.select_cases(matrix, dependencies, root, changes, base=project[3], head=head)
    assert plan["status"] == "no_inference_changes" and plan["selected_cases"] == []
    assert plan["required_cpu_contracts"] == [VERIFICATION_CONTRACTS[1]]
    assert plan["required_checks"] == ["public-cpu", "inference-tensors"]


@pytest.mark.parametrize("operation", ["modify", "delete", "mode"])
def test_dirty_production_checkout_cannot_use_an_immutable_git_tree_proof(verification_project, operation):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    source = root / "worldfoundry/models/b.py"
    if operation == "modify":
        source.write_text("# Uncommitted model implementation\n")
    elif operation == "delete":
        source.unlink()
    else:
        source.chmod(0o755)
    assert git(root, "rev-parse", "HEAD") == head
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("ignored", [False, True])
def test_untracked_python_shadow_including_ignored_files_cannot_use_verification_proof(verification_project, ignored):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    name = "ignored_shadow.py" if ignored else "untracked_shadow.py"
    shadow = root / "worldfoundry" / name
    shadow.write_text("# Runtime module absent from the recorded commit\n")
    status = git(root, "status", "--porcelain", "--", str(shadow.relative_to(root)))
    assert (not status) is ignored
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_untracked_interpreter_cache_is_not_confused_with_python_source_shadow(verification_project):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    cache = root / "worldfoundry/models/__pycache__"
    cache.mkdir()
    (cache / "a.cpython-311.pyc").write_bytes(b"generated interpreter cache")
    assert verification_proofs(verification_project, head=head, changes=changes)


@pytest.mark.parametrize("excluded_root", ["docs", "tests", ".github"])
def test_runtime_symlink_to_changed_excluded_file_cannot_hide_runtime_bytes(verification_project, excluded_root):
    root, matrix, dependencies, _ = verification_project
    target = root / excluded_root / "runtime-input.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("original input consumed through a runtime alias\n")
    alias = root / "assets/aliased-input.txt"
    alias.symlink_to("../" + excluded_root + "/runtime-input.txt")
    git(root, "add", ".")
    git(root, "commit", "-qm", "runtime alias to excluded input")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project, extra={str(target.relative_to(root)): "changed runtime input\n"})
    assert alias.read_text() == "changed runtime input\n"
    assert_conservative_verification_fallback(project, head=head, changes=changes)


def test_runtime_symlink_to_an_immutable_tracked_license_is_validated(verification_project):
    root, matrix, dependencies, _ = verification_project
    alias = root / "thirdparty/vendor/LICENSE"
    alias.symlink_to("../../LICENSE")
    git(root, "add", ".")
    git(root, "commit", "-qm", "immutable runtime license alias")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project)
    proofs = verification_proofs(project, head=head, changes=changes)
    assert len(proofs) == 1 and proofs[0]["path"] == VERIFICATION_SUITE
    assert impact.select_cases(matrix, dependencies, root, changes, base=project[3], head=head)["selected_cases"] == [MG2_CASE]


def test_excluded_intermediate_symlink_cannot_redirect_an_unchanged_runtime_alias(verification_project):
    root, matrix, dependencies, _ = verification_project
    route = root / "docs/runtime-route.py"
    route.symlink_to("../worldfoundry/models/a.py")
    alias = root / "worldfoundry/runtime-alias.py"
    alias.symlink_to("../docs/runtime-route.py")
    git(root, "add", ".")
    git(root, "commit", "-qm", "runtime alias through excluded routing file")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    original = alias.read_bytes()
    route.unlink()
    route.symlink_to("../worldfoundry/models/b.py")
    head, changes = commit_verification_change(project)
    assert alias.read_bytes() != original
    assert git(root, "diff", project[3], head, "--", "worldfoundry") == ""
    assert_conservative_verification_fallback(project, head=head, changes=changes)


def test_immutable_symlink_chain_stays_inside_the_inspected_runtime_tree(verification_project):
    root, matrix, dependencies, _ = verification_project
    (root / "LICENSE_ALIAS").symlink_to("LICENSE")
    (root / "thirdparty/vendor/LICENSE").symlink_to("../../LICENSE_ALIAS")
    git(root, "add", ".")
    git(root, "commit", "-qm", "immutable runtime license chain")
    project = (root, matrix, dependencies, git(root, "rev-parse", "HEAD"))
    head, changes = commit_verification_change(project)
    assert verification_proofs(project, head=head, changes=changes)


@pytest.mark.parametrize("relative", ["torch.py", "numpy/__init__.py", "thirdparty/untracked_code.py",
                                      "shadow_extension.so", "shadow_types.pyi", "shadow_script.pyw",
                                      "shadow_extension.pyd", "ignored_shadow.py"])
def test_untracked_importable_code_outside_worldfoundry_cannot_shadow_a_verified_runtime(verification_project, relative):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    shadow = root / relative
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(b"untracked importable runtime shadow")
    if relative == "ignored_shadow.py":
        assert git(root, "status", "--porcelain", "--", relative) == ""
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_untracked_package_symlink_cannot_import_from_an_excluded_staging_directory(verification_project):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    staging = root / "tmp/alternate-numpy"
    staging.mkdir(parents=True)
    (staging / "__init__.py").write_text("# Package outside recorded production source\n")
    (root / "numpy").symlink_to("tmp/alternate-numpy", target_is_directory=True)
    assert_conservative_verification_fallback(verification_project, head=head, changes=changes)


def test_untracked_logs_and_excluded_staging_code_do_not_change_production_proof(verification_project):
    root, _, _, _ = verification_project
    head, changes = commit_verification_change(verification_project)
    (root / "root.log").write_text("validation progress\n")
    staging = root / "tmp/temporary_validation.py"
    staging.parent.mkdir()
    staging.write_text("# Temporary orchestration outside runtime import roots\n")
    assert verification_proofs(verification_project, head=head, changes=changes)
