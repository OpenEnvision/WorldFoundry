"""Relocation proofs cannot turn edited or unsupported models into covered cases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "manual"


def tool(name):
    spec = importlib.util.spec_from_file_location("relocation_test_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


impact = tool("geometry_regression_impact")
gate = tool("inference_regression_gate")
suite = tool("geometry_regression_suite")
OLD = "worldfoundry/synthesis/visual_generation/world_model"
NEW = "worldfoundry/synthesis/visual_generation"
CONTRACT = "tests/test_relocated_resources.py"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write(root, path, data):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if isinstance(data, bytes) else data.encode())


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    write(root, "worldfoundry/a.py", "class Model: pass\n")
    write(root, "matrix.json", json.dumps({"a": {"id": "a", "target": "worldfoundry.a:Model"}}))
    rule = {
        "from_root": OLD,
        "to_root": NEW,
        "packages": ["model"],
        "files": [],
        "rewrite_only_paths": ["worldfoundry/pipeline.py"],
        "relative_import_rewrites": [],
        "deleted_metadata": [],
        "required_cpu_contracts": [CONTRACT],
        "reason": "Declared relocation only; no model numerical certification.",
    }
    policy = {
        "schema_version": 1,
        "shared_paths": ["matrix.json", "dependencies.json"],
        "ignored_paths": ["tests/**"],
        "components": {"a": ["worldfoundry/a.py"]},
        "cases": {"a": ["a"]},
        "relocation_contracts": [rule],
        "cpu_only_components": [{
            "paths": ["benchmarks/inference/correctness.py"],
            "reason": "Numerical benchmark acceptance tooling.",
            "required_cpu_contracts": [CONTRACT],
        }],
    }
    write(root, "dependencies.json", json.dumps(policy))
    write(root, CONTRACT, "def test_resource(): pass\n")
    for args in [("init", "-q"), ("config", "user.name", "Relocation test"),
                 ("config", "user.email", "test@example.invalid")]:
        git(root, *args)
    return root, policy


def commit(root):
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture change")
    return git(root, "rev-parse", "HEAD")


def move(root, relative):
    new = root / NEW / relative
    new.parent.mkdir(parents=True, exist_ok=True)
    (root / OLD / relative).rename(new)


def plan(root, base, *, changes=None):
    head = git(root, "rev-parse", "HEAD")
    paths = impact.changed_paths(root, base, head) if changes is None else changes
    result = impact.select_cases(root / "matrix.json", root / "dependencies.json", root, paths, base=base, head=head)
    result.update(base_revision=base, head_revision=head)
    return result


@pytest.mark.parametrize("relative,data", [
    ("model/block.py", "def denoise(x): return x * 2\n"),
    ("model/stats.json", '{"action_min": [-1, -2]}\n'),
    ("model/indices.npy", b"\x00\xff\x01\x80"),
])
def test_exact_commit_blobs_prove_source_and_binary_resource_moves(project, relative, data):
    root, _ = project
    write(root, OLD + "/" + relative, data)
    base = commit(root)
    move(root, relative)
    commit(root)
    result = plan(root, base)
    assert result["status"] == "no_inference_changes"
    assert result["uncovered_paths"] == []
    assert result["selected_cases"] == []
    proof, = result["proven_relocations"]
    assert proof["old_path"] == OLD + "/" + relative
    assert proof["new_path"] == NEW + "/" + relative
    assert proof["kind"] == "identical_bytes"
    assert proof["old_sha256"] == proof["new_sha256"]
    assert result["required_cpu_contracts"] == [CONTRACT]


def test_exact_namespace_rewrite_accepts_imports_and_lazy_targets(project):
    root, _ = project
    before = ('from worldfoundry.synthesis.visual_generation.world_model.model import Backend\n'
              'TARGET = "worldfoundry.synthesis.visual_generation.world_model.model:Backend"\n'
              'def denoise(x): return x * 2\n')
    write(root, OLD + "/model/block.py", before)
    base = commit(root)
    move(root, "model/block.py")
    write(root, NEW + "/model/block.py", before.replace("visual_generation.world_model.", "visual_generation."))
    commit(root)
    result = plan(root, base)
    assert result["status"] == "no_inference_changes"
    proof, = result["proven_relocations"]
    assert proof["kind"] == "declared_namespace_rewrite"
    assert proof["old_sha256"] != proof["new_sha256"]


@pytest.mark.parametrize("edit", [" * 3", " + 1", " / 2"])
def test_high_similarity_rename_with_any_numerical_edit_remains_uncovered(project, edit):
    root, _ = project
    before = "# retained documentation\n" * 30 + "def denoise(x): return x * 2\n"
    write(root, OLD + "/model/block.py", before)
    base = commit(root)
    move(root, "model/block.py")
    write(root, NEW + "/model/block.py", before.replace(" * 2", edit))
    commit(root)
    assert git(root, "diff", "--name-status", "--find-renames", base, "HEAD").startswith("R")
    result = plan(root, base)
    assert result["status"] == "uncovered"
    assert result["proven_relocations"] == []
    assert result["uncovered_paths"] == sorted([OLD + "/model/block.py", NEW + "/model/block.py"])
    assert result["selected_cases"] == ["a"]


def test_move_outside_declared_packages_is_uncovered(project):
    root, _ = project
    write(root, OLD + "/unlisted/block.py", "def denoise(x): return x\n")
    base = commit(root)
    move(root, "unlisted/block.py")
    commit(root)
    assert plan(root, base)["status"] == "uncovered"


def test_explicit_paths_without_commit_history_cannot_claim_a_relocation(project):
    root, _ = project
    write(root, NEW + "/model/block.py", "def denoise(x): return x\n")
    commit(root)
    result = impact.select_cases(root / "matrix.json", root / "dependencies.json", root,
                                 [OLD + "/model/block.py", NEW + "/model/block.py"])
    assert result["status"] == "uncovered"
    assert result["proven_relocations"] == []


def test_dirty_relocated_source_does_not_match_the_proven_head_blob(project):
    root, _ = project
    write(root, OLD + "/model/block.py", "def denoise(x): return x\n")
    base = commit(root)
    move(root, "model/block.py")
    commit(root)
    write(root, NEW + "/model/block.py", "def denoise(x): return x + 1\n")
    assert plan(root, base)["status"] == "uncovered"


def test_resource_reader_keeps_file_relative_resolution_after_proven_move(project):
    root, _ = project
    reader = ('import json\nfrom pathlib import Path\n'
              'def actions(): return json.loads((Path(__file__).parent / "data/stats.json").read_text())\n')
    write(root, OLD + "/model/runtime.py", reader)
    write(root, OLD + "/model/data/stats.json", '{"action_min": [-1.0, -2.0]}\n')
    base = commit(root)
    move(root, "model/runtime.py")
    move(root, "model/data/stats.json")
    commit(root)
    result = plan(root, base)
    assert result["status"] == "no_inference_changes"
    assert len(result["proven_relocations"]) == 2
    spec = importlib.util.spec_from_file_location("moved_reader", root / NEW / "model/runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.actions() == {"action_min": [-1.0, -2.0]}


@pytest.mark.parametrize("mutate", [False, True])
def test_deleted_metadata_requires_the_exact_declared_content_digest(project, mutate):
    root, policy = project
    metadata = b'"""Obsolete container docstring only."""\n'
    path = OLD + "/__init__.py"
    policy["relocation_contracts"][0]["deleted_metadata"] = [{
        "path": path, "sha256": hashlib.sha256(metadata).hexdigest(), "reason": "Docstring-only metadata.",
    }]
    write(root, "dependencies.json", json.dumps(policy))
    write(root, path, metadata + (b"def hidden_model(): return 0\n" if mutate else b""))
    base = commit(root)
    (root / path).unlink()
    commit(root)
    result = plan(root, base)
    assert result["status"] == ("uncovered" if mutate else "no_inference_changes")
    assert bool(result["proven_relocations"]) is not mutate


@pytest.mark.parametrize("numerical_edit", [False, True])
def test_unmoved_namespace_only_import_fix_cannot_hide_a_runtime_edit(project, numerical_edit):
    root, _ = project
    before = ('from worldfoundry.synthesis.visual_generation.world_model.model import Backend\n'
              'def denoise(x): return x * 2\n')
    write(root, "worldfoundry/pipeline.py", before)
    base = commit(root)
    after = before.replace("visual_generation.world_model.", "visual_generation.")
    if numerical_edit:
        after = after.replace(" * 2", " * 3")
    write(root, "worldfoundry/pipeline.py", after)
    commit(root)
    assert plan(root, base)["status"] == ("uncovered" if numerical_edit else "no_inference_changes")


def test_exact_benchmark_tooling_contract_does_not_whitelist_the_benchmark_tree(project):
    root, _ = project
    write(root, "benchmarks/inference/correctness.py", "def budget(): return 0\n")
    base = commit(root)
    result = plan(root, base, changes=["benchmarks/inference/correctness.py"])
    assert result["status"] == "no_inference_changes"
    assert result["required_cpu_contracts"] == [CONTRACT]
    assert len(result["cpu_only_changes"]) == 1
    result = plan(root, base, changes=["benchmarks/inference/new_model.py"])
    assert result["status"] == "uncovered"


def test_cpu_only_wildcard_declaration_is_rejected(project):
    root, policy = project
    policy["cpu_only_components"][0]["paths"] = ["benchmarks/**"]
    write(root, "dependencies.json", json.dumps(policy))
    with pytest.raises(ValueError, match="wildcard"):
        impact.load_definitions(root / "matrix.json", root / "dependencies.json")


def test_missing_concrete_cpu_contract_cannot_authorize_a_relocation(project):
    root, _ = project
    write(root, OLD + "/model/block.py", "def denoise(x): return x\n")
    base = commit(root)
    move(root, "model/block.py")
    (root / CONTRACT).unlink()
    commit(root)
    with pytest.raises(ValueError, match="no inspected implementation"):
        plan(root, base)


def test_snapshot_and_merge_gate_recompute_proofs_and_reject_omitted_cpu_contracts(project, tmp_path):
    root, _ = project
    write(root, OLD + "/model/block.py", "def denoise(x): return x\n")
    base = commit(root)
    move(root, "model/block.py")
    commit(root)
    result = plan(root, base)
    gate.verify_checkout_plan(result, root, root / "matrix.json", root / "dependencies.json")
    snapshot = tmp_path / "snapshot"
    shutil.copytree(root, snapshot, ignore=shutil.ignore_patterns(".git"))
    assert suite.validate_plan(result, snapshot / "matrix.json", snapshot / "dependencies.json",
                               snapshot, result["source_tree"], history_root=root) == []
    result["required_cpu_contracts"] = []
    with pytest.raises(ValueError, match="required CPU proof"):
        gate.verify_checkout_plan(result, root, root / "matrix.json", root / "dependencies.json")
    with pytest.raises(ValueError, match="required CPU proof"):
        suite.validate_plan(result, snapshot / "matrix.json", snapshot / "dependencies.json",
                            snapshot, result["source_tree"], history_root=root)
