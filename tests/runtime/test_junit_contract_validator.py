"""Whole-suite JUnit contracts reject skipped, substituted and partial runs."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tests/manual/validate_junit_contract.py"
_SPEC = importlib.util.spec_from_file_location("junit_validator_contract", _SCRIPT)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


@pytest.fixture
def contract(tmp_path):
    source = tmp_path / "tests/test_case.py"
    source.parent.mkdir()
    source.write_text("def test_one(): pass\ndef test_two(): pass\n")
    nodes = ["tests/test_case.py::test_one", "tests/test_case.py::test_two"]
    manifest = validator.build_manifest(nodes, tmp_path, ["tests/test_case.py"])
    suite = ET.Element("testsuite", tests="2", errors="0", failures="0", skipped="0")
    for item in manifest["tests"]:
        ET.SubElement(suite, "testcase", classname=item["classname"], name=item["name"])
    path = tmp_path / "junit.xml"
    ET.ElementTree(suite).write(path)
    return tmp_path, manifest, suite, path


def test_exact_passing_nodes_and_parameterized_class_addresses(contract):
    root, manifest, _, path = contract
    assert validator.validate_junit(manifest, path, root)["executed_tests"] == 2
    assert validator.junit_identity("tests/test_case.py::TestOwner::test_math[alpha::beta]") == (
        "tests.test_case.TestOwner", "test_math[alpha::beta]"
    )


@pytest.mark.parametrize("mutation", ["skipped", "failure", "error", "empty", "partial", "extra", "duplicate",
                                     "replacement", "no_identity", "bad_counter", "hidden_skip_counter"])
def test_rejected_results_or_same_size_substitution_cannot_pass(contract, mutation):
    _, manifest, suite, path = contract
    cases = list(suite)
    if mutation in {"skipped", "failure", "error"}:
        ET.SubElement(cases[0], mutation)
    elif mutation == "empty":
        suite.clear()
        suite.set("tests", "0")
    elif mutation == "partial":
        suite.remove(cases[1])
        suite.set("tests", "1")
    elif mutation == "extra":
        ET.SubElement(suite, "testcase", classname="tests.test_case", name="test_other")
        suite.set("tests", "3")
    elif mutation == "duplicate":
        cases[1].set("name", cases[0].get("name"))
    elif mutation == "replacement":
        cases[1].set("name", "test_other")
    elif mutation == "no_identity":
        cases[0].attrib.pop("classname")
    elif mutation == "bad_counter":
        suite.set("tests", "1")
    else:
        suite.set("skipped", "1")
    ET.ElementTree(suite).write(path)
    with pytest.raises(ValueError):
        validator.validate_junit(manifest, path)


def test_test_source_changes_after_collection_cannot_reuse_an_old_report(contract):
    root, manifest, _, path = contract
    (root / "tests/test_case.py").write_text("def test_changed(): pass\n")
    with pytest.raises(ValueError, match="changed after collection"):
        validator.validate_junit(manifest, path)


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "false_identity", "omit_source", "escape_root"])
def test_invalid_manifests_do_not_reduce_required_selection(contract, mutation):
    _, manifest, _, path = contract
    manifest = copy.deepcopy(manifest)
    if mutation == "empty":
        manifest["tests"] = []
    elif mutation == "duplicate":
        manifest["tests"].append(manifest["tests"][0])
    elif mutation == "false_identity":
        manifest["tests"][0]["name"] = "other"
    elif mutation == "omit_source":
        manifest["source_sha256"].clear()
    else:
        manifest["tests"][0]["nodeid"] = "../outside.py::test_one"
    with pytest.raises(ValueError):
        validator.validate_junit(manifest, path)


def _project(root):
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests/test_default.py"]\n')
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_default.py").write_text("def test_one(): pass\ndef test_two(): pass\n")
    (tests / "test_extra.py").write_text("def test_extra(): pass\n")


def _run(root, extra_args=()):
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    return subprocess.run([sys.executable, str(_SCRIPT), "run", "--source-root", str(root),
                           "--manifest", str(root / "manifest.json"), "--junit", str(root / "junit.xml"),
                           "--include-public-defaults", "--test", "tests/test_extra.py", "--", "-q", *extra_args],
                          env=env, cwd=root, capture_output=True, text=True)


def test_full_runner_includes_public_defaults_and_extra_contracts(tmp_path):
    _project(tmp_path)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["tests"]) == 3
    assert '"complete_selection_verified": true' in result.stdout


def test_test_args_partial_selection_cannot_bypass_postcheck(tmp_path):
    _project(tmp_path)
    result = _run(tmp_path, ["-k", "one"])
    assert result.returncode != 0
    assert "selection differs" in result.stdout
    assert len(json.loads((tmp_path / "manifest.json").read_text())["tests"]) == 3


@pytest.mark.parametrize("outcome", ["passed", "failure", "skipped"])
def test_real_unittest_subtests_preserve_parent_count_and_rejected_outcomes(tmp_path, outcome):
    _project(tmp_path)
    assertion = "self.assertTrue(True)"
    if outcome == "failure":
        assertion = "self.fail('subtest failure')"
    elif outcome == "skipped":
        assertion = "self.skipTest('subtest unavailable')"
    (tmp_path / "tests/test_extra.py").write_text(
        "import unittest\nclass TestChild(unittest.TestCase):\n"
        "    def test_children(self):\n"
        "        for child in range(2):\n"
        "            with self.subTest(child=child):\n"
        f"                {assertion}\n"
    )
    result = _run(tmp_path)
    if outcome == "passed":
        assert result.returncode == 0, result.stdout + result.stderr
        assert '"complete_selection_verified": true' in result.stdout
        suite = ET.parse(tmp_path / "junit.xml").getroot().find("testsuite")
        assert suite.get("tests") == str(len(list(suite.iter("testcase"))))
    else:
        assert result.returncode != 0
        assert '"status": "failed"' in result.stdout


@pytest.mark.parametrize("stage", ["collection", "execution"])
def test_missing_dependency_or_runtime_skip_is_rejected(tmp_path, stage):
    _project(tmp_path)
    skipped = "import pytest\n"
    skipped += "pytest.skip('missing dependency', allow_module_level=True)\n" if stage == "collection" else (
        "def test_extra(): pytest.skip('missing device')\n"
    )
    (tmp_path / "tests/test_extra.py").write_text(skipped)
    result = _run(tmp_path)
    assert result.returncode != 0
    if stage == "collection":
        assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"
    else:
        assert '"status": "failed"' in result.stdout and "skipped" in result.stdout
