"""Collect a complete pytest contract, then require exact passing JUnit nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def junit_identity(nodeid: str) -> tuple[str, str]:
    """Match pytest's JUnit address encoding, including parameter separators."""
    path, bracket, parameters = nodeid.partition("[")
    names = path.split("::")
    if len(names) < 2 or any(not part for part in names):
        raise ValueError(f"Invalid collected test node: {nodeid}")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + parameters
    return ".".join(names[:-1]), names[-1]


def build_manifest(nodeids: list[str], root: Path, selection: list[str]) -> dict:
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise ValueError("Complete contract collection must contain distinct test nodes")
    tests, sources = [], {}
    root = root.resolve()
    for nodeid in nodeids:
        classname, name = junit_identity(nodeid)
        source = nodeid.partition("::")[0]
        path = (root / source).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Collected test source escapes or is absent from the checkout: {source}")
        sources[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        tests.append({"nodeid": nodeid, "classname": classname, "name": name})
    if len({(item["classname"], item["name"]) for item in tests}) != len(tests):
        raise ValueError("Collected nodes have ambiguous JUnit identities")
    return {"schema_version": 1, "status": "collected", "source_root": str(root),
            "selection": selection, "tests": tests, "source_sha256": sources}


def _tag(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def validate_junit(manifest: dict, junit: Path, source_root: Path | None = None) -> dict:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "collected":
        raise ValueError("A successful complete collection manifest is required")
    tests, sources = manifest.get("tests"), manifest.get("source_sha256")
    if not isinstance(tests, list) or not tests or not isinstance(sources, dict) or not sources:
        raise ValueError("Contract manifest is empty or incomplete")
    expected = []
    for item in tests:
        if not isinstance(item, dict) or not isinstance(item.get("nodeid"), str):
            raise ValueError("Contract manifest contains an invalid node")
        identity = junit_identity(item["nodeid"])
        if identity != (item.get("classname"), item.get("name")):
            raise ValueError("Contract manifest changes a node's JUnit identity")
        expected.append(identity)
    if len(expected) != len(set(expected)):
        raise ValueError("Contract manifest contains duplicate test identities")
    declared_root = manifest.get("source_root")
    if not isinstance(declared_root, str):
        raise ValueError("Contract manifest has no source root")
    root = Path(declared_root).resolve() if source_root is None else source_root.resolve()
    if root != Path(declared_root).resolve():
        raise ValueError("Contract manifest belongs to a different checkout")
    required_sources = {item["nodeid"].partition("::")[0] for item in tests}
    if set(sources) != required_sources:
        raise ValueError("Contract manifest omits or invents test source hashes")
    for source, digest in sources.items():
        path = (root / source).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError(f"Contract test source changed after collection: {source}")
    document = ET.parse(junit).getroot()
    if _tag(document) not in {"testsuite", "testsuites"}:
        raise ValueError("JUnit report has no test suite")
    cases = [element for element in document.iter() if _tag(element) == "testcase"]
    if not cases:
        raise ValueError("JUnit contract executed no tests")
    for element in document.iter():
        if _tag(element) in {"skipped", "failure", "error"}:
            raise ValueError(f"JUnit contract contains a {_tag(element)} result")
        if _tag(element) not in {"testsuite", "testsuites"}:
            continue
        for counter in ("failures", "errors", "skipped", "disabled"):
            if counter in element.attrib and (not element.get(counter).isdigit() or int(element.get(counter)) != 0):
                raise ValueError(f"JUnit suite reports rejected {counter}")
        if "tests" in element.attrib:
            count = element.get("tests")
            actual = sum(_tag(child) == "testcase" for child in element.iter())
            if not count.isdigit() or int(count) != actual:
                raise ValueError("JUnit suite test count contradicts its executed nodes")
    executed = [(case.get("classname"), case.get("name")) for case in cases]
    if any(not all(identity) for identity in executed):
        raise ValueError("JUnit testcase has no exact identity")
    if len(executed) != len(set(executed)):
        raise ValueError("JUnit contract contains duplicate test execution")
    if Counter(executed) != Counter(expected):
        missing, extra = set(expected) - set(executed), set(executed) - set(expected)
        raise ValueError(f"JUnit selection differs from complete collection: missing={sorted(missing)}, extra={sorted(extra)}")
    return {"status": "passed", "executed_tests": len(cases), "skipped": 0,
            "complete_selection_verified": True, "source_files_verified": len(sources)}


class _CollectionManifest:
    def __init__(self):
        self.errors = []
        self.manifest = None

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.errors.append(f"{report.nodeid}: {'failed' if report.failed else 'skipped'}")

    def pytest_collection_finish(self, session):
        try:
            self.manifest = build_manifest([item.nodeid for item in session.items], session.config.rootpath, [])
        except ValueError as error:
            self.errors.append(str(error))


def collect_contract(selection: list[str], output: Path) -> int:
    import pytest

    plugin = _CollectionManifest()
    code = int(pytest.main(["--collect-only", "-q", *selection], plugins=[plugin]))
    if code or plugin.errors or plugin.manifest is None:
        payload = {"schema_version": 1, "status": "failed", "pytest_exit_code": code,
                   "errors": plugin.errors or ["No complete collection was produced"]}
        code = code or 1
    else:
        payload = plugin.manifest
        payload["selection"] = selection
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return code


def execute_contract(selection: list[str]) -> int:
    """Keep JUnit counts aligned with independently collected parent nodes.

    Pytest 9 reports successful unittest subtests as passing call reports, but
    its JUnit writer folds them into the parent testcase. Remove only those
    extra passing counters; failures and skips retain their original reports.
    """
    import pytest
    from _pytest.junitxml import xml_key

    class _ParentNodeCounts:
        def pytest_configure(self, config):
            self.config = config

        @pytest.hookimpl(trylast=True)
        def pytest_runtest_logreport(self, report):
            if (report.passed and report.when == "call"
                    and type(report).__name__ in {"SubtestReport", "SubTestReport"}
                    and hasattr(report, "context")):
                writer = self.config.stash.get(xml_key, None)
                if writer is not None:
                    writer.stats["passed"] -= 1

    return int(pytest.main(selection, plugins=[_ParentNodeCounts()]))


def fixed_selection(root: Path, tests: list[str], marker: str | None, include_public_defaults: bool) -> list[str]:
    paths = list(tests)
    if include_public_defaults:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with (root / "pyproject.toml").open("rb") as stream:
            defaults = tomllib.load(stream)["tool"]["pytest"]["ini_options"]["testpaths"]
        if not isinstance(defaults, list) or not defaults or not all(isinstance(path, str) for path in defaults):
            raise ValueError("Public pytest testpaths are empty or invalid")
        paths = [*defaults, *paths]
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("An explicit complete contract selection is required")
    return [*paths, *(["-m", marker] if marker else [])]


def run_contract(selection: list[str], execution_args: list[str], manifest: Path, junit: Path, root: Path) -> int:
    """Use fresh collection/execution processes and always validate their report."""
    manifest, junit = manifest.resolve(), junit.resolve()
    for path in (manifest, junit):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
    collected = subprocess.run([sys.executable, str(Path(__file__).resolve()), "collect", "--output", str(manifest),
                                "--", *selection], cwd=root)
    if collected.returncode:
        return collected.returncode
    executed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "execute", "--", *selection, *execution_args,
                               "--junitxml=" + str(junit)], cwd=root)
    try:
        result = validate_junit(json.loads(manifest.read_text()), junit, root)
    except (ValueError, OSError, TypeError, ET.ParseError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return executed.returncode or 1
    print(json.dumps(result, sort_keys=True))
    return executed.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--include-public-defaults", action="store_true")
    collect.add_argument("selection", nargs=argparse.REMAINDER)
    execute = commands.add_parser("execute")
    execute.add_argument("selection", nargs=argparse.REMAINDER)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--junit", type=Path, required=True)
    validate.add_argument("--source-root", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--junit", type=Path, required=True)
    run.add_argument("--source-root", type=Path, default=Path.cwd())
    run.add_argument("--include-public-defaults", action="store_true")
    run.add_argument("--test", action="append", default=[])
    run.add_argument("--marker")
    run.add_argument("execution_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            selection = args.selection[1:] if args.selection[:1] == ["--"] else args.selection
            if args.include_public_defaults:
                selection = fixed_selection(Path.cwd(), selection, None, True)
            return collect_contract(selection, args.output)
        if args.command == "execute":
            selection = args.selection[1:] if args.selection[:1] == ["--"] else args.selection
            return execute_contract(selection)
        if args.command == "run":
            root = args.source_root.resolve()
            selection = fixed_selection(root, args.test, args.marker, args.include_public_defaults)
            execution = args.execution_args[1:] if args.execution_args[:1] == ["--"] else args.execution_args
            return run_contract(selection, execution, args.manifest, args.junit, root)
        result = validate_junit(json.loads(args.manifest.read_text()), args.junit, args.source_root)
    except (ValueError, OSError, TypeError, ET.ParseError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
