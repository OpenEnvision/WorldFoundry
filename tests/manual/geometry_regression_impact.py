"""Select short 3D replays without importing models or requiring weights/GPU."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid repository path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or path == PurePosixPath("."):
        raise ValueError(f"Invalid repository path: {value!r}")
    return path.as_posix()


def load_definitions(matrix: Path, dependencies: Path) -> tuple[dict, dict]:
    cases = json.loads(matrix.read_text())
    policy = json.loads(dependencies.read_text())
    if not isinstance(cases, dict) or not cases:
        raise ValueError("The regression matrix is empty or invalid")
    for name, case in cases.items():
        if (
            not isinstance(case, dict)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name)
            or case.get("id") != name
        ):
            raise ValueError(f"Invalid case id: {name!r}")
        if not re.fullmatch(r"worldfoundry(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*", case.get("target", "")):
            raise ValueError(f"Invalid case target: {name}")
    if policy.get("schema_version") != 1 or set(policy.get("cases", {})) != set(cases):
        raise ValueError("Every matrix case needs an explicit dependency declaration")
    components = policy.get("components", {})
    if not components:
        raise ValueError("No inference components declared")
    for paths in [*components.values(), policy.get("shared_paths", []), policy.get("ignored_paths", [])]:
        if not isinstance(paths, list) or not paths:
            raise ValueError("Dependency path lists must be nonempty")
        for path in paths:
            relative_path(path)
    for name, used in policy["cases"].items():
        if not isinstance(used, list) or not used or any(component not in components for component in used):
            raise ValueError(f"Unknown or missing component for {name}")
    return cases, policy


def changed_paths(root: Path, base: str, head: str) -> list[str]:
    """Include both sides of renames and deleted paths; filenames remain unquoted."""
    refs = [revision(root, ref + "^{commit}") for ref in (base, head)]
    if not all(refs):
        raise ValueError("Diff base and head must resolve to commits")
    raw = (
        subprocess.check_output(["git", "-C", str(root), "diff", "--name-status", "-z", "--find-renames", *refs, "--"])
        .decode()
        .split("\0")
    )
    result = set()
    index = 0
    while index < len(raw) - 1:
        status = raw[index]
        count = 2 if status.startswith(("R", "C")) else 1
        for path in raw[index + 1 : index + count + 1]:
            result.add(relative_path(path))
        index += count + 1
    return sorted(result)


def revision(root: Path, ref: str = "HEAD") -> str | None:
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        return None
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", ref], capture_output=True, text=True
    )
    return process.stdout.strip() if process.returncode == 0 else None


class ImportGraph:
    """Read conditional/relative imports and literal lazy module targets as AST."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        process = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "worldfoundry"], capture_output=True)
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        is_checkout = top.returncode == 0 and Path(top.stdout.strip()).resolve() == self.root
        paths = (
            process.stdout.decode().split("\0")
            if process.returncode == 0 and is_checkout
            else [p.relative_to(root).as_posix() for p in (root / "worldfoundry").rglob("*.py")]
        )
        self.modules = {}
        for path in paths:
            if path.endswith(".py"):
                module = path[:-3].replace("/", ".")
                self.modules[module.removesuffix(".__init__")] = path
        self.cache = {}

    def imports(self, name: str) -> set[str]:
        if name in self.cache:
            return self.cache[name]
        path = self.root / self.modules[name]
        if not path.resolve().is_relative_to(self.root):
            raise ValueError(f"Import escapes source tree: {name}")
        tree = ast.parse(path.read_text(), filename=str(path))
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        imports = set()
        # Importing a module also executes any non-namespace parent __init__.
        parts = name.split(".")
        imports.update(".".join(parts[:index]) for index in range(1, len(parts)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
                    base = ".".join([*prefix, *([node.module] if node.module else [])])
                else:
                    base = node.module or ""
                imports.add(base)
                imports.update(base + "." + alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Includes import_module literals and lazy facade tables.
                module = node.value.partition(":")[0]
                if re.fullmatch(r"worldfoundry(?:\.[a-zA-Z_]\w*)+", module):
                    imports.add(module)
        result = {item for item in imports if item in self.modules and item != name}
        self.cache[name] = result
        return result

    def closure(self, target: str) -> set[str]:
        target = target.partition(":")[0]
        if target not in self.modules:
            raise ValueError(f"Case target has no tracked implementation: {target}")
        seen, pending = set(), [target]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            pending.extend(self.imports(name) - seen)
        return {self.modules[name] for name in seen}


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def select_cases(
    matrix: Path, dependencies: Path, source_root: Path, changes: list[str], *, evidence: Path | None = None
) -> dict:
    cases, policy = load_definitions(matrix, dependencies)
    changes = sorted({relative_path(path) for path in changes})
    graph = ImportGraph(source_root)
    reasons = {name: [] for name in cases}
    graph_errors = {}
    closures, traces = {}, {}
    for name, case in cases.items():
        try:
            closures[name] = graph.closure(case["target"])
        except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
            closures[name] = set()
            graph_errors[name] = str(exc)
        traces[name] = set()
        if evidence is not None and (evidence / name / "manifest.json").is_file():
            manifest = json.loads((evidence / name / "manifest.json").read_text())
            if manifest.get("status") == "passed" and manifest.get("case", {}).get("id") == name:
                traces[name] = {relative_path(path) for path in manifest.get("source_hashes", {})}
    ignored, uncovered = [], []
    for path in changes:
        matched = False
        if matches(path, policy["shared_paths"]) or path == matrix.relative_to(source_root).as_posix():
            for name in cases:
                reasons[name].append({"path": path, "kind": "shared_component_or_case_matrix"})
            matched = True
        else:
            for name in cases:
                used = [
                    component for component in policy["cases"][name] if matches(path, policy["components"][component])
                ]
                kinds = ["declared_component"] if used else []
                if path in closures[name]:
                    kinds.append("static_import")
                if path in traces[name]:
                    kinds.append("recorded_import")
                if kinds:
                    reasons[name].append({"path": path, "kind": "+".join(kinds), "components": used})
                    matched = True
        if not matched:
            if matches(path, policy["ignored_paths"]):
                ignored.append(path)
            else:
                uncovered.append(path)
    # Unknown inference paths/failed graph analysis expand selection, but never
    # claim the existing cases cover an unintegrated model or unresolved branch.
    if uncovered or graph_errors:
        for name in cases:
            reasons[name].append(
                {"kind": "conservative_fallback", "paths": uncovered, "graph_errors": sorted(graph_errors)}
            )
    selected = [name for name in cases if reasons[name]]
    return {
        "schema_version": 1,
        "status": "uncovered" if uncovered or graph_errors else ("planned" if selected else "no_inference_changes"),
        "source_revision": revision(source_root),
        "source_tree": revision(source_root, "HEAD^{tree}"),
        "matrix_sha256": digest(matrix),
        "dependencies_sha256": digest(dependencies),
        "changed_paths": changes,
        "selected_cases": selected,
        "reasons": {name: reasons[name] for name in selected},
        "ignored_paths": ignored,
        "uncovered_paths": uncovered,
        "graph_errors": graph_errors,
        "required_checks": ["public-cpu", "inference-tensors", *(["real-weight-replays"] if selected else [])],
        "coverage_scope": "Declared short 3D cases; a plan is not successful inference evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("tests/manual/geometry_regression_cases.json"))
    parser.add_argument("--dependencies", type=Path, default=Path("tests/manual/geometry_regression_dependencies.json"))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.base is None and not args.changed_file:
            raise ValueError("Supply --base or at least one --changed-file; an empty diff must be explicit")
        if args.base is not None and args.changed_file:
            raise ValueError("Use either a Git diff or explicit changed files")
        if args.base and revision(args.source_root, args.head + "^{commit}") != revision(
            args.source_root, "HEAD^{commit}"
        ):
            raise ValueError("--head must match the inspected checkout; check out the requested head first")
        changes = changed_paths(args.source_root, args.base, args.head) if args.base else args.changed_file
        report = select_cases(
            args.matrix.resolve(),
            args.dependencies.resolve(),
            args.source_root.resolve(),
            changes,
            evidence=args.evidence,
        )
        if args.base:
            report.update(
                base_revision=revision(args.source_root, args.base), head_revision=revision(args.source_root, args.head)
            )
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError, TypeError) as exc:
        report = {"status": "failed", "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "selected_cases", "uncovered_paths", "error") if key in report}
        )
    )
    return 0 if report["status"] in {"planned", "no_inference_changes"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
