"""Select short inference replays without importing models or requiring weights/GPU."""

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
    for rule in [*policy.get("cpu_only_components", []), *policy.get("relocation_contracts", [])]:
        if not isinstance(rule, dict) or not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
            raise ValueError("CPU-only and relocation declarations need an explicit reason")
        contracts = rule.get("required_cpu_contracts")
        if not isinstance(contracts, list) or not contracts:
            raise ValueError("CPU-only and relocation declarations need concrete CPU contracts")
        for contract in contracts:
            relative_path(contract)
            if not contract.startswith("tests/") or not contract.endswith(".py"):
                raise ValueError("CPU contracts must name exact Python test files")
    for rule in policy.get("cpu_only_components", []):
        paths = rule.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("CPU-only components need exact paths")
        for path in paths:
            relative_path(path)
            if any(character in path for character in "*?["):
                raise ValueError("CPU-only declarations cannot use wildcard paths")
    for rule in policy.get("relocation_contracts", []):
        for key in ("from_root", "to_root"):
            relative_path(rule[key])
        if rule["from_root"] == rule["to_root"]:
            raise ValueError("Relocation roots must differ")
        for package in rule.get("packages", []):
            if not isinstance(package, str) or not re.fullmatch(r"[a-zA-Z_]\w*", package):
                raise ValueError("Relocations must list exact package names")
        for path in [*rule.get("files", []), *rule.get("rewrite_only_paths", [])]:
            relative_path(path)
            if any(character in path for character in "*?["):
                raise ValueError("Relocation file declarations cannot use wildcards")
        for rewrite in rule.get("relative_import_rewrites", []):
            if (not isinstance(rewrite, list) or len(rewrite) != 2
                    or not all(isinstance(value, str) and value.startswith("from .") for value in rewrite)):
                raise ValueError("Extra relocation rewrites must name exact relative imports")
        for removed in rule.get("deleted_metadata", []):
            relative_path(removed["path"])
            if (not re.fullmatch(r"[0-9a-f]{64}", removed.get("sha256", ""))
                    or not isinstance(removed.get("reason"), str) or not removed["reason"].strip()):
                raise ValueError("Removed metadata needs its exact content digest and reason")
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


def relocation_proofs(
    source_root: Path, policy: dict, changes: list[str], *, base: str | None, head: str,
    history_root: Path | None = None,
) -> list[dict]:
    """Prove declared moves from immutable blobs, never from rename similarity.

    Both paths, modes and contents must agree with the inspected source. An
    implementation edit cannot pass merely because Git reports a rename.
    """
    if base is None or not policy.get("relocation_contracts"):
        return []
    history = history_root or source_root
    commits = [revision(history, ref + "^{commit}") for ref in (base, head)]
    if not all(commits):
        raise ValueError("Relocation proof requires pinned Git base and head commits")

    def tree(commit):
        raw = subprocess.check_output(["git", "-C", str(history), "ls-tree", "-r", "-z", commit])
        entries = {}
        for entry in raw.split(b"\0"):
            if entry:
                metadata, path = entry.split(b"\t", 1)
                mode, kind, blob = metadata.decode().split()
                if kind == "blob":
                    entries[path.decode()] = (mode, blob)
        return entries

    before_tree, after_tree = [tree(commit) for commit in commits]
    blob_cache = {}

    def content(entry):
        blob = entry[1]
        if blob not in blob_cache:
            blob_cache[blob] = subprocess.check_output(["git", "-C", str(history), "cat-file", "blob", blob])
        return blob_cache[blob]

    def inspect(path, data):
        target = source_root / path
        return (target.is_file() and not target.is_symlink()
                and target.resolve().is_relative_to(source_root.resolve()) and target.read_bytes() == data)

    proofs, changed = [], set(changes)
    for rule in policy["relocation_contracts"]:
        old_root, new_root = rule["from_root"] + "/", rule["to_root"] + "/"
        old_namespace = rule["from_root"].replace("/", ".") + "."
        new_namespace = rule["to_root"].replace("/", ".") + "."

        def comparison(before, after):
            if before == after:
                return "identical_bytes"
            try:
                rewritten = before.decode().replace(old_namespace, new_namespace)
            except UnicodeError:
                return None
            for old, new in rule.get("relative_import_rewrites", []):
                rewritten = rewritten.replace(old, new)
            return "declared_namespace_rewrite" if rewritten.encode() == after else None

        def record(old, new, before, after, kind, reason=None):
            proofs.append({
                "old_path": old, "new_path": new, "kind": kind,
                "old_sha256": hashlib.sha256(before).hexdigest(),
                "new_sha256": hashlib.sha256(after).hexdigest() if after is not None else None,
                "base_revision": commits[0], "head_revision": commits[1],
                "reason": reason or rule["reason"],
                "required_cpu_contracts": rule["required_cpu_contracts"],
            })

        for old in sorted(changed):
            if not old.startswith(old_root):
                continue
            relative = old.removeprefix(old_root)
            if (relative.split("/", 1)[0] not in rule.get("packages", [])
                    and relative not in rule.get("files", [])):
                continue
            new = new_root + relative
            if (new not in changed or old not in before_tree or old in after_tree
                    or new in before_tree or new not in after_tree or (source_root / old).exists()
                    or before_tree[old][0] != after_tree[new][0]
                    or before_tree[old][0] not in {"100644", "100755"}):
                continue
            before, after = content(before_tree[old]), content(after_tree[new])
            kind = comparison(before, after)
            if kind and inspect(new, after):
                record(old, new, before, after, kind)
        for path in rule.get("rewrite_only_paths", []):
            if path not in changed or path not in before_tree or path not in after_tree:
                continue
            before, after = content(before_tree[path]), content(after_tree[path])
            if (before_tree[path][0] == after_tree[path][0]
                    and before_tree[path][0] in {"100644", "100755"}
                    and comparison(before, after) == "declared_namespace_rewrite" and inspect(path, after)):
                record(path, path, before, after, "declared_namespace_rewrite")
        for removed in rule.get("deleted_metadata", []):
            path = removed["path"]
            if (path in changed and path in before_tree and path not in after_tree
                    and not (source_root / path).exists() and before_tree[path][0] == "100644"):
                before = content(before_tree[path])
                if hashlib.sha256(before).hexdigest() == removed["sha256"]:
                    record(path, None, before, None, "removed_metadata", removed["reason"])
    return proofs


def select_cases(
    matrix: Path, dependencies: Path, source_root: Path, changes: list[str], *, evidence: Path | None = None,
    base: str | None = None, head: str = "HEAD", history_root: Path | None = None,
) -> dict:
    cases, policy = load_definitions(matrix, dependencies)
    changes = sorted({relative_path(path) for path in changes})
    for rule in [*policy.get("cpu_only_components", []), *policy.get("relocation_contracts", [])]:
        for contract in rule["required_cpu_contracts"]:
            test = source_root / contract
            if (not test.is_file() or test.is_symlink()
                    or not test.resolve().is_relative_to(source_root.resolve())):
                raise ValueError(f"Declared CPU contract has no inspected implementation: {contract}")
    graph = ImportGraph(source_root)
    proofs = relocation_proofs(source_root, policy, changes, base=base, head=head, history_root=history_root)
    proven_paths = {path for proof in proofs for path in (proof["old_path"], proof["new_path"]) if path}
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
    ignored, uncovered, cpu_only = [], [], []
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
            declared_cpu = [rule for rule in policy.get("cpu_only_components", []) if path in rule["paths"]]
            if path in proven_paths:
                continue
            if declared_cpu:
                cpu_only.extend({"path": path, "reason": rule["reason"],
                                 "required_cpu_contracts": rule["required_cpu_contracts"]} for rule in declared_cpu)
            elif matches(path, policy["ignored_paths"]):
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
        "proven_relocations": proofs,
        "cpu_only_changes": cpu_only,
        "required_cpu_contracts": sorted({
            contract for item in [*proofs, *cpu_only] for contract in item["required_cpu_contracts"]
        }),
        "uncovered_paths": uncovered,
        "graph_errors": graph_errors,
        "required_checks": ["public-cpu", "inference-tensors", *(["real-weight-replays"] if selected else [])],
        "coverage_scope": policy.get(
            "coverage_scope", "Declared short 3D cases; a plan is not successful inference evidence."
        ),
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
            base=args.base,
            head=args.head,
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
