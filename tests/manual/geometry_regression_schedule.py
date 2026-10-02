"""Explicitly import accepted evidence and manage this host's daily GPU cron job.

No model weights, credentials, profiles, or accepted outputs enter Git. Installing
a schedule never accepts a new baseline and never pulls or modifies source refs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "geometry_schedule_suite", Path(__file__).with_name("geometry_regression_suite.py")
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

BEGIN = "# BEGIN WorldFoundry geometry-regression"
END = "# END WorldFoundry geometry-regression"
CONTROLLERS = (
    "geometry_regression.py",
    "geometry_regression_impact.py",
    "geometry_regression_suite.py",
    "geometry_regression_schedule.py",
)


def import_reference(source: Path, destination: Path, matrix: Path) -> dict:
    """Copy an accepted baseline once, rebasing only exported artifact paths."""
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Baseline destination must be new and separate from the original")
    cases = json.loads(matrix.read_text())
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Cannot import an empty matrix")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".accepted-", dir=destination.parent))
    index = {"schema_version": 1, "original_root": str(source), "cases": {}}
    try:
        for name in cases:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
                raise ValueError("Invalid baseline case id")
            original = source / name
            if original.is_symlink() or any(p.is_symlink() for p in original.rglob("*")):
                raise ValueError(f"Baseline contains symlinks: {name}")
            target = staging / name
            shutil.copytree(original, target)
            manifest_file = target / "manifest.json"
            manifest = json.loads(manifest_file.read_text())
            if manifest.get("case", {}).get("id") != name:
                raise ValueError(f"Baseline case id does not match matrix: {name}")
            exports = []
            for raw in manifest.get("exported_files", []):
                path = Path(raw).relative_to(original)
                suite.impact.relative_path(path.as_posix())
                exports.append(str(destination / name / path))
            manifest["exported_files"] = exports
            suite.write_json(manifest_file, manifest)
            index["cases"][name] = {
                "original_manifest_sha256": suite.replay.sha256(original / "manifest.json"),
                "files": {
                    p.relative_to(target).as_posix(): suite.replay.sha256(p)
                    for p in sorted(target.rglob("*"))
                    if p.is_file()
                },
            }
        suite.write_json(staging / "accepted-index.json", index)
        # Atomic promotion; comparisons need the final absolute export paths.
        if destination.exists():
            raise ValueError("Baseline destination appeared during import")
        staging.rename(destination)
        try:
            for name in cases:
                result = suite.replay.compare_runs(source / name, destination / name)
                if result["status"] != "passed":
                    raise ValueError(f"Copied baseline differs: {name}")
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "reference": str(destination),
        "reference_index_sha256": suite.replay.sha256(destination / "accepted-index.json"),
        "cases": list(cases),
    }


def remove_block(existing: str) -> str:
    lines = existing.splitlines(keepends=True)
    out, inside = [], False
    for line in lines:
        if line.rstrip("\r\n") == BEGIN:
            if inside:
                raise ValueError("Nested WorldFoundry cron block")
            inside = True
        elif line.rstrip("\r\n") == END:
            if not inside:
                raise ValueError("Unmatched WorldFoundry cron block end")
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError("Unterminated WorldFoundry cron block")
    return "".join(out)


def cron_text(
    existing: str, python: str, controller: Path, profile: Path, log: Path, *, hour: int = 3, minute: int = 0
) -> str:
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Invalid daily cron time")
    original = remove_block(existing)
    if original and not original.endswith("\n"):
        original += "\n"
    previous = {"SHELL": "/bin/sh", "CRON_TZ": '""'}
    for line in original.splitlines():
        match = re.fullmatch(r"\s*(SHELL|CRON_TZ)\s*=\s*(.*?)\s*", line)
        if match:
            previous[match.group(1)] = match.group(2)
    args = [python, str(controller), "--profile", str(profile)]
    if any("\n" in value or "\r" in value for value in [*args, str(log)]):
        raise ValueError("Cron paths cannot contain newlines")
    command = (shlex.join(args) + " >> " + shlex.quote(str(log)) + " 2>&1").replace("%", "\\%")
    return (
        original
        + BEGIN
        + "\nSHELL=/bin/sh\nCRON_TZ=Asia/Shanghai\n"
        + f"{minute} {hour} * * * {command}\n"
        + f"SHELL={previous['SHELL']}\nCRON_TZ={previous['CRON_TZ']}\n"
        + END
        + "\n"
    )


def read_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        if result.returncode == 1 and "no crontab for" in result.stderr.lower():
            return ""
        raise RuntimeError("Cannot read crontab: " + result.stderr.strip())
    return result.stdout


def write_crontab(value: str) -> None:
    subprocess.run(["crontab", "-"], input=value, capture_output=True, text=True, timeout=10, check=True)


def install(profile_path: Path, *, hour: int = 3, minute: int = 0) -> dict:
    profile_path = profile_path.resolve()
    profile = suite.read_profile(profile_path)
    state = Path(profile["state_root"])
    index = suite.reference_index(profile)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="schedule-preflight-", dir=state) as temporary:
        source = Path(temporary) / "source"
        revision = suite.snapshot_source(profile, source)["source_revision"]
        cases, _ = suite.impact.load_definitions(source / profile["matrix"], source / profile["dependencies"])
        for name, case in cases.items():
            root = Path(profile["reference"]) / name
            suite.verify_reference(root, name, index)
            suite.materialize_case(case, root, {**os.environ, **profile.get("env", {})})
    controller = state / "controllers" / revision
    controller.mkdir(parents=True, exist_ok=True, mode=0o700)
    hashes = {}
    for name in CONTROLLERS:
        blob = subprocess.check_output(
            ["git", "-C", profile["source_root"], "show", revision + ":tests/manual/" + name], timeout=20
        )
        path = controller / name
        path.write_bytes(blob)
        hashes[name] = suite.replay.sha256(path)
    before = read_crontab()
    after = cron_text(
        before,
        profile["python"],
        controller / "geometry_regression_suite.py",
        profile_path,
        state / "cron.log",
        hour=hour,
        minute=minute,
    )
    suite.write_json(state / "crontab-before-install.json", {"crontab": before})
    write_crontab(after)
    if read_crontab() != after:
        raise RuntimeError("Installed crontab does not match expected entry")
    result = {
        "status": "installed",
        "timezone": "Asia/Shanghai",
        "hour": hour,
        "minute": minute,
        "controller_revision": revision,
        "controller_sha256": hashes,
        "profile": str(profile_path),
        "source_ref": profile["source_ref"],
        "cron": after,
    }
    suite.write_json(state / "schedule.json", result)
    return result


def uninstall(profile_path: Path) -> dict:
    profile = suite.read_profile(profile_path)
    before = read_crontab()
    after = remove_block(before)
    write_crontab(after)
    if read_crontab() != after:
        raise RuntimeError("Removed crontab does not match expected entries")
    result = {"status": "removed"}
    suite.write_json(Path(profile["state_root"]) / "schedule.json", result)
    return result


def status(profile_path: Path) -> dict:
    profile = suite.read_profile(profile_path)
    state = Path(profile["state_root"])
    cron = read_crontab()
    result = {"installed": BEGIN in cron.splitlines(), "state_root": str(state), "cron": cron}
    for name in ("schedule", "latest"):
        if (state / (name + ".json")).is_file():
            result[name] = json.loads((state / (name + ".json")).read_text())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    copy = sub.add_parser("import-reference")
    copy.add_argument("--source", type=Path, required=True)
    copy.add_argument("--destination", type=Path, required=True)
    copy.add_argument("--matrix", type=Path, default=Path("tests/manual/geometry_regression_cases.json"))
    for name in ("install", "remove", "status"):
        command = sub.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        if name == "install":
            command.add_argument("--hour", type=int, default=3)
            command.add_argument("--minute", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "import-reference":
            result = import_reference(args.source, args.destination, args.matrix)
        elif args.command == "install":
            result = install(args.profile, hour=args.hour, minute=args.minute)
        elif args.command == "remove":
            result = uninstall(args.profile)
        else:
            result = status(args.profile)
    except (Exception, KeyboardInterrupt) as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
