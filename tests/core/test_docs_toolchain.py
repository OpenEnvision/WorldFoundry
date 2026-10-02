"""Focused contracts for the portable Fumadocs toolchain."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FUMADOCS = REPO_ROOT / "docs" / "fumadocs"
LAUNCHER = FUMADOCS / "scripts" / "run-python.sh"


def test_python_backed_npm_scripts_use_portable_launcher() -> None:
    scripts = json.loads((FUMADOCS / "package.json").read_text(encoding="utf-8"))["scripts"]
    for name in (
        "api:check",
        "api:generate",
        "cli:screenshots",
        "coverage:check",
        "coverage:generate",
        "models:check",
        "models:generate",
    ):
        assert scripts[name].startswith("bash scripts/run-python.sh ")


def test_launcher_prefers_wf_docs_python(tmp_path: Path) -> None:
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python is not None
    marker = tmp_path / "marker.py"
    marker.write_text("print('portable-docs-python')\n", encoding="utf-8")
    shim = tmp_path / "docs-python"
    shim.write_text(f"#!/bin/sh\nexec '{real_python}' \"$@\"\n", encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(marker)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={**os.environ, "WF_DOCS_PYTHON": str(shim), "PYTHON": "must-not-run"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "portable-docs-python"


def test_launcher_uses_python_override_when_wf_override_is_absent(tmp_path: Path) -> None:
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python is not None
    marker = tmp_path / "marker.py"
    marker.write_text("print('python-env-fallback')\n", encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("WF_DOCS_PYTHON", None)
    environment["PYTHON"] = real_python

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(marker)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "python-env-fallback"


def test_build_checks_all_generated_docs_data_before_next_build() -> None:
    source = (REPO_ROOT / "scripts" / "docs" / "build.sh").read_text(encoding="utf-8")
    types_check = source.index("npm run types:check")
    build = source.index("npm run build")
    for command in ("api:check", "models:check", "coverage:check"):
        assert types_check < source.index(f"npm run {command}") < build


def test_makefile_docs_and_bounded_shell_gates() -> None:
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    cli_block = source[source.index("cli-entrypoint-check:") : source.index("\nlint:")]
    assert "--help" in cli_block
    assert "zoo models --json" in cli_block
    assert "zoo benchmarks --json" in cli_block

    shell_block = source[source.index("shell-check:") : source.index("\ndata-check:")]
    for root in ("scripts/setup/*.sh", "scripts/dev/*.sh", "docs/fumadocs/scripts/*.sh"):
        assert root in shell_block
    assert "git ls-files" in shell_block
    assert "bash -n" in shell_block
    assert "find " not in shell_block


def test_catalog_coverage_check_remains_exposed() -> None:
    scripts = json.loads((FUMADOCS / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert "--check" in scripts["coverage:check"]
    assert "generate-catalog-coverage.py" in scripts["coverage:generate"]


def test_docs_node_runtime_and_lockfile_registry_are_portable() -> None:
    package = json.loads((FUMADOCS / "package.json").read_text(encoding="utf-8"))
    lock_text = (FUMADOCS / "package-lock.json").read_text(encoding="utf-8")
    lock = json.loads(lock_text)

    assert package["engines"]["node"] == ">=20.19.0"
    assert lock["packages"][""]["engines"] == package["engines"]
    assert (FUMADOCS / ".nvmrc").read_text(encoding="utf-8").strip() == "22"
    assert "registry.npmmirror.com" not in lock_text


def test_local_ssd_dev_sync_refreshes_source_without_copying_dependency_trees() -> None:
    source = (FUMADOCS / "scripts" / "dev-local-ssd.sh").read_text(encoding="utf-8")

    assert "--exclude 'node_modules'" in source
    assert "--exclude 'tmp/'" in source
    assert "npm ci --prefer-offline --no-audit --no-fund" in source
    assert "run_predev()" in source
    assert "node scripts/predev.mjs" in source
    run_dev = source[source.index("run_dev() {") : source.index("\nusage() {")]
    assert "rsync -a" in run_dev
    assert "sync_to_local full" in run_dev
    assert run_dev.index("run_predev") < run_dev.index("exec npx")
    assert "CPFS" not in source
