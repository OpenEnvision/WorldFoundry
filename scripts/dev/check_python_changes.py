#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Run Ruff and ty on Python files changed by a pull request."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def changed_python_files(base: str, head: str, *, repo: Path) -> list[str]:
    """Select added and modified Python files from the merge-base diff."""
    names = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "-z", f"{base}...{head}"],
            cwd=repo,
        )
        .decode()
        .split("\0")
    )
    return [name for name in names if name.endswith(".py") and (repo / name).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Pull request base commit")
    parser.add_argument("--head", default="HEAD", help="Pull request head commit")
    args = parser.parse_args()
    repo = Path.cwd()
    files = changed_python_files(args.base, args.head, repo=repo)
    if not files:
        print("No Python changes")
        return 0
    for command in (
        [sys.executable, "-m", "ruff", "check", "--force-exclude", *files],
        [sys.executable, "-m", "ruff", "format", "--check", "--force-exclude", *files],
        [sys.executable, "-m", "ty", "check", "--python", sys.executable, "--force-exclude", *files],
    ):
        result = subprocess.run(command, cwd=repo, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
