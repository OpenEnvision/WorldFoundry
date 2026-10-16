#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Check DCO sign-offs on the commits introduced by a pull request."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def unsigned_commits(base: str, head: str, *, repo: Path) -> list[str]:
    """Return commits without a name and email in a Signed-off-by trailer."""
    commits = subprocess.check_output(
        ["git", "log", "--format=%H", f"{base}..{head}"], cwd=repo, text=True
    ).splitlines()
    unsigned = []
    for commit in commits:
        message = subprocess.check_output(["git", "show", "-s", "--format=%B", commit], cwd=repo, text=True)
        trailers = message.strip().rsplit("\n\n", 1)[-1].splitlines()
        if not any(
            re.fullmatch(r"Signed-off-by:\s+.+ <[^<>\s@]+@[^<>\s@]+>", value.strip(), re.IGNORECASE)
            for value in trailers
        ):
            unsigned.append(commit)
    return unsigned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Pull request base commit")
    parser.add_argument("--head", default="HEAD", help="Pull request head commit")
    args = parser.parse_args()
    unsigned = unsigned_commits(args.base, args.head, repo=Path.cwd())
    if unsigned:
        print("Missing DCO sign-off; sign each contribution with git commit --signoff:")
        for commit in unsigned:
            print(commit)
        return 1
    print("DCO sign-offs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
