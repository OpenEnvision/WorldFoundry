#!/usr/bin/env python3
"""Check per-CUDA-tier torch constraint stubs against their Python SSOT.

This is a local drift check only. It does not resolve or download wheels.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldfoundry.runtime.cuda_tiers import (  # noqa: E402
    SUPPORTED_CUDA_TIERS,
    TIER_TORCH_SPECS,
)

PIN_RE = re.compile(r"^(torch|torchvision|torchaudio)\s*([<>=!~].+)$")


def check_tier(path: Path, *, expected: dict[str, str]) -> list[str]:
    """Return any constraint errors found in one tier stub."""

    if not path.is_file():
        return [f"missing constraint stub: {path}"]

    errors: list[str] = []
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        match = PIN_RE.fullmatch(text)
        if match is None:
            errors.append(f"{path}: unexpected line {text!r}")
            continue
        name = match.group(1)
        if name in found:
            errors.append(f"{path}: duplicate pin for {name}")
            continue
        found[name] = text

    for name, wanted in expected.items():
        actual = found.get(name)
        if actual is None:
            errors.append(f"{path}: missing pin for {name}")
        elif actual != wanted:
            errors.append(f"{path}: {name} pin {actual!r} != SSOT {wanted!r}")
    for name in found:
        if name not in expected:
            errors.append(f"{path}: unexpected package pin {name}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_REPO_ROOT, help="repository root")
    args = parser.parse_args(argv)

    errors: list[str] = []
    for tier in SUPPORTED_CUDA_TIERS:
        errors.extend(
            check_tier(
                args.root / "requirements" / "cuda" / f"{tier}-torch.txt",
                expected=TIER_TORCH_SPECS[tier],
            )
        )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "ok: CUDA torch constraint stubs match TIER_TORCH_SPECS for "
        + ", ".join(SUPPORTED_CUDA_TIERS)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
