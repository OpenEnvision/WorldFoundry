# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired documentation contracts for inference quality and performance evidence."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_optimization_docs_cover_quality_and_performance_validation() -> None:
    docs_root = REPO_ROOT / "docs/fumadocs/content/docs/guides"
    for name in ("inference-optimization.mdx", "inference-optimization.zh.mdx"):
        text = (docs_root / name).read_text(encoding="utf-8")
        for snippet in (
            "benchmarks.inference.wan22_multigpu_e2e",
            "--reference-latents",
            "--reference-video",
            "--fail-on-fallback",
            "benchmarks.operators.attention_adapter_parity",
            "--strict",
            "benchmarks.inference.framework_ab_wan22",
            "Exclusive_Process",
        ):
            assert snippet in text, f"{name} is missing {snippet}"
