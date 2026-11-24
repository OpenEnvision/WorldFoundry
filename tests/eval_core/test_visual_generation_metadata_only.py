from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VISUAL_GENERATION_ROOT = REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation"

ALLOWED_RUNTIME_PROFILE_SYNTHESIS_FILES = frozenset(
    {
        "animatediff/animatediff_synthesis.py",
        "zeroscope/zeroscope_synthesis.py",
    }
)


def test_visual_generation_runtime_profile_synthesis_is_limited_to_in_tree_runtimes() -> None:
    offenders: list[str] = []
    for path in sorted(VISUAL_GENERATION_ROOT.rglob("*_synthesis.py")):
        relative_path = path.relative_to(VISUAL_GENERATION_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(_is_runtime_profile_synthesis_base(base) for base in node.bases):
                if relative_path not in ALLOWED_RUNTIME_PROFILE_SYNTHESIS_FILES:
                    offenders.append(f"{relative_path}:{node.name}")

    assert offenders == [], "unexpected RuntimeProfileSynthesis wrappers: " + ", ".join(offenders)


def _is_runtime_profile_synthesis_base(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "RuntimeProfileSynthesis"
    if isinstance(node, ast.Attribute):
        return node.attr == "RuntimeProfileSynthesis"
    return False
