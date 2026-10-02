from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = PROJECT_ROOT / "src" / "worldfoundry" / "skills" / "worldfoundry-evaluation-guide"


def test_worldfoundry_evaluation_skill_references_exist() -> None:
    skill = SKILL_ROOT / "SKILL.md"
    text = skill.read_text(encoding="utf-8")

    links = re.findall(r"\]\((references/[^)]+)\)", text)

    assert links
    for link in links:
        assert (SKILL_ROOT / link).is_file(), link


def test_worldfoundry_evaluation_skill_pytest_paths_exist() -> None:
    markdown_files = (SKILL_ROOT / "SKILL.md", *sorted((SKILL_ROOT / "references").glob("*.md")))
    missing: list[str] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for match in re.findall(r"PYTHONPATH=. pytest -q (tests/eval_core/test_[A-Za-z0-9_]+\.py)", text):
            if not (PROJECT_ROOT / match).is_file():
                missing.append(f"{path.relative_to(PROJECT_ROOT)}: {match}")

    assert missing == []
