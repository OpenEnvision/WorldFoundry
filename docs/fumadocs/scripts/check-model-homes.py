#!/usr/bin/env python3
"""Check current model articles without imposing an obsolete page template.

Generated articles must retain their command and related-model components.
Authored articles may choose their own section structure and media.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PAGES_DIR = Path(__file__).resolve().parents[1] / "content/docs/guides/supported-models"
PAGE_SOURCE = {"authored", "generated"}
RAW_IMAGE_RE = re.compile(r"<img\b", re.IGNORECASE)


def page_identity(path: Path) -> tuple[str, str]:
    if path.name.endswith(".zh.mdx"):
        return path.name[: -len(".zh.mdx")], "zh"
    return path.name[: -len(".mdx")], "en"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("unclosed YAML frontmatter")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter is not a mapping")
    return metadata, parts[2]


def has_component(body: str, name: str, model_id: str, locale: str) -> bool:
    for match in re.finditer(rf"<{name}\b([^>]*)/?>", body):
        attrs = match.group(1)
        if f'modelId="{model_id}"' in attrs and f'locale="{locale}"' in attrs:
            return True
    return False


def audit_page(path: Path) -> list[str]:
    model_id, locale = page_identity(path)
    text = path.read_text(encoding="utf-8")
    try:
        metadata, body = parse_frontmatter(text)
    except (ValueError, yaml.YAMLError) as exc:
        return [str(exc)]

    issues: list[str] = []
    for key in ("title", "description"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            issues.append(f"missing {key}")
    source = metadata.get("pageSource")
    if source not in PAGE_SOURCE:
        issues.append("pageSource must be authored or generated")
    if RAW_IMAGE_RE.search(body):
        issues.append("raw <img> bypasses the GitHub Pages base path; use DocsImage")
    if source == "generated":
        for component in ("ModelCommandBuilder", "ModelRelatedRecipes"):
            if not has_component(body, component, model_id, locale):
                issues.append(f"missing {component} for {model_id}/{locale}")
    return issues


def main() -> int:
    pages = sorted(
        path for path in PAGES_DIR.glob("*.mdx")
        if path.name not in {"index.mdx", "index.zh.mdx"}
    )
    by_model: dict[str, set[str]] = {}
    issues: list[tuple[str, str]] = []
    for path in pages:
        model_id, locale = page_identity(path)
        by_model.setdefault(model_id, set()).add(locale)
        issues.extend((path.name, issue) for issue in audit_page(path))
    for model_id, locales in sorted(by_model.items()):
        if locales != {"en", "zh"}:
            issues.append((model_id, f"missing locale page: {', '.join(sorted({'en', 'zh'} - locales))}"))

    print(f"model home pages: {len(pages)} ({len(by_model)} paired models)")
    if issues:
        for name, issue in issues[:30]:
            print(f"  {name}: {issue}")
        if len(issues) > 30:
            print(f"  ... {len(issues) - 30} more issues")
        return 1
    print("frontmatter, locale pairs, image paths, and generated components are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
