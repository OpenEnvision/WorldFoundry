"""Synchronize model MDX pages with the canonical model-home renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generate_model_home_prose import page_filename, page_source, render_page

PAGES_DIR = Path(__file__).resolve().parents[1] / "content/docs/guides/supported-models"
RESERVED_NAMES = {"index.mdx", "index.zh.mdx", "meta.json", "meta.zh.json"}
RESERVED_STEMS = {"index", "meta"}


def is_generated(path: Path) -> bool:
    return page_source(path) == "generated"


def render_model_page(recipe: dict[str, Any], locale: str) -> str:
    return render_page(recipe, locale, page_source_override="generated")


def iter_page_paths(recipes: list[dict[str, Any]]) -> list[tuple[Path, str, dict[str, Any], str]]:
    rows: list[tuple[Path, str, dict[str, Any], str]] = []
    for recipe in recipes:
        model_id = str(recipe.get("id") or "").strip()
        if not model_id or model_id in RESERVED_STEMS:
            continue
        for locale in ("en", "zh"):
            path = PAGES_DIR / page_filename(model_id, locale)
            rows.append((path, model_id, recipe, locale))
    return rows


def sync_model_pages(recipes: list[dict[str, Any]], *, check: bool = False) -> int:
    """Write or verify generated model MDX pages. Returns 1 when --check finds drift."""
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    expected_generated: dict[Path, str] = {}
    authored_paths: set[Path] = set()
    stale: list[str] = []
    written = 0

    for path, _model_id, recipe, locale in iter_page_paths(recipes):
        if path.exists() and not is_generated(path):
            authored_paths.add(path)
            continue
        expected_generated[path] = render_model_page(recipe, locale)

    for path, content in expected_generated.items():
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path))
            continue
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
            written += 1

    known = {path for path, *_ in iter_page_paths(recipes)}
    for path in sorted(PAGES_DIR.glob("*.mdx")):
        if path.name in RESERVED_NAMES or path in known or path in authored_paths:
            continue
        if is_generated(path):
            if check:
                stale.append(f"extra generated page: {path}")
            else:
                path.unlink()
                written += 1

    if check:
        if stale:
            for item in stale:
                print(f"stale generated model page: {item}")
            return 1
        print(
            f"model pages are current: {len(expected_generated)} generated, "
            f"{len(authored_paths)} authored under {PAGES_DIR}"
        )
        return 0

    print(
        f"wrote model pages under {PAGES_DIR} generated={len(expected_generated)} "
        f"updated={written} authored={len(authored_paths)}"
    )
    return 0
