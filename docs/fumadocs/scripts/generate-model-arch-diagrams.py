#!/usr/bin/env python3
"""Build ``lib/model-arch-diagrams.json`` from public assets and recipe metadata."""

from __future__ import annotations

import json
from pathlib import Path

from model_paper_figures import (
    caption_pair,
    overview_src,
    teaser_src,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES_PATH = ROOT / "lib" / "model-recipes-data.json"
OUT_PATH = ROOT / "lib" / "model-arch-diagrams.json"


def main() -> int:
    recipes = json.loads(RECIPES_PATH.read_text(encoding="utf-8"))
    registry: dict[str, dict[str, str]] = {}

    for recipe in recipes.get("recipes", []):
        model_id = str(recipe.get("id") or "").strip()
        if not model_id:
            continue
        docs = recipe.get("docs") or {}
        paper = docs.get("paper") or {}
        title = str(paper.get("title") or recipe.get("name") or model_id)
        title_zh = str(paper.get("titleZh") or title)

        paper_path = ROOT / "public" / "models" / model_id / "paper.png"
        paper_src = f"/models/{model_id}/paper.png" if paper_path.is_file() else None
        teaser = teaser_src(model_id)
        overview = overview_src(model_id)

        if not overview:
            for figure in docs.get("figures") or []:
                if not isinstance(figure, dict):
                    continue
                figure_src = str(figure.get("src") or "")
                if figure.get("kind", "image") != "image":
                    continue
                if not figure_src.startswith("/models/"):
                    continue
                figure_path = ROOT / "public" / figure_src.lstrip("/")
                if not figure_path.is_file() or figure_path.stat().st_size <= 800:
                    continue
                if figure_src.lower().endswith("/paper.png"):
                    continue
                if any(figure_src.lower().endswith(name) for name in ("/teaser.png", "/teaser.jpg", "/teaser.webp", "/fig1.png")):
                    continue
                overview = figure_src
                break

        entry: dict[str, str] = {"modelId": model_id}
        if paper_src:
            entry["paperSrc"] = paper_src
        if teaser:
            teaser_en, teaser_zh = caption_pair(model_id, "teaser")
            entry["teaserSrc"] = teaser
            entry["teaserAlt"] = teaser_en
            entry["teaserAltZh"] = teaser_zh
            entry["teaserCaption"] = teaser_en
            entry["teaserCaptionZh"] = teaser_zh
        if overview:
            overview_en, overview_zh = caption_pair(model_id, "overview")
            entry["src"] = overview
            entry["alt"] = overview_en
            entry["altZh"] = overview_zh
            entry["caption"] = overview_en
            entry["captionZh"] = overview_zh
        elif paper_src:
            entry["alt"] = title
            entry["altZh"] = title_zh
            entry["caption"] = title
            entry["captionZh"] = title_zh
        registry[model_id] = entry

    payload = {"version": 1, "diagrams": registry}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with_src = sum(1 for item in registry.values() if item.get("src"))
    with_teaser = sum(1 for item in registry.values() if item.get("teaserSrc"))
    with_paper = sum(1 for item in registry.values() if item.get("paperSrc"))
    print(
        f"wrote {OUT_PATH.name}: {len(registry)} models, "
        f"{with_teaser} teasers, {with_src} overview figures, {with_paper} paper thumbnails"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
