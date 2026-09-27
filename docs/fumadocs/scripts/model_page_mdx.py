"""Render per-model fumadocs MDX pages from recipe JSON.

Generated files are the article source of truth after the first write. Files
with ``pageSource: authored`` (or any existing file that is not marked
``generated``) are left untouched so HunyuanVideo-style hand-written pages
survive ``models:generate``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PAGES_DIR = Path(__file__).resolve().parents[1] / "content/docs/guides/supported-models"
RESERVED_NAMES = {"index.mdx", "index.zh.mdx", "meta.json", "meta.zh.json"}
RESERVED_STEMS = {"index", "meta"}
PAGE_SOURCE_RE = re.compile(r"^pageSource:\s*(authored|generated)\s*$", re.MULTILINE)

COPY = {
    "en": {
        "architecture": "Architecture",
        "usage": "Runtime contract",
        "use_cases": "Typical uses",
        "hardware": "Hardware guidance",
        "min_vram": "Minimum VRAM",
        "recommended": "Recommended",
        "benchmarks": "Evaluation benchmarks",
        "sources": "Upstream sources",
        "limitations": "Readiness and limitations",
        "status": "Catalog status",
        "publisher": "Publisher",
        "company": "Company",
        "university": "University",
        "lab": "Research lab",
    },
    "zh": {
        "architecture": "架构",
        "usage": "运行时契约",
        "use_cases": "典型用途",
        "hardware": "硬件建议",
        "min_vram": "最低显存",
        "recommended": "推荐配置",
        "benchmarks": "评测 Benchmark",
        "sources": "上游来源",
        "limitations": "就绪状态与限制",
        "status": "目录状态",
        "publisher": "发表机构",
        "company": "企业",
        "university": "高校",
        "lab": "研究机构",
    },
}

STATUS_LABELS_ZH = {
    "verified": "Runner 已验证",
    "integrated": "已集成",
    "runtime_ported": "运行时已移植",
    "profile": "仅有运行环境配置",
    "planned": "计划中",
    "blocked": "已阻塞",
}

SOURCE_LABELS_ZH = {
    "Project": "项目主页",
    "Paper": "论文",
    "Source": "源码",
    "Upstream README": "上游 README",
    "Documentation": "文档",
}


def page_filename(model_id: str, locale: str) -> str:
    return f"{model_id}.mdx" if locale == "en" else f"{model_id}.zh.mdx"


def page_source(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return "authored"
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "authored"
    match = PAGE_SOURCE_RE.search(parts[1])
    return match.group(1) if match else "authored"


def is_generated(path: Path) -> bool:
    return page_source(path) == "generated"


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def mdx_escape(value: str) -> str:
    return value.replace("{", "\\{").replace("}", "\\}").replace("<", "\\<")


def paragraphs(values: list[Any] | None) -> list[str]:
    result: list[str] = []
    for item in values or []:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def locale_paragraphs(docs: dict[str, Any], key: str, locale: str) -> list[str]:
    zh_key = f"{key}Zh"
    if locale == "zh":
        return paragraphs(docs.get(zh_key)) or paragraphs(docs.get(key))
    return paragraphs(docs.get(key))


def figure_block(figure: dict[str, Any], locale: str) -> str | None:
    src = str(figure.get("src") or "").strip()
    if not src:
        return None
    caption = str(figure.get("captionZh") if locale == "zh" else figure.get("caption") or "").strip()
    if not caption:
        caption = str(figure.get("caption") or "").strip()
    alt = mdx_escape(caption)
    kind = figure.get("kind") or "image"
    src_l = src.lower()
    if kind == "video" or src_l.endswith((".mp4", ".webm", ".mov", ".m4v")) or "/demos/" in src_l:
        return None
    image = f'<DocsImage src="{src}" alt="{alt}" />'
    if caption:
        return (
            f'<figure className="wf-recipe-media is-diagram">\n  {image}\n  '
            f"<figcaption>{alt}</figcaption>\n</figure>"
        )
    return f'<figure className="wf-recipe-media is-diagram">\n  {image}\n</figure>'


def lead_meta(recipe: dict[str, Any], locale: str) -> list[str]:
    docs = recipe.get("docs") or {}
    copy = COPY[locale]
    lines: list[str] = []
    publisher = docs.get("publisher") or {}
    publisher_name = publisher.get("nameZh") if locale == "zh" else publisher.get("name")
    publisher_name = publisher_name or publisher.get("name")
    kind = publisher.get("kind")
    kind_label = {
        "company": copy["company"],
        "university": copy["university"],
        "lab": copy["lab"],
    }.get(kind)
    if publisher_name:
        suffix = f" · {kind_label}" if kind_label else ""
        lines.append(f"**{copy['publisher']}:** {mdx_escape(str(publisher_name))}{suffix}")

    modalities = docs.get("modalities") or {}
    inputs = [str(item) for item in modalities.get("inputs") or [] if item]
    outputs = [str(item) for item in modalities.get("outputs") or [] if item]
    if inputs or outputs:
        arrow = " → ".join(filter(None, [" · ".join(inputs), " · ".join(outputs)]))
        lines.append(arrow)

    paper = docs.get("paper") or {}
    title = str(paper.get("title") or "").strip()
    if title:
        arxiv_id = str(paper.get("arxivId") or "").strip()
        href = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        cite = f"[{mdx_escape(title)}]({href})" if href else mdx_escape(title)
        extra = ", ".join(str(part) for part in (paper.get("venue"), paper.get("year")) if part)
        lines.append(f"{cite} ({extra})" if extra else cite)
    return lines


def render_model_page(recipe: dict[str, Any], locale: str) -> str:
    docs = recipe.get("docs") or {}
    copy = COPY[locale]
    model_id = str(recipe["id"])
    name = str(recipe.get("name") or model_id)
    overview = locale_paragraphs(docs, "overview", locale)
    summary = overview[0] if locale == "zh" and overview else str(recipe.get("summary") or name)
    architecture = locale_paragraphs(docs, "architecture", locale)
    usage = locale_paragraphs(docs, "usageNotes", locale)
    limitations = locale_paragraphs(docs, "limitations", locale)
    use_cases = locale_paragraphs(docs, "useCases", locale)
    hardware = docs.get("hardware") or {}
    figures = [item for item in (docs.get("figures") or []) if isinstance(item, dict)]
    benchmarks = [item for item in (docs.get("benchmarks") or []) if isinstance(item, dict)]
    sources = [item for item in (recipe.get("sources") or []) if isinstance(item, dict)]
    hub_prefix = "/zh/docs/evaluation/benchmark-hub" if locale == "zh" else "/docs/evaluation/benchmark-hub"

    chunks = [
        "---",
        f"title: {yaml_string(name)}",
        f"description: {yaml_string(summary)}",
        "pageSource: generated",
        "---",
        "",
    ]

    meta_lines = lead_meta(recipe, locale)
    if meta_lines:
        chunks.extend([line for line in meta_lines])
        chunks.append("")

    for paragraph in overview:
        chunks.append(mdx_escape(paragraph))
        chunks.append("")

    status = recipe.get("status") or {}
    status_label = str(status.get("label") or "").strip()
    if locale == "zh":
        status_label = STATUS_LABELS_ZH.get(str(status.get("group") or ""), status_label)
    if status_label:
        chunks.extend([f"**{copy['status']}:** {mdx_escape(status_label)}", ""])

    if sources:
        chunks.extend([f"## {copy['sources']}", ""])
        for source in sources:
            url = str(source.get("url") or "").strip()
            if url.startswith(("https://", "http://")):
                label = str(source.get("label") or source.get("kind") or "Source")
                if locale == "zh":
                    label = SOURCE_LABELS_ZH.get(label, label)
                label = mdx_escape(label)
                chunks.append(f"- [{label}]({url})")
        chunks.append("")

    if figures:
        teaser = figure_block(figures[0], locale)
        if teaser:
            chunks.append(teaser)
            chunks.append("")

    chunks.append(f'<ModelCommandBuilder modelId="{model_id}" locale="{locale}" />')
    chunks.append("")

    if architecture:
        chunks.append(f"## {copy['architecture']}")
        chunks.append("")
        for paragraph in architecture:
            chunks.append(mdx_escape(paragraph))
            chunks.append("")

    extra_figures = figures[1:]
    if extra_figures:
        for figure in extra_figures:
            block = figure_block(figure, locale)
            if block:
                chunks.append(block)
                chunks.append("")

    if usage:
        chunks.append(f"## {copy['usage']}")
        chunks.append("")
        for paragraph in usage:
            chunks.append(mdx_escape(paragraph))
            chunks.append("")

    if use_cases:
        chunks.append(f"## {copy['use_cases']}")
        chunks.append("")
        for item in use_cases:
            chunks.append(f"- {mdx_escape(item)}")
        chunks.append("")

    notes = [str(item).strip() for item in (hardware.get("notes") or []) if str(item).strip()]
    min_vram = hardware.get("minVramGb")
    recommended = hardware.get("recommended")
    if min_vram is not None or recommended or notes:
        chunks.append(f"## {copy['hardware']}")
        chunks.append("")
        if min_vram is not None:
            chunks.append(f"- **{copy['min_vram']}:** {min_vram} GB")
        if recommended:
            chunks.append(f"- **{copy['recommended']}:** {mdx_escape(str(recommended))}")
        for note in notes:
            chunks.append(f"- {mdx_escape(note)}")
        chunks.append("")

    if benchmarks:
        chunks.append(f"## {copy['benchmarks']}")
        chunks.append("")
        for bench in benchmarks:
            bench_id = str(bench.get("id") or "").strip()
            if not bench_id:
                continue
            bench_name = str(bench.get("name") or bench_id)
            reason = bench.get("reasonZh") if locale == "zh" else bench.get("reason")
            reason = str(reason or bench.get("reason") or "").strip()
            link = f"[{mdx_escape(bench_name)}]({hub_prefix}/{bench_id})"
            chunks.append(f"- {link}" + (f" — {mdx_escape(reason)}" if reason else ""))
        chunks.append("")

    if limitations:
        chunks.extend([f"## {copy['limitations']}", ""])
        for item in limitations:
            chunks.append(f"- {mdx_escape(item)}")
        chunks.append("")

    chunks.append(f'<ModelRelatedRecipes modelId="{model_id}" locale="{locale}" />')
    chunks.append("")
    return "\n".join(chunks)


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
