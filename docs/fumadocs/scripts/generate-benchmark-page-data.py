#!/usr/bin/env python3
"""Build benchmark-page-data.json from manifests and curated documentation copy.

Dimensions, data composition, and metrics are extracted from recorded fields.
Leaderboard rows are emitted only when the catalog already stores published
scores — sample_results fixtures are ignored. Missing facts stay missing.
Editorial details that are not present in manifests live in
benchmark-page-overrides.json, never in the generated output itself.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = ROOT / "worldfoundry/data/benchmarks/catalog"
TASK_ROOT = ROOT / "worldfoundry/data/benchmarks/tasks/external"
MDX_ROOT = DOCS_ROOT / "content/docs/evaluation/benchmark-hub"
OUT = DOCS_ROOT / "lib" / "benchmark-page-data.json"
OVERRIDES = DOCS_ROOT / "lib" / "benchmark-page-overrides.json"

GENERIC_SPLITS = {
    "standard",
    "eval",
    "official-validation",
    "benchmark_defined",
    "all",
}

SKIP_DIMENSION_METRIC = re.compile(
    r"(average|overall_quality|temporal_quality|frame_quality|text_alignment|_score$)",
    re.I,
)

LABEL_ZH = {
    "major_content": "主要内容",
    "attribute_control": "属性控制",
    "prompt_complexity": "Prompt 复杂度",
    "visual_quality": "视觉质量",
    "text_video_alignment": "文本-视频对齐",
    "motion_quality": "运动质量",
    "temporal_consistency": "时间一致性",
    "human_fidelity": "人体保真",
    "controllability": "可控性",
    "creativity": "创造性",
    "physics": "物理",
    "commonsense": "常识",
    "generation": "生成",
    "understanding": "理解",
    "calibration": "校准",
    "coverage": "覆盖",
    "perception_text": "感知（文本）",
    "perception_graphic": "感知（图形）",
    "formulation_text": "建模（文本）",
    "formulation_graphic": "建模（图形）",
    "deduction": "推理",
    "spatial": "空间",
    "object": "物体",
    "goal": "目标",
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-10",
    "libero_90": "LIBERO-90",
    "mt10": "MT10",
    "mt50": "MT50",
    "ml10": "ML10",
    "ml45": "ML45",
    "abc": "ABC",
    "d": "D",
    "google_robot": "Google Robot",
    "widowx_bridge": "WidowX Bridge",
    "simple_60s": "Simple 60s",
    "hard_60s": "Hard 60s",
    "natural25": "Natural-25",
    "demo_clean": "Demo clean",
    "demo_randomized": "Demo randomized",
    "physical": "物理",
    "geometry": "几何",
    "interaction": "交互",
    "text": "文本",
    "video": "视频",
    "image": "图像",
    "language": "语言",
    "vision": "视觉",
    "action": "动作",
    "state": "状态",
    "generated-video": "生成视频",
}

KIND_ZH = {
    "axis": "轴",
    "split": "划分",
    "track": "赛道",
    "category": "类别",
    "task": "任务",
    "suite": "套件",
}


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) or {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def humanize(value: str) -> str:
    text = str(value).replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", text).title() if text else ""


def license_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    status = str(value.get("status") or "").strip()
    if status == "upstream_readme_declaration":
        notes = str(value.get("notes") or "")
        match = re.search(r"\b(?:Apache-2\.0|MIT|BSD-3-Clause)\b", notes, re.I)
        return f"{match.group(0)} (upstream README declaration)" if match else "Upstream README declaration"
    return {
        "apache-2.0": "Apache-2.0",
        "mit": "MIT",
        "unlicensed_upstream": "No published upstream license",
        "restricted_noncommercial": "Restricted; see upstream terms",
    }.get(status, humanize(status))


def label_zh(value: str) -> str:
    key = str(value).strip()
    if key in LABEL_ZH:
        return LABEL_ZH[key]
    return humanize(key)


def first_url(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("url") or "")
    if isinstance(value, list):
        for item in value:
            found = first_url(item)
            if found:
                return found
    return ""


def github_urls(value: Any) -> list[str]:
    urls: list[str] = []
    for item in as_list(value):
        url = first_url(item)
        if url and url not in urls:
            urls.append(url)
    return urls


def interesting_splits(raw: Any) -> list[str]:
    splits = [str(item).strip() for item in as_list(raw) if str(item).strip()]
    useful = [item for item in splits if item.lower() not in GENERIC_SPLITS]
    return useful or []


def harvest_metric_copy(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text()
    out: dict[str, str] = {}
    for block in re.findall(r'<li className="pi-kv-row">(.*?)</li>', text, re.S):
        code = re.search(r"<code>([^<]+)</code>", block)
        desc = re.search(r"<p>(.*?)</p>", block, re.S)
        if code and desc:
            out[code.group(1).strip()] = re.sub(r"\s+", " ", desc.group(1)).strip()
    return out


def collect_catalogs() -> dict[str, dict[str, Any]]:
    catalogs: dict[str, dict[str, Any]] = {}
    for path in sorted(CATALOG_ROOT.glob("*/*.yaml")):
        if path.name.startswith("_"):
            continue
        data = load_yaml(path)
        if isinstance(data, dict) and data.get("id"):
            catalogs[str(data["id"])] = data
    return catalogs


def collect_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(TASK_ROOT.glob("*.yaml")):
        data = load_yaml(path)
        if not isinstance(data, dict):
            continue
        bench_id = str(data.get("id") or data.get("benchmark") or path.stem)
        tasks[bench_id] = data
    return tasks


def metric_rows(
    catalog: dict[str, Any],
    en_copy: dict[str, str],
    zh_copy: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in catalog.get("metrics") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        metric_id = str(item["id"])
        description = str(item.get("description") or en_copy.get(metric_id) or "").strip()
        description_zh = str(zh_copy.get(metric_id) or "").strip()
        rows.append(
            {
                "id": metric_id,
                "name": str(item.get("name") or humanize(metric_id)),
                "description": description,
                "descriptionZh": description_zh,
                "higherIsBetter": bool(item.get("higher_is_better", True)),
                "primary": bool(item.get("primary")),
                "leaderboardKey": str(item.get("leaderboard_key") or metric_id),
            }
        )
    return rows


def add_dimension(
    items: list[dict[str, Any]],
    seen: set[str],
    dim_id: str,
    *,
    name: str = "",
    name_zh: str = "",
    kind: str = "axis",
    description: str = "",
    description_zh: str = "",
) -> None:
    key = dim_id.strip() or name.strip()
    if not key or key in seen:
        return
    seen.add(key)
    items.append(
        {
            "id": key,
            "name": name or humanize(key),
            "nameZh": name_zh or label_zh(key),
            "kind": kind,
            "description": description,
            "descriptionZh": description_zh,
        }
    )


def overlay_dimensions(bench_id: str) -> list[dict[str, Any]]:
    return DIMENSION_OVERLAYS.get(bench_id, [])


def collect_dimensions(
    bench_id: str,
    catalog: dict[str, Any],
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    overlay = overlay_dimensions(bench_id)
    if overlay:
        for row in overlay:
            add_dimension(
                items,
                seen,
                str(row["id"]),
                name=str(row.get("name") or ""),
                name_zh=str(row.get("nameZh") or ""),
                kind=str(row.get("kind") or "axis"),
                description=str(row.get("description") or ""),
                description_zh=str(row.get("descriptionZh") or ""),
            )
        return items

    surface = catalog.get("evaluation_surface") or {}
    if not isinstance(surface, dict):
        surface = {}

    for aspect in as_list(surface.get("prompt_aspects")):
        add_dimension(items, seen, str(aspect), kind="axis")

    dimensions = surface.get("dimensions")
    if isinstance(dimensions, list):
        for item in dimensions:
            add_dimension(items, seen, str(item), kind="axis")

    categories = surface.get("categories")
    if isinstance(categories, list):
        for item in categories:
            add_dimension(items, seen, str(item), kind="category")

    for track in as_list(surface.get("tracks")):
        add_dimension(items, seen, str(track), kind="track")

    for split in interesting_splits(task.get("splits")):
        add_dimension(items, seen, split, kind="split")

    if not items:
        official_metrics = [
            item
            for item in catalog.get("metrics") or []
            if isinstance(item, dict)
            and item.get("id")
            and not item.get("primary")
            and "Official" in str(item.get("description") or "")
            and not SKIP_DIMENSION_METRIC.search(str(item["id"]))
        ]
        for item in official_metrics:
            add_dimension(
                items,
                seen,
                str(item["id"]),
                name=str(item.get("name") or ""),
                kind="axis",
                description=str(item.get("description") or ""),
            )

    if not items:
        for kind in catalog.get("benchmark_kind") or []:
            add_dimension(items, seen, str(kind), kind="axis")
        for domain in (catalog.get("domains") or [])[:6]:
            add_dimension(items, seen, str(domain), kind="axis")

    return items


def prompt_count(catalog: dict[str, Any]) -> int | None:
    assets = catalog.get("benchmark_assets") or {}
    if isinstance(assets, dict):
        suite = assets.get("prompt_suite") or {}
        if isinstance(suite, dict):
            for key in ("num_prompts", "prompt_count", "generation_request_count"):
                value = suite.get(key)
                if isinstance(value, int):
                    return value
    surface = catalog.get("evaluation_surface") or {}
    if isinstance(surface, dict):
        claimed = surface.get("claimed_prompts") or surface.get("prompt_count")
        if isinstance(claimed, int):
            return claimed
    return None


def overlay_data(bench_id: str) -> dict[str, Any]:
    return DATA_OVERLAYS.get(bench_id, {})


def collect_data(
    bench_id: str,
    catalog: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    overlay = overlay_data(bench_id)
    official = catalog.get("official_sources") or {}
    if not isinstance(official, dict):
        official = {}
    assets = catalog.get("benchmark_assets") or {}
    suite = assets.get("prompt_suite") if isinstance(assets, dict) else {}
    if not isinstance(suite, dict):
        suite = {}

    count = overlay.get("count")
    if not isinstance(count, int):
        count = prompt_count(catalog)
    unit = str(overlay.get("unit") or ("prompts" if count else ""))
    unit_zh = str(overlay.get("unitZh") or ("条 prompt" if unit == "prompts" else unit))

    sources: list[dict[str, str]] = []
    for item in as_list(official.get("huggingface_datasets")):
        if isinstance(item, dict) and item.get("repo_id"):
            sources.append(
                {
                    "kind": "huggingface",
                    "id": str(item["repo_id"]),
                    "license": license_label(item.get("license") or official.get("license")),
                }
            )
    for url in github_urls(official.get("github")):
        sources.append({"kind": "github", "id": url, "license": license_label(official.get("license"))})

    dataset = catalog.get("dataset") if isinstance(catalog.get("dataset"), dict) else {}
    notes = [str(item) for item in as_list(overlay.get("notes")) if item]
    notes_zh = [str(item) for item in as_list(overlay.get("notesZh")) if item]
    if dataset.get("not_applicable") and dataset.get("reason"):
        reason = str(dataset["reason"]).strip()
        if reason and reason not in notes:
            notes.append(reason)

    if suite.get("name") and not overlay.get("summary"):
        notes.append(f"Prompt suite: {suite['name']}.")

    size_label = ""
    size_label_zh = ""
    if isinstance(count, int):
        size_label = f"{count:,} {unit}".strip()
        size_label_zh = f"{count:,} {unit_zh}".strip()
    if overlay.get("sizeLabel"):
        size_label = str(overlay["sizeLabel"])
        size_label_zh = str(overlay.get("sizeLabelZh") or size_label)

    summary = str(overlay.get("summary") or size_label)
    summary_zh = str(overlay.get("summaryZh") or size_label_zh or summary)

    return {
        "summary": summary,
        "summaryZh": summary_zh,
        "sizeLabel": size_label,
        "sizeLabelZh": size_label_zh,
        "count": count,
        "unit": unit,
        "sources": sources,
        "splits": interesting_splits(task.get("splits") or overlay.get("splits")),
        "modalities": [str(item) for item in catalog.get("modalities") or []],
        "license": license_label(official.get("license")),
        "notes": notes,
        "notesZh": notes_zh,
        "notApplicable": bool(dataset.get("not_applicable")),
    }


def build_entry(
    bench_id: str,
    catalog: dict[str, Any],
    task: dict[str, Any],
    en_copy: dict[str, str],
    zh_copy: dict[str, str],
) -> dict[str, Any]:
    official = catalog.get("official_sources") or {}
    if not isinstance(official, dict):
        official = {}
    metrics = metric_rows(catalog, en_copy, zh_copy)
    dimensions = collect_dimensions(bench_id, catalog, task)
    data = collect_data(bench_id, catalog, task)
    board = LEADERBOARDS.get(bench_id, {})
    entries = [item for item in as_list(board.get("entries")) if isinstance(item, dict)]
    return {
        "id": bench_id,
        "name": str(catalog.get("name") or bench_id),
        "projectPage": str(official.get("project_page") or ""),
        "paperUrl": str(official.get("paper_url") or ""),
        "dimensions": dimensions,
        "data": data,
        "metrics": metrics,
        "leaderboard": {
            "ingested": bool(entries),
            "source": str(board.get("source") or ""),
            "sourceZh": str(board.get("sourceZh") or ""),
            "officialUrl": str(board.get("officialUrl") or official.get("project_page") or ""),
            "entries": entries,
        },
    }


# Facts already stated in catalog YAML or existing About prose. No invented counts.
DIMENSION_OVERLAYS: dict[str, list[dict[str, Any]]] = {
    "fetv": [
        {
            "id": "major_content",
            "name": "Major content",
            "nameZh": "主要内容",
            "kind": "axis",
            "description": "What the clip depicts — the primary subject or event.",
            "descriptionZh": "片段描绘的主要内容或事件。",
        },
        {
            "id": "attribute_control",
            "name": "Attribute control",
            "nameZh": "属性控制",
            "kind": "axis",
            "description": "Whether requested attributes (color, count, style) are realized.",
            "descriptionZh": "请求的属性（颜色、数量、风格等）是否被实现。",
        },
        {
            "id": "prompt_complexity",
            "name": "Prompt complexity",
            "nameZh": "Prompt 复杂度",
            "kind": "axis",
            "description": "Simple vs compositional prompt difficulty.",
            "descriptionZh": "简单 prompt 与组合式 prompt 的难度分层。",
        },
    ],
    "libero": [
        {
            "id": "libero_spatial",
            "name": "LIBERO-Spatial",
            "nameZh": "LIBERO-Spatial",
            "kind": "suite",
            "description": "Spatial-relation transfer across tabletop layouts.",
            "descriptionZh": "桌面布局上的空间关系迁移。",
        },
        {
            "id": "libero_object",
            "name": "LIBERO-Object",
            "nameZh": "LIBERO-Object",
            "kind": "suite",
            "description": "Object-identity transfer across manipulation tasks.",
            "descriptionZh": "操作任务中的物体身份迁移。",
        },
        {
            "id": "libero_goal",
            "name": "LIBERO-Goal",
            "nameZh": "LIBERO-Goal",
            "kind": "suite",
            "description": "Goal-specification transfer.",
            "descriptionZh": "目标描述迁移。",
        },
        {
            "id": "libero_100",
            "name": "LIBERO-100",
            "nameZh": "LIBERO-100",
            "kind": "suite",
            "description": "LIBERO-10 plus LIBERO-90 long-horizon mix.",
            "descriptionZh": "LIBERO-10 与 LIBERO-90 组成的长程混合套件。",
        },
    ],
    "vbench": [
        {
            "id": "temporal_quality",
            "name": "Temporal quality",
            "nameZh": "时间质量",
            "kind": "axis",
            "description": "Subject/background consistency, flicker, motion smoothness, dynamic degree.",
            "descriptionZh": "主体/背景一致性、闪烁、运动平滑与动态程度。",
        },
        {
            "id": "frame_quality",
            "name": "Frame quality",
            "nameZh": "帧质量",
            "kind": "axis",
            "description": "Aesthetic quality and imaging quality.",
            "descriptionZh": "美学质量与成像质量。",
        },
        {
            "id": "semantics",
            "name": "Semantics / text alignment",
            "nameZh": "语义 / 文本对齐",
            "kind": "axis",
            "description": "Object class, multiple objects, human action, color, spatial relationship, scene, appearance/temporal style, overall consistency.",
            "descriptionZh": "物体类别、多物体、人体动作、颜色、空间关系、场景、外观/时间风格与整体一致性。",
        },
    ],
    "vbench-2.0": [
        {"id": "creativity", "name": "Creativity", "nameZh": "创造性", "kind": "category", "description": "Composition and diversity.", "descriptionZh": "构图与多样性。"},
        {"id": "commonsense", "name": "Commonsense", "nameZh": "常识", "kind": "category"},
        {"id": "controllability", "name": "Controllability", "nameZh": "可控性", "kind": "category"},
        {"id": "human_fidelity", "name": "Human fidelity", "nameZh": "人体保真", "kind": "category"},
        {"id": "physics", "name": "Physics", "nameZh": "物理", "kind": "category"},
    ],
    "4dworldbench": [
        {
            "id": "perceptual_quality",
            "name": "Perceptual quality",
            "nameZh": "感知质量",
            "kind": "axis",
            "description": "CLIP-IQA, CLIP-aesthetic, FastVQA.",
            "descriptionZh": "CLIP-IQA、CLIP-aesthetic、FastVQA。",
        },
        {
            "id": "condition_4d_alignment",
            "name": "Condition-4D alignment",
            "nameZh": "Condition-4D 对齐",
            "kind": "axis",
            "description": "Attribute, relationship, motion, event, scene, and camera-error control.",
            "descriptionZh": "属性、关系、运动、事件、场景与相机误差控制。",
        },
        {
            "id": "physical_realism",
            "name": "Physical realism",
            "nameZh": "物理真实感",
            "kind": "axis",
            "description": "LLM-assisted physics reasoning.",
            "descriptionZh": "LLM 辅助的物理推理。",
        },
        {
            "id": "four_d_consistency",
            "name": "4D consistency",
            "nameZh": "4D 一致性",
            "kind": "axis",
            "description": "Viewpoint, motion smoothness, motion QA, and style consistency.",
            "descriptionZh": "视角、运动平滑、运动问答与风格一致性。",
        },
    ],
    "chronomagic-bench": [
        {
            "id": "temporal_coherence",
            "name": "Temporal coherence",
            "nameZh": "时间连贯",
            "kind": "axis",
            "description": "CHScore tracking-point coherence via CoTracker.",
            "descriptionZh": "CHScore：通过 CoTracker 的跟踪点连贯性。",
        },
        {
            "id": "metamorphic_amplitude",
            "name": "Metamorphic amplitude",
            "nameZh": "形态变化幅度",
            "kind": "axis",
            "description": "MTScore / GPT-4o-MTScore for large physical or biological state change.",
            "descriptionZh": "MTScore / GPT-4o-MTScore：大幅物理或生物状态变化。",
        },
    ],
}

DATA_OVERLAYS: dict[str, dict[str, Any]] = {
    "vbench": {
        "summary": "16 official dimensions; prompt suite is VBench_full_info.json in the GitHub repository",
        "summaryZh": "16 个官方维度；prompt suite 为 GitHub 仓库中的 VBench_full_info.json",
    },
    "libero": {
        "summary": "Four lifelong suites (Spatial, Object, Goal, LIBERO-100) on robosuite/MuJoCo",
        "summaryZh": "四个终身学习套件（Spatial、Object、Goal、LIBERO-100），基于 robosuite/MuJoCo",
    },
    "4dworldbench": {
        "summary": "Condition-to-4D JSON plus generated videos; official dataset download is not published",
        "summaryZh": "Condition-to-4D JSON 与生成视频；官方数据集下载尚未发布",
    },
    "fetv": {
        "count": 619,
        "unit": "prompts",
        "unitZh": "条 prompt",
        "summary": "619 prompts categorized on three orthogonal axes",
        "summaryZh": "619 条 prompt，沿三个正交轴分类",
    },
    "chronomagic-bench": {
        "count": 1649,
        "unit": "prompts",
        "unitZh": "条 prompt",
        "summary": "1,649 closed-set prompts; ChronoMagic-Bench dataset discovery records 1,799 rows",
        "summaryZh": "1,649 条闭集 prompt；ChronoMagic-Bench 数据集发现记录 1,799 行",
        "notes": ["Companion ChronoMagic / ChronoMagic-Pro / ChronoMagic-ProH training sets are listed in official sources."],
        "notesZh": ["官方来源同时列出 ChronoMagic / ChronoMagic-Pro / ChronoMagic-ProH 训练数据。"],
    },
    "evalcrafter": {"count": 700, "unit": "prompts", "unitZh": "条 prompt"},
    "t2v-compbench": {
        "count": 1400,
        "unit": "prompts",
        "unitZh": "条 prompt",
        "summary": "1,400 prompts across 7 composition categories",
        "summaryZh": "1,400 条 prompt，覆盖 7 个组合类别",
    },
    "phyfps-bench-gen": {"count": 100, "unit": "prompts", "unitZh": "条 prompt"},
    "phygenbench": {
        "count": 160,
        "unit": "prompts",
        "unitZh": "条 prompt",
        "notes": ["Catalog records 27 claimed physical laws."],
        "notesZh": ["目录记录声称覆盖 27 条物理定律。"],
    },
    "phyground": {
        "count": 250,
        "unit": "prompts",
        "unitZh": "条 prompt",
        "notes": ["Catalog records 13 claimed physical laws."],
        "notesZh": ["目录记录声称覆盖 13 条物理定律。"],
    },
    "physvidbench": {"count": 383, "unit": "prompts", "unitZh": "条 prompt"},
    "t2vworldbench": {
        "count": 1200,
        "unit": "prompts",
        "unitZh": "条 prompt",
        "notes": ["Catalog records 6 categories."],
        "notesZh": ["目录记录 6 个类别。"],
    },
    "videoverse": {"count": 300, "unit": "prompts", "unitZh": "条 prompt"},
    "wrbench": {
        "count": 500,
        "unit": "generation requests",
        "unitZh": "次生成请求",
        "summary": "Natural-25: 25 families × 100 semantic variants × 5 camera controls = 500 generation requests",
        "summaryZh": "Natural-25：25 个家族 × 100 个语义变体 × 5 种相机控制 = 500 次生成请求",
    },
    "libero-plus": {
        "count": 10030,
        "unit": "tasks",
        "unitZh": "个任务",
        "summary": "About prose records 10,030 tasks",
        "summaryZh": "简介记录 10,030 个任务",
    },
    "maniskill2": {
        "count": 20,
        "unit": "tasks",
        "unitZh": "个任务",
        "summary": "About prose records 20 tasks",
        "summaryZh": "简介记录 20 个任务",
    },
    "vlabench": {
        "count": 100,
        "unit": "tasks",
        "unitZh": "个任务",
        "summary": "About prose records 100 tasks",
        "summaryZh": "简介记录 100 个任务",
    },
    "sana-wm-bench": {
        "count": 80,
        "unit": "scenes",
        "unitZh": "个场景",
        "summary": "80 scenes on simple_60s and hard_60s splits",
        "summaryZh": "80 个场景，划分 simple_60s 与 hard_60s",
    },
    "stevo-bench": {
        "count": 225,
        "unit": "tasks",
        "unitZh": "个任务",
        "summary": "About prose records 225 tasks",
        "summaryZh": "简介记录 225 个任务",
    },
    "bridgedata-v2": {
        "notes": ["About prose records 24 environments."],
        "notesZh": ["简介记录 24 个环境。"],
    },
    "robotwin": {
        "notes": ["About prose records 731 objects."],
        "notesZh": ["简介记录 731 个物体。"],
    },
    "mirabench": {
        "notes": ["About prose records 17 metrics in 6 score families."],
        "notesZh": ["简介记录 17 项指标、6 个分数族。"],
    },
}

# Docs leaderboards live in docs/fumadocs/lib/benchmark-leaderboards.json
# (BenchmarkPageSections). Keep this dict empty so --check stays stable.
LEADERBOARDS: dict[str, dict[str, Any]] = {}


def merge_override(generated: Any, override: Any, path: str) -> Any:
    """Replace only explicitly curated leaves, preserving new manifest fields."""
    if not isinstance(generated, dict) or not isinstance(override, dict):
        return override
    unknown = set(override) - set(generated)
    if unknown:
        raise ValueError(f"unknown benchmark override fields at {path}: {sorted(unknown)}")
    return {
        key: merge_override(value, override[key], f"{path}.{key}") if key in override else value
        for key, value in generated.items()
    }


def load_page_overrides(page_ids: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != 1 or not isinstance(payload.get("pages"), dict):
        raise ValueError(f"invalid benchmark page overrides: {OVERRIDES}")
    overrides = payload["pages"]
    unknown = set(overrides) - page_ids
    if unknown:
        raise ValueError(f"overrides reference missing benchmarks: {sorted(unknown)}")
    return overrides


def build_payload() -> dict[str, Any]:
    catalogs = collect_catalogs()
    tasks = collect_tasks()
    pages: dict[str, Any] = {}
    for bench_id, catalog in catalogs.items():
        en_copy = harvest_metric_copy(MDX_ROOT / f"{bench_id}.mdx")
        zh_copy = harvest_metric_copy(MDX_ROOT / f"{bench_id}.zh.mdx")
        pages[bench_id] = build_entry(bench_id, catalog, tasks.get(bench_id, {}), en_copy, zh_copy)

    for bench_id, override in load_page_overrides(set(pages)).items():
        pages[bench_id] = merge_override(pages[bench_id], override, bench_id)

    ingested = sorted(bid for bid, page in pages.items() if page["leaderboard"]["ingested"])
    return {
        "generatedFrom": "worldfoundry/data/benchmarks/catalog + tasks/external + recorded About facts + curated docs overrides",
        "benchmarkCount": len(pages),
        "leaderboardIngestedIds": ingested,
        "leaderboardPlaceholderCount": len(pages) - len(ingested),
        "pages": pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != encoded:
            print(f"stale generated benchmark page data: {OUT}")
            return 1
        print(f"benchmark page data up to date: {OUT}")
        return 0
    OUT.write_text(encoded)
    print(
        f"wrote {payload['benchmarkCount']} benches, "
        f"{len(payload['leaderboardIngestedIds'])} ingested leaderboards, "
        f"{payload['leaderboardPlaceholderCount']} placeholders → {OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
