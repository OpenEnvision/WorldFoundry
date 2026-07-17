#!/usr/bin/env python3
"""Generate docs/fumadocs/lib/upstream-acknowledgements-data.json from WorldFoundry catalogs."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "lib" / "upstream-acknowledgements-data.json"

MODEL_FAMILY_META = {
    "video": ("Video", "视频"),
    "world_models": ("World models", "世界模型"),
    "three_d_four_d": ("3D & 4D", "3D 与 4D"),
    "vla_va_wam": ("Embodied", "具身智能"),
    "hosted_api": ("Hosted API", "托管 API"),
}

BENCHMARK_GROUP_META = {
    "video": ("Video & world benchmarks", "视频与世界模型评测"),
    "embodied": ("Embodied benchmarks", "具身智能评测"),
}

INFRASTRUCTURE = [
    {
        "name": "FastVideo",
        "url": "https://github.com/hao-ai-lab/FastVideo",
        "summary": "Unified inference and post-training framework for accelerated video generation.",
        "summary_zh": "面向加速视频生成的统一推理与后训练框架。",
    },
    {
        "name": "OpenWorldLib",
        "url": "https://github.com/OpenDCAI/OpenWorldLib",
        "summary": "Unified codebase for advanced world models.",
        "summary_zh": "面向高级世界模型的统一代码库。",
    },
    {
        "name": "VLA Evaluation Harness",
        "url": "https://github.com/allenai/vla-evaluation-harness",
        "summary": "Framework for evaluating VLA models on robot simulation benchmarks.",
        "summary_zh": "在机器人仿真 benchmark 上评测 VLA 模型的框架。",
    },
]

GITHUB_URL = re.compile(r"https?://github\.com/([^/]+/[^/#?]+)")


def load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def model_items(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [item for item in data["models"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def benchmark_item(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    if isinstance(data, list) and data:
        item = data[0] if isinstance(data[0], dict) else {}
    elif isinstance(data, dict):
        item = data
    else:
        item = {}
    bid = item.get("benchmark_id") or item.get("id") or path.stem
    name = item.get("display_name") or item.get("name") or bid
    return {"id": bid, "name": name, "data": item}


def collect_github_urls(value: Any, urls: list[str] | None = None) -> list[str]:
    if urls is None:
        urls = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"url", "official_repo_url", "repo_url", "repository"} and isinstance(child, str):
                if "github.com" in child:
                    urls.append(child)
            else:
                collect_github_urls(child, urls)
    elif isinstance(value, list):
        for child in value:
            collect_github_urls(child, urls)
    elif isinstance(value, str) and "github.com" in value:
        urls.append(value)
    return urls


def normalize_github(url: str) -> str | None:
    match = GITHUB_URL.search(url)
    if not match:
        return None
    owner_repo = match.group(1).split()[0]
    if owner_repo.endswith(".git"):
        owner_repo = owner_repo[:-4]
    return f"https://github.com/{owner_repo}"


def repo_label(url: str) -> str:
    match = GITHUB_URL.search(url)
    if not match:
        return url
    owner, repo = match.group(1).split("/", 1)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"{owner}/{repo}"


def add_entry(bucket: OrderedDict[str, dict[str, str]], name: str, url: str) -> None:
    normalized = normalize_github(url)
    if not normalized:
        return
    current = bucket.get(normalized)
    if current is None or len(name) < len(current["name"]):
        bucket[normalized] = {"name": name, "url": normalized}


def model_families() -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for family_id, (label, label_zh) in MODEL_FAMILY_META.items():
        catalog_dir = ROOT / "worldfoundry/data/models/catalog" / family_id
        if not catalog_dir.is_dir():
            continue
        bucket: OrderedDict[str, dict[str, str]] = OrderedDict()
        for path in sorted(catalog_dir.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            for item in model_items(path):
                name = item.get("display_name") or item.get("name") or path.stem
                for url in collect_github_urls(item):
                    add_entry(bucket, str(name), url)
        entries = sorted(bucket.values(), key=lambda entry: entry["name"].lower())
        families.append(
            {
                "id": family_id,
                "label": label,
                "labelZh": label_zh,
                "count": len(entries),
                "entries": entries,
            }
        )
    return families


def benchmark_groups() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group_id, (label, label_zh) in BENCHMARK_GROUP_META.items():
        catalog_dir = ROOT / "worldfoundry/data/benchmarks/catalog" / group_id
        if not catalog_dir.is_dir():
            continue
        entries: list[dict[str, str]] = []
        for path in sorted(catalog_dir.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            item = benchmark_item(path)
            urls = [normalize_github(url) for url in collect_github_urls(item["data"])]
            urls = [url for url in urls if url]
            if not urls:
                continue
            entries.append(
                {
                    "name": str(item["name"]),
                    "url": urls[0],
                    "repo": repo_label(urls[0]),
                }
            )
        entries.sort(key=lambda entry: entry["name"].lower())
        groups.append(
            {
                "id": group_id,
                "label": label,
                "labelZh": label_zh,
                "count": len(entries),
                "entries": entries,
            }
        )
    return groups


def main() -> None:
    families = model_families()
    groups = benchmark_groups()
    payload = {
        "modelsTotal": sum(family["count"] for family in families),
        "benchmarksTotal": sum(group["count"] for group in groups),
        "infrastructure": INFRASTRUCTURE,
        "modelFamilies": families,
        "benchmarkGroups": groups,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {OUT} modelRepos={payload['modelsTotal']} "
        f"benchmarks={payload['benchmarksTotal']}"
    )


if __name__ == "__main__":
    main()
