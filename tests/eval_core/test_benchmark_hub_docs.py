from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import formal_benchmark_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
FUMADOCS_ROOT = REPO_ROOT / "docs" / "fumadocs"
BENCHMARK_DETAILS_ROOT = FUMADOCS_ROOT / "content" / "docs" / "evaluation" / "benchmark-hub"
BENCHMARK_CATALOG_STATUS = FUMADOCS_ROOT / "lib" / "benchmark-catalog-status.json"
IGNORED_SCAN_DIRS = {".git", ".next", ".source", "build", "dist", "node_modules", "out", "tmp"}
FUMADOCS_SOURCE_ROOTS = tuple(
    FUMADOCS_ROOT / name for name in ("app", "components", "content", "lib", "mdx", "scripts")
)
PUBLIC_DOC_ROOTS = (REPO_ROOT / "README.md", *FUMADOCS_SOURCE_ROOTS)


def _iter_text_files(root: Path, suffixes: Iterable[str]) -> Iterator[Path]:
    allowed = set(suffixes)
    if root.is_file():
        if root.suffix in allowed:
            yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in IGNORED_SCAN_DIRS and not name.startswith((".next-", "out.bak."))
        ]
        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            if path.suffix in allowed:
                yield path


def test_fumadocs_benchmark_hub_does_not_reference_removed_app_path() -> None:
    removed_app_path = "apps/" + "benchmark-hub"
    removed_app_command = "cd " + removed_app_path
    offenders: list[str] = []
    for root in FUMADOCS_SOURCE_ROOTS:
        for path in _iter_text_files(root, {".md", ".mdx", ".ts", ".tsx", ".json"}):
            text = path.read_text(encoding="utf-8")
            if removed_app_path in text or removed_app_command in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_public_docs_do_not_reference_removed_evaluation_routes_or_gpu_zero_defaults() -> None:
    forbidden = (
        "/docs/evaluation/" + "tasks",
        "/zh/docs/evaluation/" + "tasks",
        "benchmark-" + "runs.mdx",
        "benchmark-" + "runs.zh.mdx",
        "CUDA_VISIBLE_DEVICES=" + "0",
        "--cuda-visible-devices " + "0",
        "${CUDA_VISIBLE_DEVICES:-" + "0}",
        "worldfoundry/evaluation/" + "benchmark_zoo",
        "worldfoundry/evaluation/" + "model_zoo",
        "worldfoundry/evaluation/" + "pipeline_dispatch",
        "<Folder name=\"" + "benchmark_zoo" + "\" defaultOpen>",
        "<Folder name=\"" + "model_zoo" + "\" defaultOpen>",
    )
    offenders: dict[str, list[str]] = {}

    for root in PUBLIC_DOC_ROOTS:
        for path in _iter_text_files(root, {".md", ".mdx", ".ts", ".tsx", ".json"}):
            text = path.read_text(encoding="utf-8")
            matches = [needle for needle in forbidden if needle in text]
            if matches:
                offenders[str(path.relative_to(REPO_ROOT))] = matches

    assert offenders == {}


def test_benchmark_catalog_rows_link_to_per_benchmark_detail_pages() -> None:
    component = (FUMADOCS_ROOT / "components" / "benchmark-recipe-catalog.tsx").read_text(encoding="utf-8")

    assert "benchmarkCatalogEntries" in component
    assert "function benchmarkHref" in component
    assert "`${prefix}/docs/evaluation/benchmark-hub/${entry.id}`" in component
    assert "href={benchmarkHref(entry, locale)}" in component


def test_benchmark_hub_ids_match_catalog_and_task_manifests() -> None:
    catalog_status = json.loads(BENCHMARK_CATALOG_STATUS.read_text(encoding="utf-8"))
    hub_ids = set(catalog_status)

    task_ids = {
        path.stem
        for path in (REPO_ROOT / "worldfoundry/data/benchmarks/tasks/external").glob("*.yaml")
    }
    formal_ids = set(formal_benchmark_ids())

    assert hub_ids <= task_ids
    assert hub_ids == formal_ids
    assert all((BENCHMARK_DETAILS_ROOT / f"{benchmark_id}.mdx").is_file() for benchmark_id in hub_ids)
    assert all((BENCHMARK_DETAILS_ROOT / f"{benchmark_id}.zh.mdx").is_file() for benchmark_id in hub_ids)


def test_public_docs_benchmark_flags_reference_formal_inventory() -> None:
    docs_roots = PUBLIC_DOC_ROOTS
    formal_ids = set(formal_benchmark_ids())
    flag_pattern = re.compile(r"--benchmark(?:-id)?(?:=|\s+)([A-Za-z0-9][A-Za-z0-9._-]*)")
    offenders: dict[str, list[str]] = {}

    for root in docs_roots:
        for path in _iter_text_files(root, {".md", ".mdx", ".ts", ".tsx"}):
            ids = sorted(set(flag_pattern.findall(path.read_text(encoding="utf-8"))) - formal_ids)
            if ids:
                offenders[str(path.relative_to(REPO_ROOT))] = ids

    assert offenders == {}


def test_public_benchmark_commands_use_installed_console_entrypoint() -> None:
    roots = (
        *PUBLIC_DOC_ROOTS,
        REPO_ROOT / "worldfoundry/data/benchmarks",
    )
    forbidden = (
        "python -m worldfoundry.cli",
        "python -m worldfoundry.evaluation.cli",
        "PYTHONPATH=. worldfoundry-eval",
        "worldfoundry-eval.local_open_eval",
        "- python\n- -m\n- worldfoundry.cli",
        "- python\n- -m\n- worldfoundry.evaluation.cli",
    )
    offenders: dict[str, list[str]] = {}

    for root in roots:
        for path in _iter_text_files(root, {".md", ".mdx", ".ts", ".tsx", ".yaml"}):
            text = path.read_text(encoding="utf-8")
            matches = [needle for needle in forbidden if needle in text]
            if matches:
                offenders[str(path.relative_to(REPO_ROOT))] = matches

    assert offenders == {}


def test_benchmark_hub_stats_and_cards_match_rendered_inventory() -> None:
    catalog_status = json.loads(BENCHMARK_CATALOG_STATUS.read_text(encoding="utf-8"))
    component = (FUMADOCS_ROOT / "components" / "benchmark-recipe-catalog.tsx").read_text(encoding="utf-8")
    category_counts = Counter(entry["category"] for entry in catalog_status.values())

    assert set(category_counts) == {"Embodied AI", "Video Generation", "World Models"}
    assert len(catalog_status) == sum(category_counts.values()) == len(formal_benchmark_ids())
    assert all(entry["name"] for entry in catalog_status.values())
    assert all(isinstance(entry["metrics"], list) for entry in catalog_status.values())
    assert all(entry["summary"] and entry["summaryZh"] for entry in catalog_status.values())
    assert "benchmarkCatalogEntries.length" in component
    assert "benchmarkCatalogEntries.filter((entry) => entry.category === id).length" in component
