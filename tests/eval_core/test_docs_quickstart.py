from __future__ import annotations

import json
import subprocess
from pathlib import Path

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    formal_benchmark_ids,
    iter_benchmark_catalog_manifest_paths,
)
from worldfoundry.evaluation.utils import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_documents_discovery_and_explicit_gpu_execution() -> None:
    quickstart = (REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "quickstart.mdx").read_text(encoding="utf-8")

    assert "worldfoundry-eval zoo benchmarks" in quickstart
    assert "worldfoundry-eval zoo models" in quickstart
    assert "worldfoundry-eval run" in quickstart
    legacy_command = " ".join(("worldfoundry-eval", "validation"))
    assert legacy_command not in quickstart
    assert "worldfoundry-eval contract run" not in quickstart
    assert "--check-local" in quickstart
    assert "prepare_model_infer.sh <model-id> --download" in quickstart
    assert "bootstrap_worldfoundry.sh --verify-only" in quickstart
    assert "worldfoundry_unified_env.sh" in quickstart
    assert "--pipeline.task-profile <task-profile>" in quickstart
    assert "official-validation" in quickstart
    assert "official-run" in quickstart
    for section in ("Environment", "Assets", "TUI", "CLI", "Inference", "Evaluation"):
        assert f". {section}" in quickstart


def test_public_docs_document_model_download_and_inference_contract() -> None:
    model_manifest = load_manifest(
        REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog" / "world_models" / "matrix-game-2.yaml"
    )
    repo_id = model_manifest["checkpoint"]["repos"][0]["id"]
    revision = model_manifest["checkpoint"]["repos"][0]["sha"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "matrix-game-2" in readme
    assert repo_id in readme
    assert "worldfoundry-eval zoo model-download --model-id matrix-game-2" in readme
    assert "bash scripts/inference/prepare_model_infer.sh matrix-game-2 --download" in readme
    assert "bash scripts/inference/run_infer.sh --category navigation-video --model matrix-game-2" in readme

    inference_paths = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "guides" / "inference.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "guides" / "inference.zh.mdx",
    )
    for path in inference_paths:
        text = path.read_text(encoding="utf-8")
        assert "worldfoundry-eval run" in text
        assert "--pipeline.task-profile" in text
        assert "scripts/inference/prepare_model_infer.sh <model-id> --download" in text
        assert "zoo model-download --check-local" in text

    local_asset_paths = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "guides" / "local-assets.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "guides" / "local-assets.zh.mdx",
    )
    for path in local_asset_paths:
        text = path.read_text(encoding="utf-8")
        assert "matrix-game-2" in text
        assert repo_id in text
        assert revision in text
        assert "worldfoundry-eval zoo model-download" in text
        assert "--model-id matrix-game-2" in text
        assert "--check-local" in text
        assert "Skywork--Matrix-Game-2.0" in text


def test_fumadocs_build_pins_next_workspace_root() -> None:
    next_config = (REPO_ROOT / "docs" / "fumadocs" / "next.config.mjs").read_text(encoding="utf-8")
    build_script = (REPO_ROOT / "scripts" / "docs" / "build.sh").read_text(encoding="utf-8")

    assert "outputFileTracingRoot" in next_config
    assert "outputFileTracingRoot: __dirname" in next_config
    assert "Warning: Next.js inferred your workspace root" in build_script


def test_fumadocs_build_artifacts_are_not_tracked() -> None:
    forbidden = [
        "docs/fumadocs/node_modules",
        "docs/fumadocs/.next",
        "docs/fumadocs/out",
        "docs/fumadocs/tsconfig.tsbuildinfo",
    ]

    tracked = subprocess.check_output(["git", "ls-files", "--", *forbidden], cwd=REPO_ROOT, text=True)
    assert tracked == ""


def test_public_docs_expose_all_benchmarks_run_shortcut() -> None:
    paths = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.zh.mdx",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "worldfoundry-eval run" in text
        assert "--all-benchmarks" in text
        assert "--output-dir" in text
        assert "--plan-only" in text


def test_readme_does_not_advertise_retired_gpu_validation_commands() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "e2e-gpu-validation-suite" not in readme
    assert "models gpu-validation" not in readme
    assert "--include-local-validation-evidence" not in readme


def test_public_docs_do_not_reference_task_alias_debug_flags() -> None:
    paths = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "quickstart.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "quickstart.zh.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.zh.mdx",
        *sorted((REPO_ROOT / "docs/fumadocs/content/docs/evaluation").rglob("*.mdx")),
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "--include-aliases" not in text
        assert "raw alias" not in text.lower()


def test_cli_docs_do_not_advertise_retired_gpu_validation_commands() -> None:
    paths = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.zh.mdx",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "e2e-gpu-validation-suite" not in text
        assert "e2e-gpu-validation" not in text
        assert "models gpu-validation" not in text
        assert "--model-benchmark-gpu-validation-dir" not in text
        assert "--include-local-validation-evidence" not in text


def test_release_validation_documents_public_cpu_and_gpu_gates() -> None:
    validation = (REPO_ROOT / "docs/fumadocs/content/docs/reference/validation.mdx").read_text(encoding="utf-8")
    for command in ("make test", "make test-infer-cuda-contracts", "make test-geometry"):
        assert command in validation


def test_public_docs_use_unified_run_facade_for_model_benchmark_commands() -> None:
    docs_roots = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs" / "fumadocs" / "content" / "docs").rglob("*.mdx")),
    ]
    forbidden = (
        "worldfoundry-eval run-benchmark",
        "worldfoundry-eval run-suite",
        "run-benchmark --",
        "run-suite --",
        "run-suite as the canonical",
        "run-benchmark commands",
        "生成 `run-benchmark` 命令",
    )

    offenders: list[str] = []
    for path in docs_roots:
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")

    assert offenders == []


def test_fumadocs_navigation_does_not_reference_removed_adapter_pages() -> None:
    docs_page = REPO_ROOT / "docs" / "fumadocs" / "components" / "docs-page.tsx"
    benchmark_catalog = REPO_ROOT / "docs" / "fumadocs" / "components" / "benchmark-recipe-catalog.tsx"

    docs_text = docs_page.read_text(encoding="utf-8")
    catalog_text = benchmark_catalog.read_text(encoding="utf-8")

    assert "benchmark-adapters" not in docs_text
    assert "benchmark-hub" in docs_text
    assert "benchmarkCatalogEntries" in catalog_text
    assert "href={benchmarkHref(entry, locale)}" in catalog_text


def test_benchmark_hub_docs_match_catalog_and_readiness_claims() -> None:
    entries = {
        entry["id"]: entry
        for path in iter_benchmark_catalog_manifest_paths()
        if (entry := load_manifest(path))["id"] in formal_benchmark_ids()
    }
    status = json.loads((REPO_ROOT / "docs/fumadocs/lib/benchmark-catalog-status.json").read_text(encoding="utf-8"))
    assert set(entries) == set(status) == set(formal_benchmark_ids())
    for benchmark_id, entry in entries.items():
        assert status[benchmark_id]["leaderboardValid"] is bool(entry.get("leaderboard_valid", False))
        for suffix in (".mdx", ".zh.mdx"):
            page = REPO_ROOT / "docs/fumadocs/content/docs/evaluation/benchmark-hub" / (benchmark_id + suffix)
            assert page.is_file()


def test_public_docs_do_not_advertise_non_inventory_benchmark_ids() -> None:
    docs_roots = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs",
        REPO_ROOT / "docs" / "fumadocs" / "components",
        REPO_ROOT / "docs" / "fumadocs" / "lib",
    )
    forbidden = (
        "open-source-demo",
        "MEt3R",
        "PSIVG",
        "benchmark-adapters",
        "Legacy model-type",
        "Legacy benchmark",
        "model-benchmark commands",
        "model-benchmark run",
    )
    offenders: list[str] = []

    for root in docs_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".mdx", ".ts", ".tsx", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for snippet in forbidden:
                if snippet in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")

    assert offenders == []
