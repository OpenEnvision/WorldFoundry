from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.cli import main
from worldfoundry.evaluation.utils import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_documents_gpu_first_commands() -> None:
    quickstart = (REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "quickstart.mdx").read_text(
        encoding="utf-8"
    )

    assert "worldfoundry-eval zoo benchmarks" in quickstart
    assert "worldfoundry-eval zoo models" in quickstart
    assert "worldfoundry-eval tasks list" in quickstart
    assert "worldfoundry-eval run" in quickstart
    legacy_command = " ".join(("worldfoundry-eval", "validation"))
    assert legacy_command not in quickstart
    assert "worldfoundry-eval contract run" not in quickstart
    assert "--execute-download" in quickstart
    assert "summary.next_actions" in quickstart
    assert "summary.missing_official_env" in quickstart
    assert "worldfoundry.env.template" in quickstart
    assert "run_readiness.json" in quickstart
    assert "readiness.blocks_execution=true" in quickstart
    assert "prepare_commands.sh" in quickstart
    assert "all_benchmarks_data_download_plan.sh" in quickstart
    assert "full-suite leaderboard evidence" in quickstart


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
    assert "bash scripts/inference/test_nav_video_gen.sh matrix-game-2" in readme

    inference_paths = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "guides" / "inference.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "guides" / "inference.zh.mdx",
    )
    for path in inference_paths:
        text = path.read_text(encoding="utf-8")
        assert "matrix-game-2" in text
        assert "bash scripts/inference/test_nav_video_gen.sh matrix-game-2" in text
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
    assert "path.resolve(__dirname, '..', '..')" in next_config
    assert "Warning: Next.js inferred your workspace root" in build_script


def test_fumadocs_generated_artifacts_are_not_present_in_source_tree() -> None:
    forbidden = [
        REPO_ROOT / "docs" / "fumadocs" / "node_modules",
        REPO_ROOT / "docs" / "fumadocs" / ".next",
        REPO_ROOT / "docs" / "fumadocs" / "out",
        REPO_ROOT / "docs" / "fumadocs" / "tsconfig.tsbuildinfo",
    ]

    assert [str(path.relative_to(REPO_ROOT)) for path in forbidden if path.exists()] == []


def test_public_docs_expose_all_benchmarks_run_shortcut() -> None:
    paths = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "quickstart.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "quickstart.zh.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.zh.mdx",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "worldfoundry-eval run \\" in text
        assert "--all-benchmarks" in text
        assert "tmp/worldfoundry_all_benchmarks_plan" in text
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
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.mdx",
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.zh.mdx",
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


def test_release_checklist_gpu_gate_uses_public_cli_only() -> None:
    checklist = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "maintainers" / "release-checklist.mdx"
    ).read_text(encoding="utf-8")
    gpu_section = checklist.split("## GPU And Data Gates", 1)[1].split("## Public UX", 1)[0]

    assert "PYTHONPATH=. python scripts/" not in gpu_section


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
    benchmark_hub_data = REPO_ROOT / "docs" / "fumadocs" / "lib" / "benchmark-hub-data.ts"

    docs_text = docs_page.read_text(encoding="utf-8")
    hub_text = benchmark_hub_data.read_text(encoding="utf-8")

    assert "benchmark-adapters" not in docs_text
    assert "benchmark-runners" in docs_text
    assert "benchmark-status" in docs_text
    assert "benchmark-hub" in docs_text
    assert "adapter" not in hub_text.lower()


def test_benchmark_hub_docs_match_catalog_normalizer_surface_counts() -> None:
    video_payload = load_manifest(
        REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog" / "video_world_benchmarks.yaml"
    )
    embodied_payload = load_manifest(
        (
            REPO_ROOT
            / "src"
            / "worldfoundry"
            / "data"
            / "benchmarks"
            / "catalog"
            / "embodied_world_benchmarks.yaml"
        )
    )
    all_entries = [*video_payload["entries"], *embodied_payload["entries"]]
    normalizer_count = sum(
        1 for entry in all_entries if entry["runner_availability"]["surface"] == "official_result_normalizer"
    )
    blocked_or_contract_count = sum(
        1
        for entry in video_payload["entries"]
        if entry["runner_availability"]["surface"] in {"blocked_contract_only", "contract_only", "contract_only_blocked"}
    )
    hub_page = (REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmark-hub.mdx").read_text(
        encoding="utf-8"
    )
    hub_page_zh = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmark-hub.zh.mdx"
    ).read_text(encoding="utf-8")
    hub_data = (REPO_ROOT / "docs" / "fumadocs" / "lib" / "benchmark-hub-data.ts").read_text(encoding="utf-8")

    assert normalizer_count == 37
    assert blocked_or_contract_count == 0
    assert "npm run generate:data" not in hub_page
    assert "npm run generate:data" not in hub_page_zh
    assert "typed local data" in hub_page
    assert "typed local data" in hub_page_zh
    assert "10 rows expose official-result normalizers" in hub_data
    assert "10 official-result normalizer rows" in hub_data


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
