from __future__ import annotations

from pathlib import Path


CANONICAL_MODEL_RUN = "conda run -n worldfoundry bash scripts/inference/test_nav_video_gen.sh matrix-game-2"
CANONICAL_EVALUATE = "conda run -n worldfoundry worldfoundry-eval evaluate"


def test_user_docs_share_model_first_run_command() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    doc_paths = [
        repo_root / "README.md",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "quickstart.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "quickstart.zh.mdx",
    ]

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        assert CANONICAL_MODEL_RUN in text, f"{path} lost the canonical model run command"
        assert "scorecard.json" in text, f"{path} should tell users where the first-run scorecard is"


def test_evaluation_docs_expose_materialized_output_evaluate_command() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    doc_paths = [
        repo_root / "README.md",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "quickstart.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "quickstart.zh.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "reference" / "cli.zh.mdx",
    ]

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        assert CANONICAL_EVALUATE in text, f"{path} lost the evaluate command"
        assert "--results-path tmp/results.jsonl" in text, f"{path} should show materialized result input"


def test_evaluation_docs_expose_task_materialize_command() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    doc_paths = [
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "index.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "index.zh.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.mdx",
        repo_root / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.zh.mdx",
    ]

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        assert "worldfoundry-eval task materialize" in text or "`task materialize`" in text
        assert "GenerationRequest" in text
