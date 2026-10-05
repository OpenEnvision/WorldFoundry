"""Keep infrastructure guidance aligned across the maintained docs surfaces."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_environment_and_cli_docs_cover_compile_cache_and_logging() -> None:
    for name in ("environments.mdx", "environments.zh.mdx"):
        text = _text(DOCS_ROOT / "reference" / name)
        assert "worldfoundry.core.execution.compile_cache" in text
        assert "WORLDFOUNDRY_COMPILE_CACHE_DIR" in text
        assert "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE" in text
        assert "WORLDFOUNDRY_LOG_CONTEXT" in text

    for name in ("cli.mdx", "cli.zh.mdx"):
        text = _text(DOCS_ROOT / "reference" / name)
        assert "--log-level" in text
        assert "--log-file" in text
        assert "--log-json" in text
        assert "WORLDFOUNDRY_LOG_CONTEXT" in text


def test_install_docs_keep_cpu_gpu_and_native_boundaries_explicit() -> None:
    for path in (
        REPO_ROOT / "README.md",
        DOCS_ROOT / "quickstart.mdx",
        DOCS_ROOT / "quickstart.zh.mdx",
    ):
        text = _text(path)
        assert 'pip install -e ".[tui]"' in text
        assert "bootstrap_worldfoundry.sh" in text
        assert "CPU" in text and "GPU" in text


def test_maintainer_docs_name_generated_recipe_and_cpu_eval_gates() -> None:
    for path in (
        REPO_ROOT / "CONTRIBUTING.md",
        DOCS_ROOT / "maintainers" / "contributing.mdx",
        DOCS_ROOT / "maintainers" / "contributing.zh.mdx",
    ):
        text = _text(path)
        assert ".[dev,test]" in text
        assert "test-infer" in text
        assert "test-geometry" in text
        assert "models:check" in text

    for name in ("validation.mdx", "validation.zh.mdx"):
        assert "npm run models:check" in _text(DOCS_ROOT / "reference" / name)
