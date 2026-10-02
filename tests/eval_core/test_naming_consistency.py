from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


SCAN_TARGETS = (
    REPO_ROOT / "worldfoundry" / "data" / "benchmarks",
    REPO_ROOT / "worldfoundry" / "data" / "models",
    REPO_ROOT / "worldfoundry" / "data" / "test_cases",
    REPO_ROOT / "docs" / "fumadocs" / "content",
    REPO_ROOT / "scripts" / "benchmark_zoo",
    REPO_ROOT / "scripts" / "inference",
    REPO_ROOT / "worldfoundry" / "evaluation",
    REPO_ROOT / "worldfoundry" / "model_zoo",
    REPO_ROOT / "worldfoundry" / "evaluation" / "models" / "runtime" / "profiles.py",
    REPO_ROOT / "test" / "eval_core",
    REPO_ROOT / "tools" / "vibe_code",
)

RETIRED_EVALUATION_PATH_TOKENS = (
    "adapter",
    "legacy",
    "yaml_v2",
)
RETIRED_EVALUATION_DIR_NAMES = {
    "reasoning",
}

SKIP_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "bench_results",
    "hfd_datasets",
    "neoverse_runtime",
    "fantasy_world_model",
}

BANNED_TOKENS = (
    "WorldEval" + "-X",
    "worldeval" + "x",
    "worldeval" + "-x",
    "opra" + "tion",
    "me" + "meory",
    "voa" + "ger",
    "CPU" + "-only",
    "no" + "-GPU",
    "without " + "GPU " + "inference",
    "worldfoundry:" + "smoke",
    "video" + "-smoke",
    "contract" + "-smoke",
    "contract" + "_smoke",
    "official" + "-smoke",
    "official_benchmark_" + "smoke",
    "smoke" + "_command",
    "pending_" + "smoke",
    "runtime-" + "smoke",
    "runtime_" + "smoke",
    "checkpoint_" + "smoke",
    "compatibility_" + "smoke",
    "allow_compatibility_" + "smoke",
    "official_demo_" + "validated",
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for target in SCAN_TARGETS:
        if target.is_file():
            files.append(target)
            continue
        for path in target.rglob("*"):
            if not path.is_file():
                continue
            if path == Path(__file__):
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".npz"}:
                continue
            files.append(path)
    return files


def test_project_owned_surfaces_do_not_reintroduce_retired_names() -> None:
    offenders: list[str] = []
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in BANNED_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")

    assert offenders == []


def test_evaluation_tree_does_not_reintroduce_retired_compat_paths() -> None:
    roots = (
        REPO_ROOT / "worldfoundry" / "evaluation",
        REPO_ROOT / "scripts" / "benchmark_zoo",
        REPO_ROOT / "test" / "eval_core",
    )
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            rel = path.relative_to(REPO_ROOT)
            rel_text = rel.as_posix().lower()
            if any(token in rel_text for token in RETIRED_EVALUATION_PATH_TOKENS):
                offenders.append(rel.as_posix())
                continue
            if any(part.lower() in RETIRED_EVALUATION_DIR_NAMES for part in rel.parts):
                offenders.append(rel.as_posix())

    assert offenders == []


def test_evaluation_modules_use_shared_io_helpers_directly() -> None:
    evaluation_root = REPO_ROOT / "worldfoundry" / "evaluation"
    banned_snippets = (
        "def _write_json",
        "def _write_jsonl",
        "def _write_text",
        "def _read_json_or_jsonl",
        "_jsonable =",
        "_read_json_or_jsonl =",
        " as _write_json",
        " as _write_jsonl",
        " as _write_text",
        " as _append_jsonl",
        " as _reset_jsonl",
        " as _jsonable",
    )
    offenders: list[str] = []
    for path in evaluation_root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in banned_snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")

    assert offenders == []
