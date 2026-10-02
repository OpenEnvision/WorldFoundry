from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".css",
    ".json",
    ".md",
    ".mdx",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
FORBIDDEN_MACHINE_PATHS = (
    "yang" + "boxue",
    "/mnt" + "/cpfs",
    "/mnt" + "/workspace",
    "/mnt" + "/world_foundational_model",
    "/Users" + "/",
    "/private" + "/home",
    "/home/" + "kchen",
    "/home/" + "gary",
)
PRUNED_DIRS = {
    ".git",
    ".next",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "logs",
    "node_modules",
    "out",
    "outputs",
    "tmp",
}
PRUNED_DIR_PREFIXES = (
    "out.nfs-busy-",
)


def _iter_open_source_text_files() -> list[Path]:
    roots = [
        REPO_ROOT,
        REPO_ROOT / "scripts",
        REPO_ROOT / "docs",
        REPO_ROOT / "tools",
        REPO_ROOT / "configs",
        REPO_ROOT / "lyra2_plan.json",
    ]
    files: set[Path] = set()
    self_path = Path(__file__).resolve()
    for root in roots:
        if root.is_file():
            files.add(root.resolve())
            continue
        if not root.exists():
            continue
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname not in PRUNED_DIRS and not dirname.startswith(PRUNED_DIR_PREFIXES)
            ]
            current_path = Path(current)
            for filename in filenames:
                path = current_path / filename
                if path.suffix in TEXT_SUFFIXES and path.resolve() != self_path:
                    files.add(path.resolve())
    return sorted(files)


def test_open_source_files_do_not_contain_host_specific_absolute_paths() -> None:
    hits: list[str] = []

    for path in _iter_open_source_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in FORBIDDEN_MACHINE_PATHS):
            hits.append(str(path.relative_to(REPO_ROOT)))

    assert hits == []
