"""CPU regressions for evaluation process pools, exception handlers, and annotation parsing."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "worldfoundry" / "evaluation" / "tasks"
METRICS_ROOT = TASKS_ROOT / "metrics"

MIND_PROCESS = TASKS_ROOT / "execution/runners/mind/runtime/mind/src/process.py"

LITERAL_EVAL_FILES = (
    TASKS_ROOT / "execution/runners/worldscore/runtime/worldscore/worldscore/benchmark/helpers/prompt_generator.py",
    TASKS_ROOT / "execution/runners/videoscore/runtime/videoscore/benchmark/get_spearman_corr.py",
    TASKS_ROOT / "execution/runners/videoscore/runtime/videoscore/benchmark/get_vbench_pairwise_acc.py",
    TASKS_ROOT / "execution/runners/videoscore/runtime/videoscore/benchmark/get_genaibench_pairwise_acc.py",
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_vendor_path(path: Path) -> bool:
    return "vendor" in path.parts


def _bare_except_handlers(tree: ast.AST) -> list[ast.ExceptHandler]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and node.type is None]


def _builtin_eval_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval"
    ]


def _is_spawn_context_pool(call: ast.Call) -> bool:
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "Pool"):
        return False
    ctx = func.value
    if not isinstance(ctx, ast.Call):
        return False
    if not (isinstance(ctx.func, ast.Attribute) and ctx.func.attr == "get_context"):
        return False
    if not ctx.args:
        return False
    arg0 = ctx.args[0]
    return isinstance(arg0, ast.Constant) and arg0.value == "spawn"


def test_mind_runner_pool_uses_spawn_context() -> None:
    tree = _parse(MIND_PROCESS)
    pool_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Pool"
    ]
    assert pool_calls, "expected at least one Pool(...) construction"
    non_spawn = [ast.dump(node.func) for node in pool_calls if not _is_spawn_context_pool(node)]
    assert non_spawn == [], f"Pool must come from mp.get_context('spawn'): {non_spawn}"


def test_self_written_metrics_have_no_bare_except() -> None:
    offenders: list[str] = []
    for path in METRICS_ROOT.rglob("*.py"):
        if _is_vendor_path(path):
            continue
        handlers = _bare_except_handlers(_parse(path))
        if handlers:
            rel = path.relative_to(REPO_ROOT)
            offenders.extend(f"{rel}:{node.lineno}" for node in handlers)
    assert offenders == [], f"bare except: still present in self-written metrics: {offenders}"


def test_annotation_and_gpt_paths_use_literal_eval_not_eval() -> None:
    for path in LITERAL_EVAL_FILES:
        tree = _parse(path)
        assert _builtin_eval_calls(tree) == [], f"builtin eval() remains in {path}"
        literal_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ast"
            and node.func.attr == "literal_eval"
        ]
        assert literal_calls, f"expected ast.literal_eval in {path}"
