# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU argument-validation contracts for the Causal-rCM inference entrypoint."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import FunctionType

import pytest

from worldfoundry.core.attention.cache.kv_cache_policy import CachedBlock, CacheState, SlidingWindowPolicy

pytestmark = pytest.mark.unit

INFERENCE_SOURCE = (
    Path(__file__).parents[2]
    / "worldfoundry/synthesis/visual_generation/rcm/rcm_runtime/inference/wan2pt1_t2v_causal_infer.py"
)


@pytest.fixture
def parse_arguments() -> Callable[[], argparse.Namespace]:
    # Execute the source parser, its constants and its own policy import without
    # importing the unrelated CUDA/model stack or running the __main__ block.
    tree = ast.parse(INFERENCE_SOURCE.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if (
            (isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "worldfoundry.core.attention.cache.kv_cache_policy")
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"_DEFAULT_PROMPT", "_DEFAULT_NEGATIVE_PROMPT"}
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "parse_arguments")
        )
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(INFERENCE_SOURCE), "exec"), namespace)
    return namespace["parse_arguments"]


@pytest.mark.parametrize("policy_args", [[], ["--kv_cache_policy", "keep_all"]], ids=["default", "explicit"])
def test_parse_arguments_keep_all_defaults(
    parse_arguments: Callable[[], argparse.Namespace], monkeypatch: pytest.MonkeyPatch, policy_args: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["causal-infer", "--dit_path", "missing.pt", *policy_args])

    args = parse_arguments()

    assert args.kv_cache_policy == "keep_all"
    assert (args.kv_cache_window_blocks, args.kv_cache_sink_blocks) == (6, 0)
    assert args.dit_path == "missing.pt"
    assert args.model_size == "1.3B"
    assert args.distilled is False
    assert (args.num_steps, args.first_chunk_t, args.chunk_t) == (50, 1, 1)


def test_parse_arguments_keep_all_does_not_validate_unused_budget(
    parse_arguments: Callable[[], argparse.Namespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["causal-infer", "--dit_path", "missing.pt", "--kv_cache_window_blocks", "0"])

    assert parse_arguments().kv_cache_policy == "keep_all"


@pytest.mark.parametrize("window,sinks,kept", [(6, 0, (2, 3, 4, 5, 6, 7)), (4, 1, (0, 5, 6, 7)), (1, 0, (7,))])
def test_parse_arguments_sliding_window_uses_canonical_budget(
    parse_arguments: Callable[[], argparse.Namespace],
    monkeypatch: pytest.MonkeyPatch,
    window: int,
    sinks: int,
    kept: tuple[int, ...],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "causal-infer",
            "--dit_path",
            "missing.pt",
            "--kv_cache_policy",
            "sliding_window",
            "--kv_cache_window_blocks",
            str(window),
            "--kv_cache_sink_blocks",
            str(sinks),
        ],
    )

    args = parse_arguments()

    assert args.kv_cache_policy == "sliding_window"
    assert (args.kv_cache_window_blocks, args.kv_cache_sink_blocks) == (window, sinks)
    assert isinstance(parse_arguments, FunctionType)
    assert parse_arguments.__globals__["SlidingWindowPolicy"] is SlidingWindowPolicy
    policy = SlidingWindowPolicy(window_blocks=args.kv_cache_window_blocks, sink_blocks=args.kv_cache_sink_blocks)
    state = CacheState(tuple(CachedBlock(index, index, 1) for index in range(8)), frame_tokens=1, current_block_idx=7)
    decision = policy.decide(state)
    assert decision.kept_blocks == kept
    assert decision.keep_indices is not None
    assert decision.keep_indices.tolist() == list(kept)
    assert decision.frame_positions is None


@pytest.mark.parametrize(
    "window,sinks,message",
    [
        (0, 0, "window_blocks must be >= 1"),
        (-1, 0, "window_blocks must be >= 1"),
        (6, -1, "sink_blocks must be non-negative"),
        (6, 6, r"sink_blocks \(6\) must be < window_blocks \(6\)"),
        (6, 7, r"sink_blocks \(7\) must be < window_blocks \(6\)"),
    ],
)
def test_parse_arguments_rejects_invalid_sliding_window_before_model_loading(
    parse_arguments: Callable[[], argparse.Namespace],
    monkeypatch: pytest.MonkeyPatch,
    window: int,
    sinks: int,
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "causal-infer",
            "--dit_path",
            "missing.pt",
            "--kv_cache_policy",
            "sliding_window",
            "--kv_cache_window_blocks",
            str(window),
            "--kv_cache_sink_blocks",
            str(sinks),
        ],
    )

    with pytest.raises(ValueError, match=message):
        parse_arguments()
