# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""VMBench CAS selects the prediction writer paired with its result merger."""

from __future__ import annotations

import ast
import json
import os
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from scipy.special import softmax

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3] / "worldfoundry/evals/tasks/execution/runners/vmbench/runtime/official"


def _prediction_row(source: Path, scores: list[float]) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    writer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "final_test")
    expression = next(
        node.value
        for node in ast.walk(writer)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "string" for target in node.targets)
    )
    output = torch.tensor([scores])
    namespace = {
        "ids": ["sample.mp4"],
        "i": 0,
        "output": output,
        "probabilities": torch.softmax(output, dim=1),
        "target": torch.tensor([4]),
        "chunk_nb": torch.tensor([0]),
        "split_nb": torch.tensor([0]),
    }
    return eval(compile(ast.Expression(expression), str(source), "eval"), namespace)


@pytest.fixture
def cas_entrypoint(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Any]:
    cas_source = _ROOT / "bench_utils/cas_utils.py"
    cas_module = ModuleType("bench_utils.cas_utils")
    cas_module.__dict__.update(
        os=os,
        json=json,
        np=np,
        softmax=softmax,
        Pool=lambda workers: SimpleNamespace(map=lambda function, rows: list(map(function, rows))),
        final_test=partial(_prediction_row, cas_source),
    )
    merger_nodes: list[ast.stmt] = [
        node
        for node in ast.parse(cas_source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in {"final_merge", "compute_video"}
    ]
    exec(compile(ast.Module(merger_nodes, type_ignores=[]), str(cas_source), "exec"), cas_module.__dict__)
    engine_module = ModuleType("engine_for_finetuning")
    engine_module.__dict__.update(
        final_test=partial(_prediction_row, _ROOT / "VideoMAEv2/engine_for_finetuning.py"),
        merge=object(),
        train_one_epoch=object(),
        validation_one_epoch=object(),
    )
    monkeypatch.setitem(sys.modules, "bench_utils", ModuleType("bench_utils"))
    monkeypatch.setitem(sys.modules, "bench_utils.cas_utils", cas_module)
    monkeypatch.setitem(sys.modules, "engine_for_finetuning", engine_module)
    entrypoint = _ROOT / "commonsense_adherence_score.py"
    imports: list[ast.stmt] = [
        node
        for node in ast.parse(entrypoint.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ImportFrom) and node.module in {"bench_utils.cas_utils", "engine_for_finetuning"}
    ]
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(imports, type_ignores=[]), str(entrypoint), "exec"), namespace)
    return namespace, cas_module.__dict__["final_test"]


def test_cas_entrypoint_selects_its_dedicated_writer(cas_entrypoint: tuple[dict[str, Any], Any]) -> None:
    namespace, writer = cas_entrypoint
    assert namespace["final_test"] is writer


@pytest.mark.parametrize("method", ["prob", "score"])
@pytest.mark.parametrize("scores", [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, -1.0, 0.5, 0.0, -2.0]])
def test_cas_prediction_rows_feed_the_official_merger(
    cas_entrypoint: tuple[dict[str, Any], Any], tmp_path: Path, method: str, scores: list[float]
) -> None:
    namespace, _ = cas_entrypoint
    row = namespace["final_test"](scores)
    (tmp_path / "0.txt").write_text("accuracy header\n" + row, encoding="utf-8")
    metadata = [{"filepath": "sample.mp4", "prompt": "a running person"}, {"filepath": "other.mp4"}]
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = namespace["final_merge"](str(tmp_path), 1, str(metadata_path), method=method)

    features = softmax(scores) if method == "prob" else np.asarray(scores)
    expected_score = np.sum(features * np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert result == [dict(metadata[0], commonsense_adherence_score=expected_score), metadata[1]]
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
