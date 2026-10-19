"""Exercise WorldScore's batch scheduling without importing its model dependencies."""

from __future__ import annotations

import ast
import json
import os
import weakref
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from worldfoundry.evaluation.tasks.execution.runners.worldscore import run_worldscore_official_runner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("movement", ["static", "dynamic"])
def test_worldscore_keeps_one_metric_resident_and_preserves_sample_inputs(tmp_path: Path, movement: str) -> None:
    source = (
        Path(run_worldscore_official_runner.__file__).parent
        / "runtime/worldscore/worldscore/benchmark/helpers/evaluator.py"
    )
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "process_batch"
    )
    live = weakref.WeakSet()
    constructions = []
    calls = []

    def nested():
        return defaultdict(nested)

    class Metric:
        def __init__(self, name: str) -> None:
            self.name = name
            live.add(self)

    def metric_factory(name, *args):
        assert not live
        constructions.append(name)
        return Metric(name)

    class Evaluator:
        def __init__(self, config):
            self.config = config
            self.generate_type = "t2v"
            self.dataset_root_path = str(tmp_path)
            self.metrics_results = nested()

        def calculate_metrics(self, **inputs):
            calls.append((self.aspect_evaluator.name, inputs))
            return inputs["total_frames"], inputs["total_frames"] / 100

    namespace = {
        "Evaluator": Evaluator,
        "logger": SimpleNamespace(info=Mock()),
        "json": json,
        "os": os,
        "layout_info": {"across": {"layout_type": "inter"}},
        "aspect_info": {"quality": {"type": "base", "metrics": {"a": {}, "b": {}}}},
        "get_aspect_evaluator": metric_factory,
        "empty_cache": Mock(),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    batch = []
    for index in range(3):
        directory = tmp_path / str(index)
        (directory / "frames").mkdir(parents=True)
        for frame in ("2.png", "1.jpg"):
            (directory / "frames" / frame).touch()
        (directory / "camera_data.json").write_text(json.dumps({"camera": index}))
        (directory / "image_data.json").write_text(
            json.dumps(
                {
                    "num_scenes": 2,
                    "total_frames": index + 10,
                    "camera_path": ["across"],
                    "anchor_frame_idx": [0, 1],
                    "prompt_list": [f"p{index}"],
                    "content_list": [f"c{index}"],
                    "prompt": f"p{index}",
                    "masks": [f"m{index}.png"],
                    "objects": [f"o{index}"],
                }
            )
        )
        keys = ("style", "scene", "category", str(index)) if movement == "static" else ("style", "motion", str(index))
        batch.append((*keys, directory))
    results = namespace["process_batch"]({"model": "fixture"}, batch, ["quality"], movement)
    assert constructions == ["a", "b"]
    assert len(calls) == 6
    assert not live
    assert namespace["empty_cache"].call_count == 2
    for name, inputs in calls:
        index = inputs["total_frames"] - 10
        assert [Path(path).name for path in inputs["image_list"]] == ["1.jpg", "2.png"]
        assert inputs["text_prompt_list"] == [f"p{index}"]
        assert inputs["camera_data"] == ({"camera": index} if movement == "static" else None)
        assert inputs["masked_images_all"] == (None if movement == "static" else [str(tmp_path / f"m{index}.png")])
        node = results
        for key in batch[index][:-1]:
            node = node[key]
        expected = {"score": index + 10, "score_normalized": (index + 10) / 100}
        assert node["quality"][name] == expected
        assert json.loads((tmp_path / str(index) / "evaluation.json").read_text())["quality"][name] == expected
