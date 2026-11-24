# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Video-Bench text alignment judges every requested model with its own frames."""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, TypedDict

import pytest

pytestmark = pytest.mark.unit

_PACKAGE = "worldfoundry.evals.tasks.execution.runners.videobench.runtime.videobench"
_SOURCE = Path(__file__).resolve().parents[3] / (
    "worldfoundry/evals/tasks/execution/runners/videobench/runtime/videobench/VideoTextAlignment.py"
)


class _Sample(TypedDict):
    prompt: str
    frames: dict[str, list[str]]


_SAMPLES: list[_Sample] = [
    {"prompt": "a red ball", "frames": {"model-a": ["a0", "a1"], "model-b": ["b0", "b1", "b2"]}},
    {"prompt": "a blue cube", "frames": {"model-a": ["a2"], "model-b": ["b3", "b4"]}},
]
_SCORES = {
    ("model-a", "a red ball"): 4,
    ("model-b", "a red ball"): 2,
    ("model-a", "a blue cube"): 5,
    ("model-b", "a blue cube"): 3,
}


def _decorator_factory(*_args: object, **_kwargs: object) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return lambda function: function


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Remote-judge retry policies are irrelevant without API calls.
    retry_policy = SimpleNamespace(
        retry=_decorator_factory,
        wait_exponential=_decorator_factory,
        stop_after_attempt=_decorator_factory,
        retry_if_exception_type=_decorator_factory,
        wait_random_exponential=_decorator_factory,
    )
    monkeypatch.setitem(sys.modules, "tenacity", retry_policy)
    name = f"{_PACKAGE}._test_video_text_alignment"
    spec = importlib.util.spec_from_file_location(name, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _evaluate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, models: list[str] | None
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    module = _load(monkeypatch)
    agents: list[Any] = []
    hosts: list[Any] = []

    class Dataset:
        def __init__(self, path: str) -> None:
            assert path == "full_info.json"

        def __len__(self) -> int:
            return len(_SAMPLES)

        def __getitem__(self, index: int) -> _Sample:
            return _SAMPLES[index]

    class Agent:
        def __init__(self, agent: str, logger: logging.Logger, prompt: str, config: dict[str, str]) -> None:
            self.agent = agent
            self.prompt = prompt
            self.video_prompt = ""
            agents.append(self)

        def reset(self) -> None:
            self.video_prompt = ""

    class Host:
        def __init__(
            self,
            name: str,
            logger: logging.Logger,
            prompt: str,
            config: dict[str, str],
            modelname: str,
            modelmessage: str,
            agents: list[Agent],
        ) -> None:
            self.modelname = modelname
            self.modelmessage = modelmessage
            self.agents = agents
            self.video_prompt = " "
            self.frames: dict[str, list[str]] = {}
            hosts.append(self)

        def initial_result(self) -> str:
            return f"{self.modelname}: {self.video_prompt}: {len(self.frames[self.modelname])}"

        def question(self) -> list[str]:
            return [f"question for {self.modelname}"]

        def answer(self) -> list[str]:
            return [f"answer for {self.modelname}"]

        def summarize_and_get_results(self) -> str:
            score = _SCORES[self.modelname, self.video_prompt]
            return f"Updated Video Description: fine. Evaluation Result: {score} because it matches."

        def reset(self) -> None:
            self.frames = {}
            self.video_prompt = " "

    monkeypatch.setattr(module, "Video_Dataset", Dataset)
    monkeypatch.setattr(module, "Agent", Agent)
    monkeypatch.setattr(module, "Host", Host)
    logger = logging.getLogger(module.__name__)
    try:
        result = module.eval(
            {"log_path_color": str(tmp_path / "color.log")}, "prompt", "color", "full_info.json", models
        )
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
    return result, hosts, agents


def _final(model: str, prompt: str) -> str:
    return f"Updated Video Description: fine. Evaluation Result: {_SCORES[model, prompt]} because it matches."


@pytest.mark.parametrize("models", [None, ["model-a", "model-b"], ["model-b"]])
def test_each_model_is_judged_with_its_own_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, models: list[str] | None
) -> None:
    result, hosts, agents = _evaluate(monkeypatch, tmp_path, models)

    expected_models = models or ["model-a", "model-b"]
    pairs = [(sample, model) for sample in _SAMPLES for model in expected_models]
    assert [(host.modelname, host.video_prompt) for host in hosts] == [
        (model, sample["prompt"]) for sample, model in pairs
    ]
    assert [host.modelmessage for host in hosts] == [
        f"{len(sample['frames'][model])} frames from {model}." for sample, model in pairs
    ]
    assert all(host.frames is sample["frames"] for host, (sample, _model) in zip(hosts, pairs, strict=True))
    assert len(agents) == 2 * len(hosts) == len({id(agent) for agent in agents})
    assert [[agent.agent for agent in host.agents] for host in hosts] == [["Assistant-one", "Assistant-two"]] * len(
        hosts
    )
    assert [[agent.video_prompt for agent in host.agents] for host in hosts] == [
        [sample["prompt"]] * 2 for sample, _model in pairs
    ]
    assert result == {
        "history": {
            index: {
                model: {
                    "initial_response": f"{model}: {sample['prompt']}: {len(sample['frames'][model])}",
                    "qa_history": [f"question for {model}"],
                    "final_result": _final(model, sample["prompt"]),
                }
                for model in expected_models
            }
            for index, sample in enumerate(_SAMPLES)
        },
        "score": {
            index: {"prompt_en": sample["prompt"]}
            | {model: _SCORES[model, sample["prompt"]] for model in expected_models}
            for index, sample in enumerate(_SAMPLES)
        },
        "average_scores": {
            model: sum(_SCORES[model, sample["prompt"]] for sample in _SAMPLES) / len(_SAMPLES)
            for model in expected_models
        },
    }
