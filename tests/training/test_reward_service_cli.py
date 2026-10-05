from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from worldfoundry.cli.training import register_training_subparser
from worldfoundry.training.post_training.rewards.http import NativeRewardService
from worldfoundry.training.post_training.rewards.scorers import (
    AgenticCorrectnessScorer,
    AgenticToolSuccessScorer,
    CLAPScorer,
    ScorerServiceConfig,
    VideoPickScoreScorer,
    build_configured_reward_scorer_registry,
    load_scorer_service_config,
)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {"server": {}, "scorers": {"correctness": {}}, "serve": True},
            "unsupported reward service fields",
        ),
        (
            {"server": {"failfast": True}, "scorers": {"correctness": {}}},
            "unsupported reward server fields",
        ),
    ),
)
def test_reward_service_config_rejects_unknown_fields(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ScorerServiceConfig.from_mapping(payload)


def test_reward_service_example_builds_two_lazy_scorers() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_scorer_service_config(root / "configs/post_training/reward_service_av.yaml")

    registry = build_configured_reward_scorer_registry(config)

    assert config.scorer_names == ("videopickscore", "clap")
    assert registry.names == ("clap", "videopickscore")
    assert isinstance(registry.scorer("videopickscore"), VideoPickScoreScorer)
    assert isinstance(registry.scorer("clap"), CLAPScorer)
    assert registry.scorer("videopickscore").loaded is False
    assert registry.scorer("clap").loaded is False


def test_agentic_reward_service_example_matches_qwen_reward_ids() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_scorer_service_config(root / "configs/post_training/reward_service_agentic.yaml")

    registry = build_configured_reward_scorer_registry(config)

    assert config.scorer_names == ("correctness", "tool-success")
    assert registry.names == ("correctness", "tool-success")
    correctness = registry.scorer("correctness")
    tool_success = registry.scorer("tool-success")
    assert isinstance(correctness, AgenticCorrectnessScorer)
    assert correctness.config.expected_answer_condition == "answer"
    assert isinstance(tool_success, AgenticToolSuccessScorer)
    assert tool_success.config.required_tool == "calculator"


def test_reward_service_config_supports_one_scorer(tmp_path: Path) -> None:
    path = tmp_path / "video-reward.json"
    path.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 8090, "fail_fast": False},
                "scorers": {"videopickscore": {"batch_size": 12, "device": "cpu"}},
            }
        ),
        encoding="utf-8",
    )

    config = load_scorer_service_config(path)
    registry = build_configured_reward_scorer_registry(config)

    assert config.scorer_names == ("videopickscore",)
    assert config.fail_fast is False
    assert registry.names == ("videopickscore",)
    scorer = registry.scorer("videopickscore")
    assert isinstance(scorer, VideoPickScoreScorer)
    assert scorer.config.batch_size == 12
    assert scorer.config.device == "cpu"


def test_reward_service_config_can_mix_media_and_agentic_scorers(tmp_path: Path) -> None:
    path = tmp_path / "mixed-rewards.yaml"
    path.write_text(
        "server:\n"
        "  host: 127.0.0.1\n"
        "  port: 8090\n"
        "scorers:\n"
        "  videopickscore:\n"
        "    device: cpu\n"
        "  correctness:\n"
        "    expected_answer_condition: expected_answer\n"
        "  tool-success:\n"
        "    required_tool_condition: required_tool\n",
        encoding="utf-8",
    )

    config = load_scorer_service_config(path)
    registry = build_configured_reward_scorer_registry(config)

    assert config.scorer_names == ("videopickscore", "correctness", "tool-success")
    assert registry.names == ("correctness", "tool-success", "videopickscore")
    video = registry.scorer("videopickscore")
    assert isinstance(video, VideoPickScoreScorer)
    assert video.loaded is False
    correctness = registry.scorer("correctness")
    assert isinstance(correctness, AgenticCorrectnessScorer)
    assert correctness.config.expected_answer_condition == "expected_answer"
    tool_success = registry.scorer("tool-success")
    assert isinstance(tool_success, AgenticToolSuccessScorer)
    assert tool_success.config.required_tool_condition == "required_tool"


def test_train_reward_service_cli_applies_runtime_overrides(monkeypatch, tmp_path: Path) -> None:
    import worldfoundry.training.post_training.rewards.http as http_package

    path = tmp_path / "rewards.yaml"
    path.write_text(
        "server:\n"
        "  host: 0.0.0.0\n"
        "  port: 8080\n"
        "  fail_fast: true\n"
        "scorers:\n"
        "  clap:\n"
        "    batch_size: 4\n"
        "    device: cuda\n",
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "train-reward-service",
            "--config",
            str(path),
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
            "--device",
            "cpu",
            "--no-fail-fast",
        ]
    )
    call: dict[str, object] = {}

    def serve(service: NativeRewardService, *, host: str, port: int) -> None:
        call.update(service=service, host=host, port=port)

    monkeypatch.setattr(http_package, "serve_reward_service", serve)

    assert args.func(args) == 0
    service = call["service"]
    assert isinstance(service, NativeRewardService)
    assert service.fail_fast is False
    assert service.registry.names == ("clap",)
    scorer = service.registry.scorer("clap")
    assert isinstance(scorer, CLAPScorer)
    assert scorer.config.device == "cpu"
    assert scorer.loaded is False
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 9090


def test_train_reward_service_cli_starts_agentic_scorers(monkeypatch) -> None:
    import worldfoundry.training.post_training.rewards.http as http_package

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    register_training_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "train-reward-service",
            "--config",
            str(root / "configs/post_training/reward_service_agentic.yaml"),
        ]
    )
    call: dict[str, object] = {}

    def serve(service: NativeRewardService, *, host: str, port: int) -> None:
        call.update(service=service, host=host, port=port)

    monkeypatch.setattr(http_package, "serve_reward_service", serve)

    assert args.func(args) == 0
    service = call["service"]
    assert isinstance(service, NativeRewardService)
    assert service.registry.names == ("correctness", "tool-success")
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8080


def test_reward_service_config_defaults_to_loopback_host() -> None:
    config = ScorerServiceConfig.from_mapping({"scorers": {"correctness": {}}})
    assert config.host == "127.0.0.1"
    assert config.port == 8080
