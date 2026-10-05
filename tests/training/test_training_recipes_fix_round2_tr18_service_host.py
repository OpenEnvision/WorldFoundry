"""TR-18 adjacent: omitted reward-service host defaults to loopback."""

from __future__ import annotations

from worldfoundry.training.post_training.rewards.scorers import ScorerServiceConfig


def test_omitted_server_host_defaults_to_loopback() -> None:
    config = ScorerServiceConfig.from_mapping({"scorers": {"correctness": {}}})
    assert config.host == "127.0.0.1"
    assert config.port == 8080


def test_explicit_all_interfaces_host_is_preserved() -> None:
    config = ScorerServiceConfig.from_mapping(
        {
            "server": {"host": "0.0.0.0", "port": 9090},
            "scorers": {"correctness": {}},
        }
    )
    assert config.host == "0.0.0.0"
    assert config.port == 9090
