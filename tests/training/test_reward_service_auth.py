"""Fail-closed bind/auth semantics for the native reward scorer HTTP service."""

from __future__ import annotations

import pytest

from worldfoundry.training.post_training.rewards.contracts import RewardRequest
from worldfoundry.training.post_training.rewards.http import (
    REWARD_SERVICE_TOKEN_ENV,
    HTTPRewardEvaluator,
    NativeRewardService,
    RewardScorerRegistry,
    configured_reward_service_token,
    create_reward_service_app,
    require_reward_auth_token_for_host,
    serve_reward_service,
)


def _service() -> NativeRewardService:
    registry = RewardScorerRegistry()
    registry.register("length", lambda requests: [float(len(request.prompt)) for request in requests])
    return NativeRewardService(registry)


@pytest.mark.parametrize("host", ("127.0.0.1", "localhost", "::1", "[::1]"))
def test_loopback_hosts_do_not_require_a_token(monkeypatch, host: str) -> None:
    monkeypatch.delenv(REWARD_SERVICE_TOKEN_ENV, raising=False)
    assert require_reward_auth_token_for_host(host) == ""


@pytest.mark.parametrize("host", ("0.0.0.0", "::", "", "10.1.2.3", "reward.internal"))
def test_non_loopback_hosts_fail_closed_without_token(monkeypatch, host: str) -> None:
    monkeypatch.delenv(REWARD_SERVICE_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match="refuses to bind non-loopback"):
        require_reward_auth_token_for_host(host)


def test_non_loopback_hosts_return_the_configured_token(monkeypatch) -> None:
    monkeypatch.setenv(REWARD_SERVICE_TOKEN_ENV, " secret-token ")
    assert configured_reward_service_token() == "secret-token"
    assert require_reward_auth_token_for_host("0.0.0.0") == "secret-token"


def test_serve_refuses_exposed_bind_before_importing_the_web_stack(monkeypatch) -> None:
    monkeypatch.delenv(REWARD_SERVICE_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match=REWARD_SERVICE_TOKEN_ENV):
        serve_reward_service(_service(), host="0.0.0.0", port=8080)


def test_explicit_empty_token_cannot_bypass_non_loopback_fail_closed() -> None:
    with pytest.raises(SystemExit, match=REWARD_SERVICE_TOKEN_ENV):
        serve_reward_service(_service(), host="0.0.0.0", port=8080, auth_token="  ")


def test_app_requires_bearer_token_when_configured() -> None:
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")

    app = create_reward_service_app(_service(), auth_token="secret-token")
    client = testclient.TestClient(app)

    assert client.get("/health").status_code == 401
    assert client.get("/rewards", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/score", json={"requests": []}).status_code == 401

    ok = client.get("/health", headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200
    assert ok.json()["reward_ids"] == ["length"]
    assert client.get("/rewards", params={"token": "secret-token"}).status_code == 200

    scored = client.post(
        "/score",
        json={
            "requests": [
                {
                    "request_id": "request-0",
                    "rollout_id": "rollout-0",
                    "prompt": "a moving red cube",
                    "reward_ids": ["length"],
                }
            ]
        },
        headers={"Authorization": "Bearer secret-token"},
    )
    assert scored.status_code == 200
    assert scored.json()["results"][0]["values"]["length"] == float(len("a moving red cube"))


def test_app_stays_open_without_a_token() -> None:
    testclient = pytest.importorskip("fastapi.testclient")

    client = testclient.TestClient(create_reward_service_app(_service()))
    assert client.get("/health").status_code == 200


class _RecordingSession:
    """Stub requests.Session capturing the headers each call sends."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, timeout, headers=None):  # noqa: ANN001
        self.calls.append((url, headers))
        return _OkResponse({"status": "ok", "reward_ids": []})

    def post(self, url, json, timeout, headers=None):  # noqa: ANN001
        self.calls.append((url, headers))
        results = [
            {
                "request_id": item["request_id"],
                "rollout_id": item["rollout_id"],
                "values": {"length": 1.0},
                "valid": {"length": True},
                "latency_ms": 0.5,
            }
            for item in json["requests"]
        ]
        return _OkResponse({"results": results})

    def close(self) -> None:
        pass


class _OkResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_client_sends_bearer_header_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(REWARD_SERVICE_TOKEN_ENV, "secret-token")
    session = _RecordingSession()
    evaluator = HTTPRewardEvaluator("http://10.1.2.3:8080", session=session)

    evaluator.health()
    evaluator.evaluate(
        (
            RewardRequest(
                request_id="request-0",
                rollout_id="rollout-0",
                prompt="a moving red cube",
                conditions={},
                artifacts={},
                reward_ids=("length",),
            ),
        )
    )

    expected_headers = {"Authorization": "Bearer secret-token"}
    assert session.calls == [
        ("http://10.1.2.3:8080/health", expected_headers),
        ("http://10.1.2.3:8080/score", expected_headers),
    ]


def test_client_explicit_token_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv(REWARD_SERVICE_TOKEN_ENV, "env-token")
    session = _RecordingSession()
    evaluator = HTTPRewardEvaluator(
        "http://127.0.0.1:8080",
        session=session,
        auth_token="explicit-token",
    )

    evaluator.health()

    assert session.calls[0][1] == {"Authorization": "Bearer explicit-token"}


def test_client_sends_no_auth_header_by_default(monkeypatch) -> None:
    monkeypatch.delenv(REWARD_SERVICE_TOKEN_ENV, raising=False)
    session = _RecordingSession()
    evaluator = HTTPRewardEvaluator("http://127.0.0.1:8080", session=session)

    evaluator.health()

    assert session.calls == [("http://127.0.0.1:8080/health", None)]
