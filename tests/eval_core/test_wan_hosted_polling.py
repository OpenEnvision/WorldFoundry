from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from worldfoundry.pipelines.wan._hosted_polling import poll_wan_task_status


def test_wan_polling_returns_success_and_sleeps_only_between_attempts() -> None:
    responses = iter(
        [
            {"output": {"task_status": "pending"}},
            {"output": {"task_status": "SUCCEEDED", "video_url": "https://example.test/v.mp4"}},
        ]
    )
    sleeps: list[float] = []

    result = poll_wan_task_status(
        get_task=lambda _task_id: next(responses),
        extract_status=lambda payload: payload["output"]["task_status"],
        task_id="task-1",
        service_label="Wan2.6",
        logger=logging.getLogger(__name__),
        poll_interval=0.25,
        max_retries=3,
        sleeper=sleeps.append,
    )

    assert result["output"]["video_url"].endswith("v.mp4")
    assert sleeps == [0.25]


@pytest.mark.parametrize("status", ["FAILED", "CANCELED", "UNKNOWN"])
def test_wan_polling_returns_service_failures_without_retrying(status: str) -> None:
    calls: list[str] = []
    payload = {"output": {"task_status": status}}

    result = poll_wan_task_status(
        get_task=lambda task_id: calls.append(task_id) or payload,
        extract_status=lambda response: response["output"]["task_status"],
        task_id="task-2",
        service_label="Wan2.7",
        logger=logging.getLogger(__name__),
        poll_interval=1,
        max_retries=3,
        sleeper=lambda _delay: None,
    )

    assert result is payload
    assert calls == ["task-2"]


def test_wan_polling_timeout_is_bounded_by_max_retries() -> None:
    sleeps: list[float] = []

    with pytest.raises(TimeoutError, match=r"Wan2\.7 task task-3 polling timed out"):
        poll_wan_task_status(
            get_task=lambda _task_id: {"output": {"task_status": "RUNNING"}},
            extract_status=lambda response: response["output"]["task_status"],
            task_id="task-3",
            service_label="Wan2.7",
            logger=logging.getLogger(__name__),
            poll_interval=2,
            max_retries=2,
            sleeper=sleeps.append,
        )

    assert sleeps == [2]


@pytest.mark.parametrize(
    ("filename", "service_label"),
    [
        ("pipeline_wan_2p6.py", "Wan2.6"),
        ("pipeline_wan_2p7.py", "Wan2.7"),
    ],
)
def test_wan_pipeline_wrappers_delegate_to_shared_poller(
    filename: str,
    service_label: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "worldfoundry" / "pipelines" / "wan" / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    poll_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_poll_task_status"
    )
    calls = [
        node
        for node in ast.walk(poll_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "poll_wan_task_status"
    ]

    assert len(calls) == 1
    keyword_values = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.literal_eval(keyword_values["service_label"]) == service_label
    assert "time.sleep" not in source
