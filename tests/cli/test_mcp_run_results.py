"""MCP synchronous runs return the submitted job's result."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID
from worldfoundry.mcp.tools import MCPToolContext, run_evaluation_payload
from worldfoundry.runtime.jobs import AsyncCommandJobStore


class _EvictingJobStore:
    """Finish each submitted job and evict it before the waiter reads the store again."""

    def submit(self, command: list[str], **kwargs: Any) -> SimpleNamespace:
        job_id = kwargs["job_id"]
        return SimpleNamespace(
            job_id=job_id,
            to_result=lambda include_logs: {"run_id": job_id, "status": "succeeded", "include_logs": include_logs},
        )

    def get(self, job_id: str) -> None:
        return None


def test_sync_run_returns_the_result_of_a_job_evicted_after_completion(tmp_path: Path) -> None:
    context = MCPToolContext(output_root=tmp_path, job_store=cast(AsyncCommandJobStore, _EvictingJobStore()))

    result = asyncio.run(
        run_evaluation_payload(model=CONTRACT_FIXTURE_MODEL_ID, benchmarks=["robotwin"], wait="sync", context=context)
    )

    assert result["status"] == "succeeded"
    assert result["include_logs"] is False
