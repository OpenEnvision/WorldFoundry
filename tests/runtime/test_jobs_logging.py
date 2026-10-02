from __future__ import annotations

import asyncio
import json
import sys

from worldfoundry.runtime.jobs import AsyncCommandJobStore


def test_async_job_persists_streams_events_and_child_context(tmp_path):
    async def run_job():
        store = AsyncCommandJobStore()
        job = store.submit(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['WORLDFOUNDRY_LOG_CONTEXT']); print('worker error', file=__import__('sys').stderr)",
            ],
            output_dir=tmp_path,
            metadata={"run_id": "run-job"},
        )
        assert job._task is not None
        await job._task
        return job

    job = asyncio.run(run_job())

    assert job.status == "completed"
    assert job.stdout_log_path is not None
    assert job.stderr_log_path is not None
    assert job.event_log_path is not None
    context = json.loads(job.log_text(stream="stdout").splitlines()[0])
    assert context["run_id"] == "run-job"
    assert context["job_id"] == job.job_id
    assert "worker error" in open(job.stderr_log_path, encoding="utf-8").read()
    events = [json.loads(line) for line in open(job.event_log_path, encoding="utf-8") if line.strip()]
    assert [event["event"] for event in events] == ["job.queued", "job.started", "job.finished"]
    assert all(event["run_id"] == "run-job" for event in events)
