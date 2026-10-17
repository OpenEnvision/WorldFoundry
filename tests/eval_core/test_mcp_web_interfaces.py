from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID
from worldfoundry.mcp.client import MCPClient, convert_result_to_openai_format
from worldfoundry.mcp.tools import (
    MCPToolContext,
    cancel_run_payload,
    check_benchmark_datasets_payload,
    get_run_result_payload,
    get_run_samples_payload,
    get_run_status_payload,
    list_benchmarks_payload,
    list_runs_payload,
    list_tasks_payload,
    preview_run_payload,
    run_evaluation_payload,
    server_info_payload,
)
from worldfoundry.runtime.jobs import AsyncCommandJobStore, python_module_command


def test_mcp_server_info_documentation_routes_exist(tmp_path: Path) -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs" / "fumadocs" / "content" / "docs"
    context = MCPToolContext(output_root=tmp_path, job_store=AsyncCommandJobStore())

    for route in server_info_payload(context=context)["docs"].values():
        relative = route.removeprefix("/docs/")
        assert route.startswith("/docs/")
        assert (docs_root / f"{relative}.mdx").is_file() or (docs_root / relative / "index.mdx").is_file()


def test_python_module_command_rewrites_console_entrypoint() -> None:
    command = python_module_command(("worldfoundry-eval", "run", "--json"), python_executable="/usr/bin/python")

    assert command == ("/usr/bin/python", "-m", "worldfoundry.evaluation", "run", "--json")


def test_command_job_store_captures_json_result() -> None:
    async def run() -> None:
        store = AsyncCommandJobStore()
        job = store.submit(
            [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'ok': True, 'value': 7}))",
            ]
        )
        while not job.terminal:
            await asyncio.sleep(0.05)

        assert job.status == "completed"
        assert job.result == {"ok": True, "value": 7}

    asyncio.run(run())


def test_mcp_discovery_and_preview_are_available_without_mcp_dependency(tmp_path) -> None:
    context = MCPToolContext(output_root=tmp_path, job_store=AsyncCommandJobStore())

    benchmarks = list_benchmarks_payload(query="robotwin", context=context)
    preview = preview_run_payload(
        model=CONTRACT_FIXTURE_MODEL_ID,
        benchmark="robotwin",
        output_dir=tmp_path / "run",
        prepare=True,
        data_root=tmp_path / "datasets",
        generation_cache_dir=tmp_path / "generation-cache",
        generation_cache_mode="read-write",
        context=context,
    )

    assert benchmarks["total"] >= 1
    assert preview["command"][:4] == [sys.executable, "-m", "worldfoundry.cli", "run"]
    assert preview["run_command"][:4] == [sys.executable, "-m", "worldfoundry.cli", "run"]
    assert "--json" in preview["command"]
    assert "--prepare" in preview["command"]
    assert preview["command"][preview["command"].index("--data-root") + 1] == str(tmp_path / "datasets")
    assert preview["command"][preview["command"].index("--generation-cache-dir") + 1] == str(
        tmp_path / "generation-cache"
    )
    assert preview["command"][preview["command"].index("--generation-cache-mode") + 1] == "read-write"
    assert context.job_store is not None
    assert context.job_store.list() == []


def test_mcp_default_run_output_is_scoped_by_job_id(tmp_path) -> None:
    class _CapturingJobStore:
        def __init__(self) -> None:
            self.submission = None

        def submit(self, command, **kwargs):
            self.submission = {"command": command, **kwargs}
            job_id = kwargs["job_id"]
            output_dir = str(kwargs["output_dir"])
            return SimpleNamespace(
                job_id=job_id,
                to_summary=lambda: {
                    "job_id": job_id,
                    "status": "running",
                    "output_dir": output_dir,
                    "command": list(command),
                },
            )

    async def run() -> None:
        store = _CapturingJobStore()
        context = MCPToolContext(output_root=tmp_path, job_store=store)

        submitted = await run_evaluation_payload(
            model=CONTRACT_FIXTURE_MODEL_ID,
            benchmark="robotwin",
            wait="async",
            context=context,
        )

        assert store.submission is not None
        assert store.submission["job_id"] == submitted["run_id"]
        assert Path(submitted["output_dir"]).parts[-2:] == (
            f"{CONTRACT_FIXTURE_MODEL_ID}__robotwin",
            submitted["run_id"],
        )

    asyncio.run(run())


def test_mcp_client_exports_openai_function_list_without_server() -> None:
    tools = [
        SimpleNamespace(
            name="preview_run",
            description="Preview a WorldFoundry run.",
            inputSchema={
                "type": "object",
                "properties": {"model": {"type": "string"}, "benchmark": {"type": "string"}},
                "required": ["model", "benchmark"],
            },
        )
    ]

    functions = MCPClient.tools_to_function_list(tools)

    assert functions == [
        {
            "type": "function",
            "function": {
                "name": "preview_run",
                "description": "Preview a WorldFoundry run.",
                "parameters": tools[0].inputSchema,
            },
        }
    ]


def test_mcp_client_sync_function_list_wrapper_uses_tool_metadata() -> None:
    class FakeClient(MCPClient):
        async def list_tools(self):
            return [SimpleNamespace(name="list_models", description=None, inputSchema={"type": "object"})]

    assert FakeClient().get_function_list_sync() == [
        {
            "type": "function",
            "function": {
                "name": "list_models",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_mcp_client_exposes_canonical_tool_call_api_only() -> None:
    client = MCPClient()

    assert hasattr(client, "call_tool")
    assert hasattr(client, "call_tool_sync")
    assert not hasattr(client, "run_tool")
    assert not hasattr(client, "run_tool_sync")


def test_mcp_package_exports_client_for_discovery() -> None:
    import worldfoundry.mcp as mcp_package

    assert "MCPClient" in mcp_package.__all__
    assert mcp_package.MCPClient is MCPClient


def test_mcp_list_tasks_accepts_glob_query() -> None:
    payload = list_tasks_payload(query="*robot*")

    assert payload["total"] >= 1
    assert any("robot" in json.dumps(item).casefold() for item in payload["tasks"])


def test_mcp_get_run_samples_paginates_results(tmp_path) -> None:
    async def run() -> None:
        output_dir = tmp_path / "sample-run"
        output_dir.mkdir()
        (output_dir / "results.jsonl").write_text(
            "\n".join(json.dumps({"doc_id": index, "input": f"prompt-{index}"}) for index in range(3)) + "\n",
            encoding="utf-8",
        )
        store = AsyncCommandJobStore()
        context = MCPToolContext(output_root=tmp_path, job_store=store)
        job = store.submit(
            [sys.executable, "-c", "print('{\"ok\": true}')"],
            output_dir=output_dir,
            job_id="sample-run",
        )
        while not job.terminal:
            await asyncio.sleep(0.05)

        payload = get_run_samples_payload("sample-run", offset=1, limit=1, context=context)

        assert payload["total"] == 3
        assert payload["offset"] == 1
        assert payload["limit"] == 1
        assert payload["source_path"] == "results.jsonl"
        assert payload["samples"] == [{"doc_id": 1, "input": "prompt-1"}]

    asyncio.run(run())


def test_mcp_client_sync_wrapper_rejects_active_event_loop() -> None:
    async def run() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            MCPClient().get_function_list_sync()

    asyncio.run(run())


def test_mcp_client_converts_tool_results_to_openai_content_blocks() -> None:
    result = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="ready"),
            SimpleNamespace(type="image", data="abcd", mimeType="image/jpeg"),
            {"type": "audio", "data": "efgh", "mimeType": "audio/wav"},
        ]
    )

    assert convert_result_to_openai_format(result) == [
        {"type": "text", "text": "ready"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abcd"}},
        {"type": "input_audio", "input_audio": {"data": "efgh", "format": "wav"}},
    ]


def test_mcp_benchmark_dataset_check_reports_missing_data(tmp_path) -> None:
    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "demo-benchmark",
                        "aliases": ["DemoBench"],
                        "source": {"status": "open_source"},
                        "official_sources": {"huggingface_datasets": [{"repo_id": "org/data", "license": "mit"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    context = MCPToolContext(
        output_root=tmp_path,
        benchmark_manifest_dir=manifest_dir,
        job_store=AsyncCommandJobStore(),
    )

    payload = check_benchmark_datasets_payload(
        benchmark_id="demobench",
        data_root=tmp_path / "datasets",
        context=context,
    )

    assert payload["benchmark_id"] == "demo-benchmark"
    assert payload["summary"]["not_ready"] == 1
    assert payload["results"][0]["status"] == "missing"


def test_mcp_run_lifecycle_exposes_history_status_results_and_logs(tmp_path: Path) -> None:
    async def run() -> None:
        store = AsyncCommandJobStore()
        context = MCPToolContext(output_root=tmp_path, job_store=store)
        result = {"ok": True, "generation_cache": {"hits": 2}, "eligibility": {"leaderboard_valid": False}}
        job = store.submit(
            [sys.executable, "-c", f"import sys; print('ready', file=sys.stderr); print({json.dumps(result)!r})"],
            output_dir=tmp_path / "run-a",
            metadata={"model": CONTRACT_FIXTURE_MODEL_ID, "benchmarks": ["robotwin"]},
            job_id="run-a",
        )
        task = job._task
        assert task is not None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
            history = list_runs_payload(status="completed", limit=1, context=context)
            status = get_run_status_payload("run-a", context=context)
            payload = get_run_result_payload("run-a", include_logs=True, context=context)

            assert history["total"] == 1
            assert history["runs"][0]["job_id"] == "run-a"
            assert history["runs"][0]["metadata"]["benchmarks"] == ["robotwin"]
            assert status["status"] == "completed"
            assert status["returncode"] == 0
            assert any(log["text"].strip() == "ready" for log in status["logs"])
            assert payload["result"] == result
            assert payload["stderr"].strip() == "ready"
            assert json.loads(payload["stdout"]) == result
            assert get_run_result_payload("run-a", context=context)["logs"] == []
            assert list_runs_payload(status="failed", context=context)["runs"] == []
        finally:
            if not job.terminal:
                await cancel_run_payload("run-a", context=context)

    asyncio.run(run())


def test_mcp_cancel_run_reports_terminal_status(tmp_path: Path) -> None:
    async def run() -> None:
        store = AsyncCommandJobStore()
        context = MCPToolContext(output_root=tmp_path, job_store=store)
        job = store.submit(
            [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"],
            output_dir=tmp_path / "running",
            job_id="running",
        )

        task = job._task
        assert task is not None

        async def wait_until_ready() -> None:
            while "ready" not in job.log_text():
                if job.terminal:
                    raise AssertionError(f"command ended before cancellation: {job.error}")
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_until_ready(), timeout=10)
            assert get_run_status_payload("running", context=context)["status"] == "running"
            payload = await cancel_run_payload("running", context=context)
            assert payload == {"success": True, "run_id": "running", "message": "cancelled"}
            assert get_run_result_payload("running", context=context)["status"] == "cancelled"
            assert task.done()
            assert list_runs_payload(status="cancelled", context=context)["total"] == 1
            assert (await cancel_run_payload("running", context=context))["success"] is False
        finally:
            if not job.terminal:
                await cancel_run_payload("running", context=context)

    asyncio.run(run())
