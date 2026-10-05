from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.mcp.client import MCPClient, convert_result_to_openai_format
from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID
from worldfoundry.mcp.tools import (
    MCPToolContext,
    check_benchmark_datasets_payload,
    get_run_samples_payload,
    list_benchmarks_payload,
    list_tasks_payload,
    preview_run_payload,
    run_evaluation_payload,
    server_info_payload,
)
from worldfoundry.runtime.jobs import AsyncCommandJobStore, python_module_command


def _tui_server():
    return pytest.importorskip("worldfoundry.cli.tui_server")


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
        context=context,
    )

    assert benchmarks["total"] >= 1
    assert preview["command"][:4] == [sys.executable, "-m", "worldfoundry.cli", "run"]
    assert preview["run_command"][:4] == [sys.executable, "-m", "worldfoundry.cli", "run"]
    assert "--json" in preview["command"]
    assert "--prepare" in preview["command"]
    assert preview["command"][preview["command"].index("--data-root") + 1] == str(tmp_path / "datasets")


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
            "\n".join(
                json.dumps({"doc_id": index, "input": f"prompt-{index}"})
                for index in range(3)
            )
            + "\n",
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
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,efgh"}},
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
                        "official_sources": {
                            "huggingface_datasets": [{"repo_id": "org/data", "license": "mit"}]
                        },
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


def test_web_app_health_route(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    client = TestClient(create_app(output_root=tmp_path))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["app"] == "worldfoundry-web"


def test_web_app_index_exposes_generation_cache_controls(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    client = TestClient(create_app(output_root=tmp_path))
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="cache-mode"' in response.text
    assert 'id="cache-dir"' in response.text
    assert "generation_cache_dir" in response.text
    assert "generation_cache_mode" in response.text


def test_web_app_restricts_cors_to_local_origins_by_default(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    client = TestClient(create_app(output_root=tmp_path))
    bad_origin = client.options(
        "/api/run/preview",
        headers={"Origin": "https://example.invalid", "Access-Control-Request-Method": "POST"},
    )
    local_origin = client.options(
        "/api/run/preview",
        headers={"Origin": "http://127.0.0.1:8000", "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in bad_origin.headers
    assert local_origin.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"


def test_web_app_token_protects_command_routes(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    client = TestClient(create_app(output_root=tmp_path, auth_token="secret-token"))
    denied = client.post(
        "/api/run/preview",
        json={"model_id": CONTRACT_FIXTURE_MODEL_ID, "benchmark_id": "robotwin"},
    )
    index = client.get("/", params={"token": "secret-token"})
    allowed = client.post(
        "/api/run/preview",
        json={"model_id": CONTRACT_FIXTURE_MODEL_ID, "benchmark_id": "robotwin"},
    )

    assert denied.status_code == 401
    assert index.status_code == 200
    assert allowed.status_code == 200
    assert allowed.json()["command"][:2] == ["worldfoundry-eval", "run"]


def test_web_serve_refuses_remote_host_without_token(tmp_path) -> None:
    serve = _tui_server().serve

    with pytest.raises(SystemExit, match="non-loopback host without authentication"):
        serve(host="0.0.0.0", port=9999, open_browser=False, output_root=tmp_path)


def test_web_app_exposes_benchmark_dataset_status(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    manifest_dir = tmp_path / "benchmark_zoo"
    manifest_dir.mkdir()
    (manifest_dir / "benchmarks.yaml").write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "benchmark_id": "demo-benchmark",
                        "source": {"status": "open_source"},
                        "official_sources": {
                            "huggingface_datasets": [{"repo_id": "org/data", "license": "mit"}]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(output_root=tmp_path, benchmark_manifest_dir=manifest_dir))
    response = client.get(
        "/api/assets/benchmark-datasets",
        params={"benchmark_id": "demo-benchmark", "data_root": str(tmp_path / "datasets")},
    )

    assert response.status_code == 200
    assert response.json()["summary"]["by_status"] == {"missing": 1}


def test_web_app_preview_accepts_prepare_data_root(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    client = TestClient(create_app(output_root=tmp_path))
    response = client.post(
        "/api/run/preview",
        json={
            "model_id": CONTRACT_FIXTURE_MODEL_ID,
            "benchmark_id": "robotwin",
            "prepare": True,
            "data_root": str(tmp_path / "datasets"),
            "generation_cache_dir": str(tmp_path / "generation-cache"),
            "generation_cache_mode": "read-write",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "--prepare" in payload["command"]
    assert payload["command"][payload["command"].index("--data-root") + 1] == str(tmp_path / "datasets")
    assert payload["command"][payload["command"].index("--generation-cache-dir") + 1] == str(tmp_path / "generation-cache")
    assert payload["command"][payload["command"].index("--generation-cache-mode") + 1] == "read-write"


def test_web_app_exposes_runtime_preflight_preview(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    client = TestClient(create_app(output_root=tmp_path))
    response = client.post("/api/readiness/preview", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["command"][:3] == ["worldfoundry-eval", "preflight", "runtime"]
    assert "--profile" in payload["command"]
    assert payload["output_dir"] == str(tmp_path / "runtime_preflight")
    assert payload["artifacts"]["report_json"] == str(tmp_path / "runtime_preflight" / "runtime_preflight_report.json")


def test_web_app_serves_run_history_and_samples(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    create_app = _tui_server().create_app

    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    summary = {
        "schema_version": "worldfoundry-run-summary",
        "run": {
            "run_id": "run-a",
            "status": "succeeded",
            "started_at": "2026-05-24T00:00:00Z",
            "generation_cache": {"enabled": True, "mode": "read-write", "hits": 2, "misses": 0, "writes": 0},
        },
        "benchmark": {"benchmark_name": "robotwin", "task_type": "robotics"},
        "model": {"model_id": CONTRACT_FIXTURE_MODEL_ID, "model_name": "Contract Model"},
        "dataset": {"dataset_id": "fixture", "sample_count": 1},
        "counts": {"sample_count": 1, "successful_samples": 1, "failed_samples": 0},
        "metrics": {"leaderboard": {"success_rate": 1.0}},
        "leaderboard": {"success_rate": 1.0},
        "eligibility": {"score_valid": True, "leaderboard_valid": False},
        "artifacts": {"summary": str(run_dir / "summary.json")},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "results.jsonl").write_text(
        json.dumps({"doc_id": 0, "input": "pick", "target": "success", "filtered_resps": ["success"]}) + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(output_root=tmp_path))
    runs_response = client.get("/logs/runs")

    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert runs[0]["model_name"] == "Contract Model"
    assert runs[0]["tasks"] == ["robotwin"]
    assert runs[0]["generation_cache"]["hits"] == 2

    result_response = client.get(f"/logs/runs/{runs[0]['run_id']}/results")
    assert result_response.status_code == 200
    assert result_response.json()["run"]["run_id"] == "run-a"

    samples_response = client.get(f"/logs/runs/{runs[0]['run_id']}/samples/robotwin")
    assert samples_response.status_code == 200
    assert samples_response.json()["samples"][0]["doc_id"] == 0

    index_response = client.get("/logs/index")
    assert index_response.status_code == 200
    assert "WorldFoundry Runs" in index_response.text
