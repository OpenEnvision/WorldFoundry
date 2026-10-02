"""Regression tests for the CLI/MCP review fixes (plan/code_review/05_cli_mcp.md).

Covers the contracts introduced by the CM-01/07/25/26/29/31/32 repairs:
startup-path laziness, structured ``--json`` errors, the MCP error envelope,
async studio waits with finite defaults, absolute MCP default paths, tool-face
consistency, and the persistent ``MCPClient`` session — plus the CM-08 usage
error exit contract (``CliUsageError`` → exit 2 vs runtime failures → exit 1),
the CM-28 persistent MCP job store wiring, and the CM-29 shared default
context (server context written back through ``set_default_context``).

CPU-only; no GPU, network, or optional third-party packages required.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "worldfoundry.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=timeout,
    )


class FakeMCP:
    """Minimal stand-in for FastMCP that records registered tool functions."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture(scope="module")
def registered_tools() -> dict[str, object]:
    from worldfoundry.mcp.tools.registration import register_tools

    fake = FakeMCP()
    register_tools(fake)
    return fake.tools


# ── CM-01: query paths must not import torch or the orchestration stack ──


def test_parser_build_stays_off_heavy_import_paths() -> None:
    # ``run_mode``/``runtime_preflight`` are light parser-data leaves and stay
    # eager; torch and the ~0.7s ``orchestration.service`` chain must not load.
    code = (
        "import sys\n"
        "from worldfoundry.cli.main import _build_parser\n"
        "_build_parser()\n"
        "assert 'torch' not in sys.modules, 'torch imported during parser build'\n"
        "heavy = 'worldfoundry.evaluation.tasks.execution.orchestration.service'\n"
        "assert heavy not in sys.modules, 'orchestration.service imported during parser build'\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_help_exits_zero_without_torch() -> None:
    result = _run_cli(["--help"])
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


# ── CM-07: --json failures keep stdout parseable and exit 1 ─────────────


def test_json_error_contract_and_exit_code() -> None:
    result = _run_cli(["zoo", "benchmark-show", "--benchmark-id", "does-not-exist-xyz", "--json"])
    assert result.returncode == 1, (result.returncode, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["error"]["type"]
    assert "does-not-exist-xyz" in payload["error"]["message"]
    # Human-facing summary goes to stderr, without a traceback by default.
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_usage_error_keeps_exit_code_two() -> None:
    result = _run_cli(["definitely-not-a-command"])
    assert result.returncode == 2
    assert "usage:" in result.stderr


# ── CM-08: handler usage errors exit 2 via CliUsageError, runtime stays 1 ─


def test_handler_usage_error_exits_two_with_concise_stderr(tmp_path: Path) -> None:
    result = _run_cli(["run", "--engine", "in-process", "--output-dir", str(tmp_path / "out")])
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "error: select a run target" in result.stderr
    assert "Traceback" not in result.stderr


def test_handler_usage_error_json_envelope(tmp_path: Path) -> None:
    result = _run_cli(["run", "--engine", "in-process", "--json", "--output-dir", str(tmp_path / "out")])
    assert result.returncode == 2, (result.returncode, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error"]["type"] == "usage"
    assert "select a run target" in payload["error"]["message"]


def test_score_usage_error_fails_fast_before_heavy_import(tmp_path: Path) -> None:
    # The flag validation must run (and exit 2) before the orchestration
    # stack is imported, so the mistake costs milliseconds, not seconds.
    code = (
        "import sys\n"
        "from worldfoundry.cli.main import main\n"
        "rc = main(['score', '--benchmark', 'x', '--output-dir', sys.argv[1]])\n"
        "assert rc == 2, rc\n"
        "heavy = 'worldfoundry.evaluation.tasks.execution.orchestration.service'\n"
        "assert heavy not in sys.modules, 'orchestration.service imported for a usage error'\n"
        "print('USAGE_CONTRACT_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "score-out")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "USAGE_CONTRACT_OK" in result.stdout
    assert "error: score --benchmark requires --artifacts" in result.stderr


def test_usage_error_returns_two_in_process_without_systemexit(tmp_path: Path) -> None:
    # Historic ``return 2`` handler behaviour: in-process callers of ``main``
    # get the exit code as a return value, never a raised SystemExit.
    code = (
        "import sys\n"
        "from worldfoundry.cli.main import main\n"
        "rc = main(['evaluate', '--samples-path', 'x.json', '--output-dir', sys.argv[1]])\n"
        "assert rc == 2, rc\n"
        "print('RETURN_VALUE_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "eval-out")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "RETURN_VALUE_OK" in result.stdout


def test_runtime_failure_keeps_exit_code_one_distinct_from_usage() -> None:
    # CM-07's runtime-failure contract must survive the CM-08 split: an
    # unknown id is a runtime lookup failure (exit 1), not a usage error.
    result = _run_cli(["zoo", "benchmark-show", "--benchmark-id", "does-not-exist-xyz"])
    assert result.returncode == 1, (result.returncode, result.stderr)


def test_run_print_config_without_model_is_usage_error(tmp_path: Path) -> None:
    # CM-05 routing audit: ``run --print-config`` without a positional model
    # used to surface as a runtime ValueError (exit 1); it is a usage mistake
    # and must exit 2 with the guidance message.
    result = _run_cli(["run", "--print-config", "--output-dir", str(tmp_path / "out")])
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "error: model-specific configuration requires `run <model-id>`" in result.stderr
    assert "Traceback" not in result.stderr


# ── CM-25: registered tools return the structured error envelope ────────


def test_tool_errors_return_envelope_instead_of_raising(registered_tools: dict[str, object]) -> None:
    list_runs = registered_tools["list_runs"]
    payload = list_runs(limit=0)
    assert payload["ok"] is False
    assert payload["error_type"] == "error"
    assert "limit" in payload["error"]


def test_async_tool_errors_return_envelope(registered_tools: dict[str, object]) -> None:
    evaluate = registered_tools["evaluate"]
    payload = asyncio.run(evaluate(model="m", benchmark="b", wait="bogus-strategy"))
    assert payload["ok"] is False
    assert payload["error_type"] == "error"
    assert "wait" in payload["error"]


# ── CM-26: studio waits are async with finite default timeouts ──────────


def test_studio_wait_tools_are_async(registered_tools: dict[str, object]) -> None:
    assert inspect.iscoroutinefunction(registered_tools["wait_for_studio_job"])
    assert inspect.iscoroutinefunction(registered_tools["submit_studio_inference"])


def test_studio_wait_default_timeout_is_finite() -> None:
    from worldfoundry.mcp.tools import studio

    assert studio.DEFAULT_STUDIO_WAIT_TIMEOUT_S == 600.0
    for fn, param in (
        (studio.wait_for_studio_job_payload, "timeout_s"),
        (studio.wait_for_studio_job_payload_async, "timeout_s"),
        (studio.submit_studio_inference_payload, "wait_timeout_s"),
        (studio.submit_studio_inference_payload_async, "wait_timeout_s"),
    ):
        default = inspect.signature(fn).parameters[param].default
        assert default == studio.DEFAULT_STUDIO_WAIT_TIMEOUT_S


# ── CM-29: MCP defaults resolve to absolute paths ────────────────────────


def test_mcp_default_output_root_is_absolute() -> None:
    from worldfoundry.mcp.tools.context import DEFAULT_MCP_OUTPUT_ROOT

    assert DEFAULT_MCP_OUTPUT_ROOT.is_absolute()


def test_readiness_default_dataset_root_is_absolute() -> None:
    from worldfoundry.mcp.tools.readiness import _default_dataset_root

    assert _default_dataset_root().is_absolute()


# ── CM-31: advertised tool face matches the registrations ────────────────


def test_mcp_tool_names_match_registration(registered_tools: dict[str, object]) -> None:
    from worldfoundry.mcp.tools.server_info import MCP_TOOL_NAMES

    assert sorted(MCP_TOOL_NAMES) == sorted(registered_tools)


def test_preview_run_covers_evaluate_selection_surface(registered_tools: dict[str, object]) -> None:
    preview_params = set(inspect.signature(registered_tools["preview_run"]).parameters)
    evaluate_params = set(inspect.signature(registered_tools["evaluate"]).parameters)
    # Wait-strategy knobs only make sense for submission, not preview.
    assert evaluate_params - preview_params <= {"wait", "wait_timeout_s"}


# ── CM-32: MCPClient supports a persistent session ────────────────────────


def test_mcp_client_persistent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.mcp.client import MCPClient

    created: list[object] = []

    class FakeSession:
        async def list_tools(self):
            class Result:
                tools = ["one"]

            return Result()

        async def call_tool(self, name, arguments):
            return {"name": name, "arguments": arguments}

    class FakeContext:
        def __init__(self) -> None:
            self.closed = False

        async def __aenter__(self):
            created.append(self)
            return FakeSession()

        async def __aexit__(self, *exc):
            self.closed = True
            return False

    async def fake_client(self):
        return FakeContext()

    monkeypatch.setattr(MCPClient, "_client", fake_client)

    async def scenario() -> None:
        # Default mode still opens one session per call.
        client = MCPClient()
        await client.list_tools()
        await client.call_tool("x")
        assert len(created) == 2
        assert all(ctx.closed for ctx in created)

        created.clear()
        async with MCPClient() as persistent:
            await persistent.list_tools()
            await persistent.call_tool("y")
            await persistent.call_tool("z")
        assert len(created) == 1
        assert created[0].closed

    asyncio.run(scenario())


# ── CM-28: the MCP context wires a persistent, bounded job store ──────────


def test_mcp_context_default_store_is_persistent_and_bounded(tmp_path: Path) -> None:
    from worldfoundry.mcp.tools.context import (
        DEFAULT_MCP_MAX_TRACKED_JOBS,
        MCP_JOB_INDEX_FILENAME,
        MCPToolContext,
    )

    ctx = MCPToolContext(output_root=tmp_path)
    assert ctx.job_store is not None
    assert ctx.job_store.state_path == tmp_path / MCP_JOB_INDEX_FILENAME
    assert ctx.job_store.max_jobs == DEFAULT_MCP_MAX_TRACKED_JOBS


def test_mcp_context_restores_persisted_runs_for_payloads(tmp_path: Path) -> None:
    from worldfoundry.mcp.tools.context import MCP_JOB_INDEX_FILENAME, MCPToolContext
    from worldfoundry.mcp.tools.runs import get_run_status_payload, list_runs_payload
    from worldfoundry.runtime.jobs import JOB_STORE_STATE_SCHEMA_VERSION

    index_path = tmp_path / MCP_JOB_INDEX_FILENAME
    index_path.write_text(
        json.dumps(
            {
                "schema_version": JOB_STORE_STATE_SCHEMA_VERSION,
                "jobs": [
                    {
                        "job_id": "persisted-run",
                        "run_id": "persisted-run",
                        "pid": 1234567,
                        "status": "completed",
                        "created_at": "2026-08-25T00:00:00+00:00",
                        "started_at": "2026-08-25T00:00:01+00:00",
                        "completed_at": "2026-08-25T00:00:09+00:00",
                        "returncode": 0,
                        "output_dir": str(tmp_path / "persisted-run"),
                        "command": ["worldfoundry-eval", "run", "--json"],
                        "metadata": {"surface": "mcp"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ctx = MCPToolContext(output_root=tmp_path)
    listed = list_runs_payload(context=ctx)
    assert listed["total"] == 1
    assert listed["runs"][0]["job_id"] == "persisted-run"
    assert listed["runs"][0]["restored"] is True
    status = get_run_status_payload("persisted-run", context=ctx)
    assert status["status"] == "completed"
    assert status["output_dir"] == str(tmp_path / "persisted-run")


# ── CM-29: one default context shared by server and bare payload calls ────


@pytest.fixture()
def _default_context_guard():
    from worldfoundry.mcp.tools import context as context_module

    original = context_module.get_default_context()
    try:
        yield
    finally:
        context_module.set_default_context(original)


def test_set_default_context_is_observed_by_payload_functions(
    tmp_path: Path, _default_context_guard
) -> None:
    from worldfoundry.mcp.tools.context import MCPToolContext, get_default_context, set_default_context
    from worldfoundry.mcp.tools.runs import list_runs_payload

    installed = set_default_context(MCPToolContext(output_root=tmp_path))
    assert get_default_context() is installed
    # ``runs`` resolves the default lazily via get_default_context(), so a
    # bare payload call consumes the newly installed context's store.
    payload = list_runs_payload()
    assert payload["total"] == 0
    assert installed.job_store is not None
    assert installed.job_store.state_path == tmp_path / "jobs-index.json"


def test_create_mcp_server_writes_context_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _default_context_guard
) -> None:
    import worldfoundry.mcp.server as server_module
    from worldfoundry.mcp.tools.context import get_default_context

    class FakeFastMCP:
        def __init__(self, name, *, instructions=None, json_response=False) -> None:
            self.name = name
            self.tools: dict[str, object] = {}

        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn

            return decorator

    monkeypatch.setenv("WORLDFOUNDRY_MCP_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(server_module, "require_fastmcp", lambda: FakeFastMCP)
    before = get_default_context()
    server = server_module.create_mcp_server()
    after = get_default_context()
    assert after is not before
    assert after.output_root == tmp_path

    # The registered tools and bare payload calls consume one shared store:
    # a job submitted into the default context is visible through the tool.
    async def scenario() -> dict:
        job = after.job_store.submit(
            [sys.executable, "-c", "print('shared-store')"],
            metadata={"run_id": "shared-store-run"},
        )
        await job._task
        return server.tools["list_runs"]()

    envelope = asyncio.run(scenario())
    assert envelope["ok"] is True
    assert envelope["total"] == 1
    assert envelope["runs"][0]["run_id"] == "shared-store-run"
