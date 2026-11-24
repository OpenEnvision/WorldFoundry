"""Contracts for the MCP tool surface: commands, schemas, hints, and errors."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import socket
import typing

import pytest

from worldfoundry.cli.main import _build_parser
from worldfoundry.mcp.tools import TOOL_SPECS, MCPToolContext, preview_run_payload
from worldfoundry.mcp.tools.registration import register_tools
from worldfoundry.runtime.jobs import AsyncCommandJobStore


class _RecordingMCP:
    """Stand-in for FastMCP that keeps each tool with the metadata it was registered with."""

    def __init__(self) -> None:
        self.tools: dict[str, tuple[object, dict]] = {}

    def tool(self, **kwargs):
        def decorator(fn):
            self.tools[kwargs["name"]] = (fn, kwargs)
            return fn

        return decorator


@pytest.fixture(scope="module")
def bound() -> dict[str, tuple[object, dict]]:
    mcp = _RecordingMCP()
    register_tools(mcp)
    return mcp.tools


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"benchmarks": ["vbench"]},
        {"benchmarks": ["vbench"], "mode": "official-validation", "plan_only": True, "resume": True},
        {"benchmarks": ["vbench"], "generation_cache_dir": "/tmp/cache", "generation_cache_mode": "read-write"},
        {"benchmarks": ["vbench"], "model_variant": "v1", "metrics": ["artifact_count"], "task_name": "t"},
        {"benchmarks": ["vbench"], "requests_path": "/tmp/requests.jsonl", "output_artifact": "video"},
        {"benchmarks": ["vbench"], "generated_artifact_dir": "/tmp/videos"},
        {"benchmarks": ["vbench", "fetv"], "model_variant": "v1", "metrics": ["artifact_count"], "resume": True},
        {"benchmarks": ["vbench", "fetv"], "generated_artifact_dir": "/tmp/videos", "plan_only": True},
        {"suite_ids": ["smoke"]},
    ],
)
def test_previewed_commands_are_accepted_by_the_cli(tmp_path, options) -> None:
    preview = preview_run_payload(
        model="wan2.1", context=MCPToolContext(output_root=tmp_path, job_store=AsyncCommandJobStore()), **options
    )
    argv = preview["command"][3:]  # drop ``python -m worldfoundry.cli``
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            _build_parser().parse_args(argv)
        except SystemExit:  # pragma: no cover - failure path
            pytest.fail(f"CLI rejected the MCP command {argv}: {stderr.getvalue().strip()}")


@pytest.mark.unit
def test_every_tool_is_async_hinted_and_documented(bound) -> None:
    assert set(bound) == set(TOOL_SPECS)
    for name, (fn, kwargs) in bound.items():
        assert inspect.iscoroutinefunction(fn), name
        assert kwargs["title"] and kwargs["description"], name
        assert "context" not in inspect.signature(fn).parameters, name
        hints = typing.get_type_hints(fn, include_extras=True)
        for param in inspect.signature(fn).parameters:
            if param == "timeout_s":
                continue
            metadata = getattr(hints[param], "__metadata__", ())
            assert any(getattr(item, "description", None) for item in metadata), f"{name}.{param} lacks a description"


@pytest.mark.unit
def test_behaviour_hints_separate_reads_from_mutations() -> None:
    mutating = {name for name, spec in TOOL_SPECS.items() if not spec.read_only}
    assert mutating == {"evaluate", "cancel_run", "submit_studio_inference", "stop_studio_job"}
    assert {name for name, spec in TOOL_SPECS.items() if spec.destructive} == {"cancel_run", "stop_studio_job"}
    assert all(TOOL_SPECS[name].open_world for name in TOOL_SPECS if "studio" in name)


@pytest.mark.unit
def test_unknown_model_is_not_found_with_suggestions(bound) -> None:
    fn, _ = bound["get_model_info"]
    payload = asyncio.run(fn(model_id="wan2.11"))
    assert payload["ok"] is False
    assert payload["error_type"] == "not_found"
    assert "did you mean" in payload["error"]


@pytest.mark.unit
def test_unreachable_studio_reports_unavailable_with_start_hint(bound) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # closed again before the request
    fn, _ = bound["list_studio_models"]
    payload = asyncio.run(fn(base_url=f"http://127.0.0.1:{port}", timeout_s=2))
    assert payload["ok"] is False
    assert payload["error_type"] == "unavailable"
    assert "run_workspace.sh" in payload["error"]


@pytest.mark.unit
def test_catalog_lists_default_to_summary_pages(bound) -> None:
    fn, _ = bound["list_models"]
    first = asyncio.run(fn(limit=5))
    assert first["ok"] is True
    assert len(first["models"]) == 5
    assert first["next_offset"] == 5
    assert "notes" not in first["models"][0]
    full = asyncio.run(fn(limit=1, detail="full"))
    assert "notes" in full["models"][0]
