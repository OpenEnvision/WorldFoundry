# `worldfoundry.mcp`

[Model Context Protocol](https://modelcontextprotocol.io) server that lets an LLM agent
discover models and benchmarks, launch and inspect evaluation runs, and submit Studio
inference jobs. It reuses the same catalog and runtime contracts as the CLI.

## Usage

```bash
python -m pip install -e ".[mcp]"
wf mcp                                          # stdio (default)
wf mcp --transport streamable-http --port 8000  # HTTP at http://127.0.0.1:8000/mcp
wf mcp --transport sse                          # legacy SSE clients
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `WORLDFOUNDRY_STUDIO_WORKSPACE_URL` | Workspace used by Studio tools | `http://127.0.0.1:7870` |
| `WORLDFOUNDRY_MCP_RUN_ROOT` | Where MCP-launched runs are written | `runs/mcp` |

Python clients can use `worldfoundry.mcp.client.MCPClient`; `convert_result_to_openai_format()`
adapts tool results to OpenAI Chat content blocks. Tool contracts (paging, enums, error types)
are documented in the [CLI reference](https://openenvision.github.io/WorldFoundry/docs/reference/cli).

## Layout

| Path | Purpose |
| --- | --- |
| `__init__.py` | `main()` console entry point; transport, host, and port options |
| `server.py` | `create_mcp_server()` — builds the FastMCP server and its agent instructions |
| `client.py` | Stdio client and result-format conversion |
| `tools/spec.py` | `@mcp_tool` registry and `bind_tools()` — async wrapping, context injection, annotations |
| `tools/registration.py` | `register_tools()` — binds every registered tool to a server |
| `tools/discovery.py` | `list_models`, `get_model_info`, `list_benchmarks`, `list_tasks`, … |
| `tools/metrics.py` | `list_metrics`, `show_metric` |
| `tools/readiness.py` | `check_benchmark_datasets` |
| `tools/runs.py` | `preview_run`, `evaluate`, `list_runs`, `get_run_status`, `get_run_result`, `cancel_run`, … |
| `tools/studio.py` | `submit_studio_inference`, `list_studio_jobs`, `wait_for_studio_job`, `get_studio_job_logs`, … |
| `tools/responses.py` | `{"ok": ...}` envelope and the `error_type` taxonomy |
| `tools/server_info.py`, `context.py`, `query.py` | Server metadata and tool list, shared context, query matching |

To add a tool, decorate its payload function with `@mcp_tool(name, title=..., read_only=...)` in a
module under `tools/`, and describe each parameter with `Annotated[T, Field(description=...)]`.
Payloads raise `ToolNotFoundError`, `ToolUnavailableError`, or `ValueError`; the wrapper turns them
into error envelopes. Synchronous payloads run in a worker thread, so blocking I/O is fine.
