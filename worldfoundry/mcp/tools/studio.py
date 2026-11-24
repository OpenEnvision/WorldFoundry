"""Studio workspace payloads for the WorldFoundry MCP server.

These tools call the HTTP API of a running Studio workspace (``wf`` Studio,
started with ``bash scripts/workspace/run_workspace.sh``): list its models,
submit inference jobs, poll them, and read their logs and artifacts. The server
runs each call in a worker thread, so slow requests and long waits never block
other tool calls.
"""

from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from pydantic import Field

from .responses import ToolNotFoundError, ToolUnavailableError
from .spec import mcp_tool

# Default Studio workspace URL, overrideable via the ``WORLDFOUNDRY_STUDIO_WORKSPACE_URL`` environment variable.
DEFAULT_STUDIO_WORKSPACE_URL = os.environ.get("WORLDFOUNDRY_STUDIO_WORKSPACE_URL", "http://127.0.0.1:7870")

# Finite default wait budget: an agent tool call must never park the MCP
# server on an unbounded poll loop by default. Pass ``timeout_s=0`` (or a
# negative value) explicitly to opt in to an unlimited wait.
DEFAULT_STUDIO_WAIT_TIMEOUT_S = 600.0

_TERMINAL_STUDIO_JOB_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled"})

BaseUrl = Annotated[
    str | None,
    Field(description="Studio workspace URL; defaults to WORLDFOUNDRY_STUDIO_WORKSPACE_URL or http://127.0.0.1:7870."),
]
JobId = Annotated[str, Field(description="Studio job id from submit_studio_inference or list_studio_jobs.")]
WaitTimeout = Annotated[float, Field(description="Maximum seconds to wait; 0 or negative waits without limit.")]
PollInterval = Annotated[float, Field(gt=0, description="Seconds between status polls.")]


# ── Model discovery ────────────────────────────────────────────────────


@mcp_tool("list_studio_models", title="List Studio models", read_only=True, open_world=True)
def list_studio_models_payload(
    *,
    query: Annotated[
        str | None, Field(description="Case-insensitive substring across id, name, family, category, summary.")
    ] = None,
    workload_type: Annotated[str | None, Field(description="Exact workload type.")] = None,
    supports_from_pretrained: Annotated[
        bool | None, Field(description="Only models that do (true) or do not (false) load via from_pretrained.")
    ] = None,
    base_url: BaseUrl = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """List the models a running Studio workspace can launch; use their ``id`` with submit_studio_inference."""

    query_params = {"workload_type": workload_type} if workload_type else None
    models = _request_json("GET", "/api/models", query=query_params, base_url=base_url, timeout_s=timeout_s)
    if not isinstance(models, list):
        raise RuntimeError("Studio /api/models returned a non-list payload")
    rows = [item for item in models if isinstance(item, dict)]
    if query:
        needle = query.casefold()
        rows = [
            row
            for row in rows
            if any(
                needle in str(row.get(key, "")).casefold()
                for key in ("id", "name", "family", "category", "summary", "workload_type")
            )
        ]
    if workload_type:
        expected = workload_type.casefold()
        rows = [row for row in rows if str(row.get("workload_type", "")).casefold() == expected]
    if supports_from_pretrained is not None:
        rows = [row for row in rows if bool(row.get("supports_from_pretrained")) is supports_from_pretrained]
    return {"models": rows, "total": len(rows), "base_url": _normalize_base_url(base_url)}


@mcp_tool("get_studio_model_info", title="Get Studio model details", read_only=True, open_world=True)
def get_studio_model_info_payload(
    model_id: Annotated[str, Field(description="Exact Studio model id from list_studio_models.")],
    *,
    base_url: BaseUrl = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return one Studio model entry: its variants, task profiles, and input fields."""

    rows = list_studio_models_payload(base_url=base_url, timeout_s=timeout_s)["models"]
    for row in rows:
        if row.get("id") == model_id:
            return row
    raise ToolNotFoundError.for_id("Studio model", model_id, (str(row.get("id", "")) for row in rows))


# ── Inference submission ────────────────────────────────────────────────


@mcp_tool(
    "submit_studio_inference", title="Submit a Studio inference job", read_only=False, idempotent=False, open_world=True
)
def submit_studio_inference_payload(
    *,
    model_id: Annotated[str, Field(description="Studio model id from list_studio_models.")],
    base_url: BaseUrl = None,
    variant_id: Annotated[str, Field(description="Model variant id; empty uses the default.")] = "",
    task_profile_id: Annotated[str, Field(description="Task profile id; empty uses the default.")] = "",
    prompt: Annotated[str, Field(description="Text prompt.")] = "",
    negative_prompt: Annotated[str, Field(description="Negative prompt.")] = "",
    input_path: Annotated[str, Field(description="Input image, video, or directory on the workspace host.")] = "",
    backend: Annotated[str, Field(description="Execution backend; empty selects auto.")] = "",
    device: Annotated[str, Field(description="Device such as cuda or cuda:1; empty uses the workspace default.")] = "",
    params: Annotated[dict[str, Any] | None, Field(description="Additional model-form parameters.")] = None,
    call_kwargs: Annotated[dict[str, Any] | None, Field(description="Keyword arguments for the pipeline call.")] = None,
    load_kwargs: Annotated[dict[str, Any] | None, Field(description="Keyword arguments for pipeline loading.")] = None,
    env_overrides: Annotated[dict[str, str] | None, Field(description="Environment overrides for the job.")] = None,
    wait: Annotated[bool, Field(description="Poll until the job finishes or wait_timeout_s elapses.")] = False,
    wait_timeout_s: WaitTimeout = DEFAULT_STUDIO_WAIT_TIMEOUT_S,
    poll_interval_s: PollInterval = 5,
    timeout_s: float = 30,
) -> dict[str, Any]:
    """Submit an inference job to a Studio workspace and return the job (or its final state with wait=true)."""

    payload: dict[str, Any] = {
        "job_type": "inference",
        "model_id": model_id,
        "variant_id": variant_id,
        "task_profile_id": task_profile_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "input_path": input_path,
        "backend": backend or "auto",
        "device": device,
        "params": dict(params or {}),
        "call_kwargs": dict(call_kwargs or {}),
        "load_kwargs": dict(load_kwargs or {}),
        "env_overrides": dict(env_overrides or {}),
    }
    job = _request_json("POST", "/api/jobs", payload=payload, base_url=base_url, timeout_s=timeout_s)
    if wait:
        job_id = str(job.get("id") or job.get("job_id") or "")
        if job_id:
            return wait_for_studio_job_payload(
                job_id, base_url=base_url, timeout_s=wait_timeout_s, poll_interval_s=poll_interval_s
            )
    return job


# ── Job lifecycle ──────────────────────────────────────────────────────


@mcp_tool("list_studio_jobs", title="List Studio jobs", read_only=True, open_world=True)
def list_studio_jobs_payload(
    *,
    job_type: Annotated[str | None, Field(description="Only jobs of this type, such as inference.")] = None,
    base_url: BaseUrl = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """List jobs known to a Studio workspace."""

    query = {"job_type": job_type} if job_type else None
    jobs = _request_json("GET", "/api/jobs", query=query, base_url=base_url, timeout_s=timeout_s)
    if not isinstance(jobs, list):
        raise RuntimeError("Studio /api/jobs returned a non-list payload")
    return {"jobs": jobs, "total": len(jobs), "base_url": _normalize_base_url(base_url)}


@mcp_tool("get_studio_job", title="Get a Studio job", read_only=True, open_world=True)
def get_studio_job_payload(job_id: JobId, *, base_url: BaseUrl = None, timeout_s: float = 15) -> dict[str, Any]:
    """Return one Studio job: status, progress, and result when finished."""

    return _request_json("GET", f"/api/jobs/{job_id}", base_url=base_url, timeout_s=timeout_s)


@mcp_tool("wait_for_studio_job", title="Wait for a Studio job", read_only=True, open_world=True)
def wait_for_studio_job_payload(
    job_id: JobId,
    *,
    base_url: BaseUrl = None,
    timeout_s: WaitTimeout = DEFAULT_STUDIO_WAIT_TIMEOUT_S,
    poll_interval_s: PollInterval = 5,
) -> dict[str, Any]:
    """Poll a Studio job until it completes, fails, or is cancelled; return the last state on timeout."""

    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while True:
        payload = get_studio_job_payload(job_id, base_url=base_url, timeout_s=max(1.0, min(poll_interval_s, 30.0)))
        if str(payload.get("status") or "").lower() in _TERMINAL_STUDIO_JOB_STATUSES:
            return payload
        if deadline is not None and time.monotonic() >= deadline:
            return payload
        time.sleep(max(0.25, poll_interval_s))


@mcp_tool(
    "stop_studio_job", title="Stop a Studio job", read_only=False, destructive=True, idempotent=True, open_world=True
)
def stop_studio_job_payload(job_id: JobId, *, base_url: BaseUrl = None, timeout_s: float = 15) -> dict[str, Any]:
    """Cancel a queued or running Studio job."""

    return _request_json("POST", f"/api/jobs/{job_id}/stop", base_url=base_url, timeout_s=timeout_s)


@mcp_tool("get_studio_job_logs", title="Get Studio job logs", read_only=True, open_world=True)
def get_studio_job_logs_payload(
    job_id: JobId,
    *,
    after: Annotated[int, Field(ge=0, description="Return log entries after this offset (from a previous call).")] = 0,
    base_url: BaseUrl = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return new log lines for a Studio job; pass the returned ``offset`` as ``after`` to continue."""

    return _request_json(
        "GET", f"/api/jobs/{job_id}/logs", query={"after": after}, base_url=base_url, timeout_s=timeout_s
    )


# ── Artifacts ──────────────────────────────────────────────────────────


@mcp_tool("list_studio_artifacts", title="List Studio artifacts", read_only=True, open_world=True)
def list_studio_artifacts_payload(
    *,
    base_url: BaseUrl = None,
    limit: Annotated[int, Field(ge=1, le=200, description="Maximum artifacts to return.")] = 20,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """List artifacts produced in the Studio workspace session, newest first."""

    limit = max(1, min(int(limit), 200))
    artifacts = _request_json("GET", "/api/artifacts", base_url=base_url, timeout_s=timeout_s)
    if not isinstance(artifacts, list):
        raise RuntimeError("Studio /api/artifacts returned a non-list payload")
    rows = [item for item in artifacts if isinstance(item, dict)]
    return {"artifacts": rows[:limit], "total": len(rows), "base_url": _normalize_base_url(base_url)}


@mcp_tool("get_studio_manifest", title="Get a Studio job manifest", read_only=True, open_world=True)
def get_studio_manifest_payload(job_id: JobId, *, base_url: BaseUrl = None, timeout_s: float = 15) -> dict[str, Any]:
    """Return a finished Studio job's manifest path, output directory, artifacts, and metadata."""

    job = get_studio_job_payload(job_id, base_url=base_url, timeout_s=timeout_s)
    result = job.get("result")
    if not isinstance(result, dict):
        result = {}
    return {
        "job_id": job_id,
        "base_url": _normalize_base_url(base_url),
        "manifest_path": str(result.get("manifest_path") or ""),
        "output_dir": str(result.get("output_dir") or job.get("output_dir") or ""),
        "artifacts": result.get("artifacts") or {},
        "metadata": result.get("metadata") or {},
    }


# ── Internal HTTP helpers ──────────────────────────────────────────────


def _request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    base_url: str | None = None,
    timeout_s: float = 15,
) -> Any:
    """Send one request to the Studio API and parse the JSON response (``None`` for an empty body)."""

    root = _normalize_base_url(base_url)
    url = _api_url(base_url, path, query=query)
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise ToolNotFoundError(f"Studio API {method.upper()} {path}: not found ({detail.strip()[:200]})") from exc
        raise RuntimeError(f"Studio API {method.upper()} {path} failed with HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ToolUnavailableError(
            f"Studio workspace is not reachable at {root} ({reason}). Start it with "
            "`bash scripts/workspace/run_workspace.sh`, or pass base_url / set WORLDFOUNDRY_STUDIO_WORKSPACE_URL."
        ) from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _api_url(base_url: str | None, path: str, *, query: dict[str, Any] | None = None) -> str:
    """Join the base URL, path, and non-``None`` query parameters."""

    url = urljoin(_normalize_base_url(base_url) + "/", path.lstrip("/"))
    if query:
        url = f"{url}?{urlencode({key: value for key, value in query.items() if value is not None})}"
    return url


def _normalize_base_url(base_url: str | None) -> str:
    """Strip trailing slashes and require an ``http(s)://`` URL."""

    url = (base_url or DEFAULT_STUDIO_WORKSPACE_URL).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("Studio base_url must start with http:// or https://")
    return url


__all__ = [
    "DEFAULT_STUDIO_WAIT_TIMEOUT_S",
    "DEFAULT_STUDIO_WORKSPACE_URL",
    "get_studio_job_logs_payload",
    "get_studio_job_payload",
    "get_studio_manifest_payload",
    "get_studio_model_info_payload",
    "list_studio_artifacts_payload",
    "list_studio_jobs_payload",
    "list_studio_models_payload",
    "stop_studio_job_payload",
    "submit_studio_inference_payload",
    "wait_for_studio_job_payload",
]
