"""Studio workspace payloads for the WorldFoundry MCP server.

Provides payload-building functions that interact with a local Studio
workspace HTTP API — listing models, submitting inference jobs, polling
job status, and retrieving artifacts.  Each function returns a structured
dictionary suitable for MCP tool responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

# Default Studio workspace URL, overrideable via the ``WORLDFOUNDRY_STUDIO_WORKSPACE_URL`` environment variable.
DEFAULT_STUDIO_WORKSPACE_URL = os.environ.get("WORLDFOUNDRY_STUDIO_WORKSPACE_URL", "http://127.0.0.1:7870")

# Finite default wait budget: an agent tool call must never park the MCP
# server on an unbounded poll loop by default. Pass ``timeout_s=0`` (or a
# negative value) explicitly to opt in to an unlimited wait.
DEFAULT_STUDIO_WAIT_TIMEOUT_S = 600.0

_TERMINAL_STUDIO_JOB_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled"})


# ── Model discovery ────────────────────────────────────────────────────


def list_studio_models_payload(
    *,
    base_url: str | None = None,
    query: str | None = None,
    workload_type: str | None = None,
    supports_from_pretrained: bool | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return Studio workspace model entries, optionally filtered for agents.

    Args:
        base_url: Studio workspace base URL (defaults to :data:`DEFAULT_STUDIO_WORKSPACE_URL`).
        query: Case-insensitive substring filter across ``id``, ``name``, ``family``,
            ``category``, ``summary``, and ``workload_type`` fields.
        workload_type: Filter by exact workload type (case-insensitive).
        supports_from_pretrained: Filter by ``supports_from_pretrained`` flag.
        timeout_s: HTTP request timeout in seconds.

    Raises:
        RuntimeError: If the Studio ``/api/models`` endpoint returns a non-list payload.

    Returns:
        Dictionary with ``models``, ``total``, and ``base_url`` keys.
    """

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


def get_studio_model_info_payload(
    model_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return one Studio model entry by id.

    Args:
        model_id: Exact model identifier to look up.
        base_url: Studio workspace base URL (optional).
        timeout_s: HTTP request timeout in seconds.

    Raises:
        ValueError: If ``model_id`` is not found among Studio models.

    Returns:
        Model dictionary from the Studio API.
    """

    for row in list_studio_models_payload(base_url=base_url, timeout_s=timeout_s)["models"]:
        if row.get("id") == model_id:
            return row
    raise ValueError(f"Studio model not found: {model_id}")


# ── Inference submission ────────────────────────────────────────────────


def submit_studio_inference_payload(
    *,
    model_id: str,
    base_url: str | None = None,
    variant_id: str = "",
    task_profile_id: str = "",
    prompt: str = "",
    negative_prompt: str = "",
    input_path: str = "",
    backend: str = "",
    device: str = "",
    params: dict[str, Any] | None = None,
    call_kwargs: dict[str, Any] | None = None,
    load_kwargs: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    wait: bool = False,
    wait_timeout_s: float = DEFAULT_STUDIO_WAIT_TIMEOUT_S,
    poll_interval_s: float = 5,
    timeout_s: float = 30,
) -> dict[str, Any]:
    """Submit an inference job to the Studio workspace API.

    Args:
        model_id: Model identifier to run inference on.
        base_url: Studio workspace base URL (optional).
        variant_id: Model variant selector (optional).
        task_profile_id: Task profile to apply (optional).
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt for generation.
        input_path: Path to input file (optional).
        backend: Backend selector (defaults to ``"auto"``).
        device: Device selector (optional).
        params: Additional inference parameters.
        call_kwargs: Keyword arguments passed to the model call.
        load_kwargs: Keyword arguments passed to model loading.
        env_overrides: Environment variable overrides for the job.
        wait: Poll the job until it reaches a terminal state.
        wait_timeout_s: Maximum seconds to wait (0 or negative = unlimited;
            default ``DEFAULT_STUDIO_WAIT_TIMEOUT_S``).
        poll_interval_s: Seconds between status polls.
        timeout_s: HTTP request timeout for the initial submission.

    Returns:
        Job dictionary from the Studio API.  If ``wait`` is **True**, returns
        the final job state after polling.
    """

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
                job_id,
                base_url=base_url,
                timeout_s=wait_timeout_s,
                poll_interval_s=poll_interval_s,
            )
    return job


async def submit_studio_inference_payload_async(
    *,
    model_id: str,
    base_url: str | None = None,
    variant_id: str = "",
    task_profile_id: str = "",
    prompt: str = "",
    negative_prompt: str = "",
    input_path: str = "",
    backend: str = "",
    device: str = "",
    params: dict[str, Any] | None = None,
    call_kwargs: dict[str, Any] | None = None,
    load_kwargs: dict[str, Any] | None = None,
    env_overrides: dict[str, str] | None = None,
    wait: bool = False,
    wait_timeout_s: float = DEFAULT_STUDIO_WAIT_TIMEOUT_S,
    poll_interval_s: float = 5,
    timeout_s: float = 30,
) -> dict[str, Any]:
    """Event-loop-safe variant of :func:`submit_studio_inference_payload`.

    The blocking HTTP submission runs in a worker thread and any follow-up
    polling awaits :func:`wait_for_studio_job_payload_async`, so long waits
    never stall the MCP server event loop (CM-26).
    """

    job = await asyncio.to_thread(
        lambda: submit_studio_inference_payload(
            model_id=model_id,
            base_url=base_url,
            variant_id=variant_id,
            task_profile_id=task_profile_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_path=input_path,
            backend=backend,
            device=device,
            params=params,
            call_kwargs=call_kwargs,
            load_kwargs=load_kwargs,
            env_overrides=env_overrides,
            wait=False,
            timeout_s=timeout_s,
        )
    )
    if wait:
        job_id = str(job.get("id") or job.get("job_id") or "")
        if job_id:
            return await wait_for_studio_job_payload_async(
                job_id,
                base_url=base_url,
                timeout_s=wait_timeout_s,
                poll_interval_s=poll_interval_s,
            )
    return job


# ── Job lifecycle ──────────────────────────────────────────────────────


def list_studio_jobs_payload(
    *,
    base_url: str | None = None,
    job_type: str | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """List Studio workspace jobs, optionally filtered by type.

    Args:
        base_url: Studio workspace base URL (optional).
        job_type: Filter by job type (e.g. ``"inference"``).
        timeout_s: HTTP request timeout in seconds.

    Raises:
        RuntimeError: If the Studio ``/api/jobs`` endpoint returns a non-list payload.

    Returns:
        Dictionary with ``jobs``, ``total``, and ``base_url`` keys.
    """
    query = {"job_type": job_type} if job_type else None
    jobs = _request_json("GET", "/api/jobs", query=query, base_url=base_url, timeout_s=timeout_s)
    if not isinstance(jobs, list):
        raise RuntimeError("Studio /api/jobs returned a non-list payload")
    return {"jobs": jobs, "total": len(jobs), "base_url": _normalize_base_url(base_url)}


def get_studio_job_payload(
    job_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return a single Studio job by id.

    Args:
        job_id: Studio job identifier.
        base_url: Studio workspace base URL (optional).
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Job dictionary from the Studio API.
    """
    return _request_json("GET", f"/api/jobs/{job_id}", base_url=base_url, timeout_s=timeout_s)


def wait_for_studio_job_payload(
    job_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_STUDIO_WAIT_TIMEOUT_S,
    poll_interval_s: float = 5,
) -> dict[str, Any]:
    """Poll a Studio job until it reaches a terminal status or timeout.

    Terminal statuses are: ``completed``, ``failed``, ``cancelled``, ``canceled``.

    Blocking variant for synchronous scripts; server code must use
    :func:`wait_for_studio_job_payload_async` instead.

    Args:
        job_id: Studio job identifier to poll.
        base_url: Studio workspace base URL (optional).
        timeout_s: Maximum seconds to wait (0 or negative = unlimited;
            default ``DEFAULT_STUDIO_WAIT_TIMEOUT_S``).
        poll_interval_s: Seconds between status polls.

    Returns:
        Final job dictionary when a terminal status is reached, or the last
        polled state if the timeout expires.
    """

    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while True:
        payload = get_studio_job_payload(job_id, base_url=base_url, timeout_s=max(1.0, min(poll_interval_s, 30.0)))
        if str(payload.get("status") or "").lower() in _TERMINAL_STUDIO_JOB_STATUSES:
            return payload
        if deadline is not None and time.monotonic() >= deadline:
            return payload
        time.sleep(max(0.25, poll_interval_s))


async def wait_for_studio_job_payload_async(
    job_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_STUDIO_WAIT_TIMEOUT_S,
    poll_interval_s: float = 5,
) -> dict[str, Any]:
    """Event-loop-safe variant of :func:`wait_for_studio_job_payload`.

    Polls with ``asyncio.sleep`` and runs each blocking HTTP status request in
    a worker thread, mirroring the ``evaluate`` tool's wait loop, so a long
    Studio job can never wedge the MCP server event loop (CM-26).
    """

    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while True:
        payload = await asyncio.to_thread(
            lambda: get_studio_job_payload(
                job_id,
                base_url=base_url,
                timeout_s=max(1.0, min(poll_interval_s, 30.0)),
            )
        )
        if str(payload.get("status") or "").lower() in _TERMINAL_STUDIO_JOB_STATUSES:
            return payload
        if deadline is not None and time.monotonic() >= deadline:
            return payload
        await asyncio.sleep(max(0.25, poll_interval_s))


def stop_studio_job_payload(
    job_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Cancel a running or pending Studio job.

    Args:
        job_id: Studio job identifier.
        base_url: Studio workspace base URL (optional).
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Stop response dictionary with ``ok``, ``message``, and ``job`` keys.
    """
    return _request_json("POST", f"/api/jobs/{job_id}/stop", base_url=base_url, timeout_s=timeout_s)


def get_studio_job_logs_payload(
    job_id: str,
    *,
    after: int = 0,
    base_url: str | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return incremental log lines for a Studio job.

    Args:
        job_id: Studio job identifier.
        after: Skip the first *after* log entries.
        base_url: Studio workspace base URL (optional).
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Log payload dictionary with ``offset``, ``logs``, and ``text`` keys.
    """
    return _request_json(
        "GET",
        f"/api/jobs/{job_id}/logs",
        query={"after": after},
        base_url=base_url,
        timeout_s=timeout_s,
    )


# ── Artifacts ──────────────────────────────────────────────────────────


def list_studio_artifacts_payload(
    *,
    base_url: str | None = None,
    limit: int = 20,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """List Studio workspace artifacts.

    Args:
        base_url: Studio workspace base URL (optional).
        limit: Maximum number of artifacts to return (clamped to 1–200).
        timeout_s: HTTP request timeout in seconds.

    Raises:
        RuntimeError: If the Studio ``/api/artifacts`` endpoint returns a non-list payload.

    Returns:
        Dictionary with ``artifacts``, ``total``, and ``base_url`` keys.
    """
    limit = max(1, min(int(limit), 200))
    artifacts = _request_json("GET", "/api/artifacts", base_url=base_url, timeout_s=timeout_s)
    if not isinstance(artifacts, list):
        raise RuntimeError("Studio /api/artifacts returned a non-list payload")
    rows = [item for item in artifacts if isinstance(item, dict)]
    return {
        "artifacts": rows[:limit],
        "total": len(rows),
        "base_url": _normalize_base_url(base_url),
    }


def get_studio_manifest_payload(
    job_id: str,
    *,
    base_url: str | None = None,
    timeout_s: float = 15,
) -> dict[str, Any]:
    """Return manifest metadata for a Studio job from its result payload.

    The workspace API embeds manifest paths on ``GET /api/jobs/{job_id}``;
    there is no dedicated ``/manifest`` route.

    Args:
        job_id: Studio job identifier.
        base_url: Studio workspace base URL (optional).
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Dictionary with ``job_id``, ``manifest_path``, ``output_dir``,
        ``artifacts``, and ``metadata`` keys.
    """
    job = get_studio_job_payload(job_id, base_url=base_url, timeout_s=timeout_s)
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
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
    """Send an HTTP request to the Studio API and parse the JSON response.

    Args:
        method: HTTP method (``"GET"``, ``"POST"``, etc.).
        path: API path relative to ``base_url`` (e.g. ``"/api/models"``).
        payload: JSON body for POST/PUT requests (optional).
        query: Query parameters to append to the URL (optional).
        base_url: Studio workspace base URL (optional).
        timeout_s: HTTP request timeout in seconds.

    Raises:
        RuntimeError: If the Studio API returns a non-2xx HTTP status.

    Returns:
        Parsed JSON response, or **None** if the response body is empty.
    """
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
        raise RuntimeError(f"Studio API {method.upper()} {path} failed with HTTP {exc.code}: {detail}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _api_url(base_url: str | None, path: str, *, query: dict[str, Any] | None = None) -> str:
    """Construct a full Studio API URL from a base URL, path, and query params.

    Filters out **None** values from ``query`` before encoding.
    """
    url = urljoin(_normalize_base_url(base_url) + "/", path.lstrip("/"))
    if query:
        url = f"{url}?{urlencode({key: value for key, value in query.items() if value is not None})}"
    return url


def _normalize_base_url(base_url: str | None) -> str:
    """Normalise and validate a Studio base URL.

    Strips whitespace and trailing slashes, then verifies the URL starts
    with ``http://`` or ``https://``.

    Raises:
        ValueError: If the URL does not start with ``http://`` or ``https://``.
    """
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
    "submit_studio_inference_payload_async",
    "wait_for_studio_job_payload",
    "wait_for_studio_job_payload_async",
]
