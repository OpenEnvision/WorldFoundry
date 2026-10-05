"""Shared DashScope video-synthesis client for the hosted Wan models.

Wan 2.5, 2.6 and 2.7 all sit behind the same Alibaba DashScope async task API:
submit to one video-synthesis route with ``X-DashScope-Async`` set, then poll
``/tasks/<id>``. Only the request bodies differ between versions.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from ..api_video_client import REQUEST_TIMEOUT, ApiVideoSynthesis

#: Status advertised to the registry: these adapters call an external service.
RUNTIME_STATUS = {
    "runtime_mode": "api",
    "backend_stage": "external_service",
    "in_tree_backend": False,
    "external_service": True,
}


class DashScopeVideoSynthesis(ApiVideoSynthesis):
    """Async task client for DashScope's video-synthesis endpoint."""

    DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1"

    #: Route the endpoint is expected to expose, appended when absent.
    TASK_PATH: ClassVar[str] = "/services/aigc/video-generation/video-synthesis"

    RUNTIME_STATUS = RUNTIME_STATUS
    IN_TREE_BACKEND = False
    BACKEND_STAGE = "external_service"
    EXTERNAL_SERVICE = True

    def _headers(self, async_request: bool = True) -> Dict[str, str]:
        """Return request headers, opting into async task submission when asked."""
        headers = super()._headers()
        if async_request:
            headers["X-DashScope-Async"] = "enable"
        return headers

    def _create_url(self) -> str:
        """Return the submission URL, tolerating an endpoint that already names it."""
        if self.endpoint.endswith(self.TASK_PATH):
            return self.endpoint
        return f"{self.endpoint}{self.TASK_PATH}"

    def _task_status_url(self, task_id: str) -> str:
        """Return the polling URL for ``task_id``."""
        base_endpoint = self.endpoint
        if base_endpoint.endswith(self.TASK_PATH):
            base_endpoint = base_endpoint[: -len(self.TASK_PATH)]
        return f"{base_endpoint}/tasks/{task_id}"

    def _post_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an async generation task and return the decoded response."""
        response = self._requests().post(
            self._create_url(),
            headers=self._headers(async_request=True),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Return the current status and results of a submitted task."""
        response = self._requests().get(
            self._task_status_url(task_id),
            headers=self._headers(async_request=False),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()


__all__ = ["RUNTIME_STATUS", "DashScopeVideoSynthesis"]
