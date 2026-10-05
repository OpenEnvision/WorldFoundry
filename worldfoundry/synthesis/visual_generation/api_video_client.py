"""Shared client plumbing for hosted video-generation APIs.

The hosted backends differ only in their endpoint, their auth header and the
routes they call. URL joining, JSON request/response handling, streamed artifact
download and credential-based construction are the same everywhere, so they live
here instead of once per vendor.

:class:`CredentialedSynthesis`
    Endpoint and credential handling alone, for backends that talk to their
    vendor through an SDK rather than raw HTTP.

:class:`ApiVideoSynthesis`
    REST backends driven through ``requests`` (Kling, Luma, MiniMax, Runway,
    World Labs, DashScope).

:class:`OpenAiVideoSynthesis`
    Backends reached through an OpenAI-compatible SDK client (Sora).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional

from ..base_synthesis import BaseSynthesis

#: Every hosted call is a long-running generation request; fail rather than hang.
REQUEST_TIMEOUT = 300

#: Artifacts are large enough to stream, small enough that 1 MiB chunks are fine.
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class CredentialedSynthesis(BaseSynthesis):
    """Endpoint and credential handling shared by every hosted backend.

    Inherits :class:`~worldfoundry.synthesis.base_synthesis.BaseSynthesis` so
    hosted API backends (Kling, Luma, MiniMax, Runway, Sora, Veo, ...) take part
    in ``isinstance``-based dispatch alongside local models. Contract mapping:

    - ``api_init`` is overridden as a *classmethod factory* returning a
      configured instance (the base declares an instance method); every
      pipeline call site uses ``Vendor.api_init(endpoint=..., api_key=...)``.
    - ``from_pretrained`` keeps the base ``NotImplementedError``: hosted
      backends have no local weights to load.
    - ``predict`` is implemented by each vendor adapter.
    """

    #: Endpoint used when the caller does not name one.
    DEFAULT_ENDPOINT: ClassVar[str] = ""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: str = "your_api_key",
        logger=None,
    ) -> None:
        """Store the endpoint, credential and optional logger."""
        super().__init__()
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.api_key = api_key
        self.logger = logger

    @classmethod
    def api_init(cls, *args: Any, **kwargs: Any):
        """Construct the client, dropping options this backend does not accept.

        Pipeline loaders pass one common option bundle to every backend, so the
        keys a given vendor does not understand are filtered out here rather
        than being re-declared in each adapter's signature. The constructor
        stays the single place a backend's defaults are written down.

        Unlike ``BaseSynthesis.api_init`` (an instance method), this is a
        factory: it returns a new configured instance.
        """
        accepted = inspect.signature(cls).parameters
        takes_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in accepted.values()
        )
        if takes_var_kwargs:
            return cls(*args, **kwargs)
        return cls(*args, **{key: value for key, value in kwargs.items() if key in accepted})


class ApiVideoSynthesis(CredentialedSynthesis):
    """REST client for a hosted video-generation API.

    Subclasses set :attr:`DEFAULT_ENDPOINT` and, when their auth scheme differs
    from a bearer token, override :meth:`_headers`.
    """

    #: Header carrying the credential, and the template rendering its value.
    AUTH_HEADER: ClassVar[str] = "Authorization"
    AUTH_VALUE: ClassVar[str] = "Bearer {api_key}"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: str = "your_api_key",
        logger=None,
    ) -> None:
        """Drop any trailing slash so routes join onto the endpoint cleanly."""
        super().__init__(endpoint=endpoint, api_key=api_key, logger=logger)
        self.endpoint = self.endpoint.rstrip("/")

    @staticmethod
    def _requests():
        """Import ``requests`` on first call so importing an adapter stays cheap."""
        import requests

        return requests

    def _auth_headers(self) -> Dict[str, str]:
        """Return the credential header alone, for routes that take no body."""
        return {self.AUTH_HEADER: self.AUTH_VALUE.format(api_key=self.api_key)}

    def _headers(self) -> Dict[str, str]:
        """Return the auth and content-type headers for a JSON request."""
        return {**self._auth_headers(), "Content-Type": "application/json"}

    def _url(self, route: str) -> str:
        """Join ``route`` onto the endpoint without doubling the separator."""
        return f"{self.endpoint}/{route.lstrip('/')}"

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``payload`` as JSON and return the decoded response."""
        response = self._requests().post(
            self._url(route),
            headers=self._headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _get(
        self,
        route: str,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """GET ``route`` and return the decoded response.

        Args:
            route: Route to append to the endpoint.
            params: Query string parameters.
            headers: Replaces the default headers; pass :meth:`_auth_headers`
                for backends that reject a content type on a bodyless request.
        """
        response = self._requests().get(
            self._url(route),
            headers=dict(headers) if headers is not None else self._headers(),
            params=dict(params) if params else None,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def download_video(
        self,
        video_url: str,
        save_path: str,
        chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    ) -> str:
        """Stream the artifact at ``video_url`` to ``save_path`` and return the path."""
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._requests().get(video_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # skip keep-alive chunks
                        file.write(chunk)
        return str(output_path)


class OpenAiVideoSynthesis(CredentialedSynthesis):
    """Client for a video backend reached through an OpenAI-compatible SDK."""

    def __init__(self, endpoint: Optional[str] = None, api_key: str = "your_api_key", logger=None) -> None:
        """Open the SDK client against ``endpoint``, defaulting the logger by module."""
        super().__init__(endpoint=endpoint, api_key=api_key, logger=logger)
        if self.logger is None:
            self.logger = logging.getLogger(type(self).__module__)
        self.client = self._openai_client(api_key=self.api_key, base_url=self.endpoint)

    @staticmethod
    def _openai_client(api_key: str, base_url: str):
        """Create the OpenAI client, importing the SDK only once it is needed."""
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url)


__all__ = [
    "DOWNLOAD_CHUNK_SIZE",
    "REQUEST_TIMEOUT",
    "ApiVideoSynthesis",
    "CredentialedSynthesis",
    "OpenAiVideoSynthesis",
]
