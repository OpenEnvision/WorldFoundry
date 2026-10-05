"""Serving primitives shared by WorldFoundry Studio frontends."""

from .auth import (
    AUTH_TOKEN_ENV,
    configured_auth_token,
    is_loopback_host,
    request_token_valid,
    require_auth_token_for_host,
    token_matches,
)
from .http import (
    StudioThreadingHTTPServer,
    parse_byte_range,
    path_allowed,
    send_file_response,
    send_json_response,
    send_text_response,
)
from .network import bind_security_warning, get_external_ip, public_url_for_bind
from .telemetry import StudioServiceTelemetry

__all__ = [
    "AUTH_TOKEN_ENV",
    "StudioServiceTelemetry",
    "StudioThreadingHTTPServer",
    "bind_security_warning",
    "configured_auth_token",
    "get_external_ip",
    "is_loopback_host",
    "parse_byte_range",
    "path_allowed",
    "public_url_for_bind",
    "request_token_valid",
    "require_auth_token_for_host",
    "send_file_response",
    "send_json_response",
    "send_text_response",
    "token_matches",
]
