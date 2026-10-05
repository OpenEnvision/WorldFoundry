"""Shared bind-address safety and token authentication for Studio servers.

Every Studio frontend defaults to loopback and stays unauthenticated there.
When an operator binds a non-loopback address, each server stack must require
the shared bearer token from ``WORLDFOUNDRY_STUDIO_AUTH_TOKEN`` (sent as
``Authorization: Bearer <token>`` or ``?token=<token>``) or refuse to start.
"""

from __future__ import annotations

import hmac
import ipaddress
import os

AUTH_TOKEN_ENV = "WORLDFOUNDRY_STUDIO_AUTH_TOKEN"


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host only accepts local connections.

    Unknown hostnames and bind-all addresses ("", "0.0.0.0", "::") are treated
    as non-loopback so auth requirements fail closed.
    """

    value = (host or "").strip()
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    candidate = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def configured_auth_token() -> str:
    """Return the operator-configured Studio auth token ("" when unset)."""

    return os.getenv(AUTH_TOKEN_ENV, "").strip()


def require_auth_token_for_host(host: str, *, server_name: str = "Studio server") -> str:
    """Return the token that requests must carry for this bind host.

    Loopback binds keep the historical unauthenticated behavior and return "".
    Non-loopback binds require ``WORLDFOUNDRY_STUDIO_AUTH_TOKEN``; when it is
    missing this raises ``SystemExit`` so the server never starts exposed.
    """

    if is_loopback_host(host):
        return ""
    token = configured_auth_token()
    if not token:
        raise SystemExit(
            f"{server_name} refuses to bind non-loopback host {host!r} without authentication. "
            f"Set {AUTH_TOKEN_ENV} to a shared secret (clients must send "
            "'Authorization: Bearer <token>' or append '?token=<token>'), "
            "or bind --host 127.0.0.1 and use an SSH tunnel."
        )
    return token


def token_matches(candidate: str | None, token: str) -> bool:
    """Constant-time comparison of a client-supplied token candidate."""

    if not token or not candidate:
        return False
    return hmac.compare_digest(candidate.strip(), token)


def request_token_valid(
    token: str,
    *,
    authorization_header: str | None,
    query_token: str | None,
) -> bool:
    """Validate ``Authorization: Bearer <token>`` or ``?token=<token>``.

    An empty required ``token`` means auth is disabled and every request passes.
    """

    if not token:
        return True
    header = (authorization_header or "").strip()
    if header.lower().startswith("bearer ") and token_matches(header[7:], token):
        return True
    return token_matches(query_token, token)
