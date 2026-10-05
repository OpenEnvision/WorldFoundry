from __future__ import annotations

import socket


def get_external_ip() -> str:
    """Return the outward-facing local IP address for this host."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def public_url_for_bind(host: str, port: int) -> str:
    """Return a useful browser URL for a service bind address."""

    visible_host = get_external_ip() if host in {"", "0.0.0.0", "::"} else host
    return f"http://{visible_host}:{int(port)}"


def bind_security_warning(host: str) -> str | None:
    """Return a prominent warning when a bind host is reachable from the network.

    Callers should print this next to any advertised public URL. Returns None
    for loopback binds.
    """

    from .auth import AUTH_TOKEN_ENV, configured_auth_token, is_loopback_host

    if is_loopback_host(host):
        return None
    token_state = (
        "token auth is enabled"
        if configured_auth_token()
        else f"{AUTH_TOKEN_ENV} is NOT set"
    )
    return (
        f"SECURITY WARNING: binding {host!r} exposes this Studio server to the "
        f"network ({token_state}). Anyone who can reach this address can use "
        "its endpoints. Prefer --host 127.0.0.1 plus an SSH tunnel."
    )
