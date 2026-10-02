"""Gradio import guards and runtime patches for WorldFoundry Studio."""

from __future__ import annotations

import inspect
import os
import time
import warnings
from typing import Any

import httpx

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def mask_socks_proxy_env_for_gradio() -> dict[str, str]:
    """Avoid Gradio's local httpx probes failing when socksio is absent."""

    masked: dict[str, str] = {}
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.environ.get(name)
        if value and value.strip().lower().startswith(
            ("socks://", "socks4://", "socks4h://", "socks5://", "socks5h://")
        ):
            masked[name] = value
            os.environ.pop(name, None)
    return masked


_GRADIO_IMPORT_PROXY_ENV = mask_socks_proxy_env_for_gradio()
try:
    import gradio as gr
finally:
    os.environ.update(_GRADIO_IMPORT_PROXY_ENV)

_ORIGINAL_TEMPLATE_RESPONSE: Any = None
_ORIGINAL_URL_OK: Any = None
_ORIGINAL_GET_API_INFO: Any = None


def _install_template_response_guard() -> None:
    """Route Gradio 4.14's older TemplateResponse call through Starlette >= 1.0."""

    try:
        import gradio.routes as gr_routes
    except Exception:
        return

    templates = getattr(gr_routes, "templates", None)
    if templates is None or getattr(templates, "_worldfoundry_template_response_guard", False):
        return

    template_response = getattr(templates, "TemplateResponse", None)
    if template_response is None:
        return

    try:
        params = list(inspect.signature(template_response).parameters)
    except (TypeError, ValueError):
        return

    if not params or params[0] != "request":
        return

    global _ORIGINAL_TEMPLATE_RESPONSE
    original = template_response
    _ORIGINAL_TEMPLATE_RESPONSE = template_response

    def guarded_template_response(*args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.get("context")
            request = context.get("request") if isinstance(context, dict) else None
            if request is None:
                raise TypeError(
                    "Gradio older TemplateResponse call requires context['request'] "
                    "when running with Starlette >= 1.0."
                )
            return original(request, name, context, *args[2:], **kwargs)
        return original(*args, **kwargs)

    templates.TemplateResponse = guarded_template_response
    templates._worldfoundry_template_response_guard = True


def _install_proxy_safe_url_check() -> None:
    """Make Gradio's localhost readiness probe ignore shell proxy settings."""

    try:
        import gradio.networking as gr_networking
    except Exception:
        return

    if getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False):
        return

    global _ORIGINAL_URL_OK
    _ORIGINAL_URL_OK = getattr(gr_networking, "url_ok", None)

    def proxy_safe_url_ok(url: str) -> bool:
        try:
            with httpx.Client(timeout=3, verify=False, trust_env=False) as client:
                for _ in range(5):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        response = client.head(url)
                    if response.status_code in (200, 401, 302):
                        return True
                    time.sleep(0.5)
        except Exception:
            return False
        return False

    gr_networking.url_ok = proxy_safe_url_ok
    gr_networking._worldfoundry_proxy_safe_url_ok = True


def _install_api_info_guard() -> None:
    """Avoid Gradio API schema crashes on components that emit bool schemas."""

    try:
        import gradio.blocks as gr_blocks
    except Exception:
        return

    blocks_cls = getattr(gr_blocks, "Blocks", None)
    if blocks_cls is None or getattr(blocks_cls, "_worldfoundry_api_info_guard", False):
        return

    original = getattr(blocks_cls, "get_api_info", None)
    if original is None:
        return

    global _ORIGINAL_GET_API_INFO
    _ORIGINAL_GET_API_INFO = original

    def guarded_get_api_info(self, all_endpoints: bool = False):
        try:
            return original(self, all_endpoints=all_endpoints)
        except TypeError as exc:
            if "all_endpoints" in str(exc) and "unexpected keyword" in str(exc):
                try:
                    return original(self)
                except TypeError as nested_exc:
                    exc = nested_exc
            error_text = str(exc)
            if "bool" not in error_text or "iterable" not in error_text:
                raise
            return {"named_endpoints": {}, "unnamed_endpoints": {}}

    blocks_cls.get_api_info = guarded_get_api_info
    blocks_cls._worldfoundry_api_info_guard = True


def install_gradio_patches() -> None:
    """Install Gradio compatibility patches. Idempotent; call from Studio entrypoints."""

    _install_template_response_guard()
    _install_proxy_safe_url_check()
    _install_api_info_guard()


def uninstall_gradio_patches() -> None:
    """Restore Gradio callables replaced by install_gradio_patches()."""

    global _ORIGINAL_TEMPLATE_RESPONSE, _ORIGINAL_URL_OK, _ORIGINAL_GET_API_INFO

    try:
        import gradio.routes as gr_routes
    except Exception:
        gr_routes = None
    templates = getattr(gr_routes, "templates", None) if gr_routes is not None else None
    if templates is not None and getattr(templates, "_worldfoundry_template_response_guard", False):
        if _ORIGINAL_TEMPLATE_RESPONSE is not None:
            templates.TemplateResponse = _ORIGINAL_TEMPLATE_RESPONSE
        templates._worldfoundry_template_response_guard = False
    _ORIGINAL_TEMPLATE_RESPONSE = None

    try:
        import gradio.networking as gr_networking
    except Exception:
        gr_networking = None
    if gr_networking is not None and getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False):
        if _ORIGINAL_URL_OK is not None:
            gr_networking.url_ok = _ORIGINAL_URL_OK
        gr_networking._worldfoundry_proxy_safe_url_ok = False
    _ORIGINAL_URL_OK = None

    try:
        import gradio.blocks as gr_blocks
    except Exception:
        gr_blocks = None
    blocks_cls = getattr(gr_blocks, "Blocks", None) if gr_blocks is not None else None
    if blocks_cls is not None and getattr(blocks_cls, "_worldfoundry_api_info_guard", False):
        if _ORIGINAL_GET_API_INFO is not None:
            blocks_cls.get_api_info = _ORIGINAL_GET_API_INFO
        blocks_cls._worldfoundry_api_info_guard = False
    _ORIGINAL_GET_API_INFO = None


def gradio_progress(*, track_tqdm: bool = False) -> Any:
    progress_cls = getattr(gr, "Progress", None)
    if callable(progress_cls):
        return progress_cls(track_tqdm=track_tqdm)
    return None
