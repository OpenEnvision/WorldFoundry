"""API-based runtime configuration and management helpers for remote pipelines."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any


_PLACEHOLDER_API_KEYS = {"", "your_api_key", "your api key"}
_PIPELINE_LOADER_METADATA_KEYS = {
    "model_id",
    "pipeline_binding",
    "profile_id",
    "runtime_profile",
    "variant_id",
}


def resolve_api_key(api_key: str | None, env_names: Sequence[str], service_name: str) -> str:
    """Resolve an API key from explicit input or documented environment variables.

    Args:
        api_key: API key passed by the caller.
        env_names: Environment variable names checked in priority order.
        service_name: Human-readable service name used in error messages.
    """

    value = (api_key or "").strip()
    if value and value not in _PLACEHOLDER_API_KEYS:
        return value
    for env_name in env_names:
        # Attempt to retrieve from environment variables as a fallback resolution
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    joined = "/".join(env_names)
    raise ValueError(f"{service_name} API key is required. Pass api_key or set {joined}.")


def load_api_pipeline_from_pretrained(
    pipeline_cls: type,
    *,
    model_path: Any = None,
    required_components: Mapping[str, Any] | None = None,
    device: str = "cuda",
    model_id: str | None = None,
    default_endpoint: str,
    service_name: str,
    **kwargs: Any,
) -> Any:
    """Adapt the unified pipeline loader contract to an API pipeline.

    Evaluation resolves all pipelines through ``from_pretrained`` even when a
    hosted provider has no checkpoint.  API-only pipelines use the mapping as
    connection options and delegate to their existing ``api_init`` factory.
    Resolver metadata is removed before provider-specific options are passed
    through.
    """

    del device, model_id
    options: dict[str, Any] = {}
    if isinstance(model_path, Mapping):
        options.update(model_path)
    elif model_path is not None:
        raise ValueError(f"{service_name} is API-only; pass API options as a mapping, not a checkpoint path.")
    options.update(required_components or {})
    options.update(kwargs)
    for key in _PIPELINE_LOADER_METADATA_KEYS:
        options.pop(key, None)
    endpoint = options.pop("endpoint", None) or default_endpoint
    api_key = options.pop("api_key", None)
    logger = options.pop("logger", None)
    return pipeline_cls.api_init(endpoint=endpoint, api_key=api_key, logger=logger, **options)
